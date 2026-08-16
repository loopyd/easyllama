from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import threading
import time
import types
from typing import Any
import urllib.error
import urllib.request

from docker import from_env
from docker.errors import APIError, DockerException, ImageNotFound
from docker.types import DeviceRequest

from .config import (
    CHAT_TEMPLATE_DIR_CONTAINER,
    LLAMA_SWAP_BIN,
    MMPROJ_DIR_CONTAINER,
    MODELS_DIR_CONTAINER,
    RUNTIME_CONTAINER,
    RUNTIME_HOST,
    Settings,
    container_config_path,
    effective_config_path,
    listen_url,
    load_auth,
    mmproj_arg,
    resolve_ls_config,
    resolved_api_key,
)
from .helpers import ProgressReporter, env_override
from .logger import get_logger
from .servers import mode_def as server_mode_def, mode_defs as server_mode_defs
from .servers.common import hf_file, hf_get, hf_spec

LOGGER = get_logger(__name__)


def _build_summary(settings: Settings, target: str) -> str:
    return server_mode_def(settings.mode).build_summary(
        settings,
        image_name=settings.image_name,
        target=target,
    )


def _stop_proc(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def compute_cuda_architectures(settings: Settings) -> str:
    if settings.cmake_cuda_architectures != "auto":
        return settings.cmake_cuda_architectures
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        LOGGER.warning(
            "nvidia-smi not found or failed; using fallback CUDA arch %s",
            settings.default_cuda_architectures,
        )
        return settings.default_cuda_architectures
    values: set[str] = set()
    for line in result.stdout.splitlines():
        match = re.findall(r"\d+", line)
        if not match:
            continue
        major = match[0]
        minor = match[1][0] if len(match) > 1 and match[1] else "0"
        values.add(f"{major}{minor}")
    if not values:
        LOGGER.warning(
            "failed to detect compute capability; using fallback CUDA arch %s",
            settings.default_cuda_architectures,
        )
        return settings.default_cuda_architectures
    return ";".join(sorted(values, key=int))


def _http_json(
    url: str, *, headers: dict[str, str] | None = None, timeout: float | None = None
) -> dict[str, object]:
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _http_response(
    url: str, *, headers: dict[str, str] | None = None, timeout: float | None = None
) -> tuple[int, str]:
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return response.status, body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return exc.code, body
    except (TimeoutError, urllib.error.URLError) as exc:
        return 0, str(exc)


def _http_status(url: str, *, headers: dict[str, str] | None = None) -> int:
    status, _ = _http_response(url, headers=headers)
    return status


def model_status(
    settings: Settings,
    model_id: str,
    *,
    headers: dict[str, str],
    timeout: float | None = None,
) -> dict[str, object] | None:
    try:
        payload = _http_json(f"{listen_url(settings)}/running", headers=headers, timeout=timeout)
    except (TimeoutError, urllib.error.URLError):
        return None
    running = payload.get("running", [])
    if not isinstance(running, list):
        return None
    for item in running:
        if not isinstance(item, dict):
            continue
        if item.get("model") == model_id:
            return item
    return None


def model_ready(settings: Settings, model_id: str, *, headers: dict[str, str]) -> bool:
    item = model_status(settings, model_id, headers=headers)
    return bool(item and item.get("state") == "ready")


def _warmup_state(item: dict[str, object] | None) -> str:
    if not item:
        return "loading"
    state = str(item.get("state") or "unknown").strip().lower()
    return state or "unknown"


def _update_warmup_progress(
    reporter: ProgressReporter,
    *,
    started_at: float,
    warmup_timeout: int,
    status: int | None,
    item: dict[str, object] | None,
) -> None:
    elapsed = min(int(time.monotonic() - started_at), warmup_timeout)
    reporter.format_args["http_status"] = status if status is not None else "?"
    reporter.format_args["state"] = _warmup_state(item)
    reporter.format_args["elapsed"] = elapsed
    delta = elapsed - int(reporter.downloaded)
    if delta > 0:
        reporter.update(delta)


def _trigger_warmup(
    base_url: str,
    model_id: str,
    *,
    headers: dict[str, str],
    timeout: int,
) -> list[int | str | None]:
    """Start the upstream request without cancelling the model while it loads."""
    result: list[int | str | None] = [None, None]

    def request() -> None:
        result[:] = _http_response(
            f"{base_url}/upstream/{model_id}/health", headers=headers, timeout=timeout
        )

    threading.Thread(target=request, daemon=True).start()
    return result


def _prefetch_models(settings: Settings, model_ids: list[str]) -> None:
    """Download warmup GGUFs on the host so easyllama owns byte progress."""
    import yaml

    config = yaml.safe_load(resolve_ls_config(settings).read_text(encoding="utf-8")) or {}
    macros = config.get("macros", {})
    models = config.get("models", {})
    total_models = len(model_ids)
    for position, model_id in enumerate(model_ids, start=1):
        command = str(models.get(model_id, {}).get("cmd", ""))
        match = re.search(r"(?:^|\s)-hf\s+(\S+)", command)
        if not match:
            continue
        spec = match.group(1)
        for _ in range(len(macros)):
            expanded = re.sub(
                r"\$\{([A-Za-z0-9_-]+)\}",
                lambda item: str(macros.get(item.group(1), item.group(0))),
                spec,
            )
            if expanded == spec:
                break
            spec = expanded
        repo, selector = hf_spec(spec)
        if repo and selector:
            hf_get(
                repo,
                hf_file(repo, selector, model_id, suffixes=(".gguf",)),
                model_id,
                cache_dir=settings.models_dir,
                warmup=f"Warming model {position}/{total_models}: {model_id}",
            )


def warmup_models(settings: Settings, model_ids: list[str]) -> int:
    auth = load_auth(settings)
    if auth.hf_token:
        os.environ["HF_TOKEN"] = auth.hf_token
    api_key = resolved_api_key(settings, auth)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    base_url = listen_url(settings)
    warmup_timeout = int(env_override("WARMUP_TIMEOUT", "1800") or "1800")
    warmup_poll_interval = float(env_override("WARMUP_POLL_INTERVAL", "2") or "2")
    if settings.runtime_mode == RUNTIME_HOST:
        runtime = DockerRuntime(settings)
        runtime.ensure_daemon()
        if not runtime.is_running():
            raise SystemExit(f"container {settings.container_name} is not running; start it first")
    if _http_status(f"{base_url}/health") >= 400:
        raise SystemExit(f"llama-swap is not reachable at {base_url}")
    selected_ids = list(model_ids)
    if not selected_ids:
        payload = _http_json(f"{base_url}/v1/models", headers=headers)
        data = payload.get("data", [])
        if isinstance(data, list):
            selected_ids = [
                item["id"] for item in data if isinstance(item, dict) and item.get("id")
            ]
    if not selected_ids:
        LOGGER.warning("No models selected for warmup")
        return 0
    _prefetch_models(settings, selected_ids)
    failure_states = {"error", "failed", "stopped", "terminated"}
    total_models = len(selected_ids)
    for index, model_id in enumerate(selected_ids, start=1):
        started_at = time.monotonic()
        deadline = started_at + warmup_timeout
        reporter = ProgressReporter(
            model_id,
            total=warmup_timeout,
            log_threshold=max(1, int(warmup_poll_interval)),
            level=logging.INFO,
            start_template="Warming model {position}/{count}: {name}",
            update_template=(
                "Warming model {position}/{count}: {name} "
                "— loading; {elapsed}s elapsed, state={state}"
            ),
            finish_template="Warmed model {position}/{count}: {name} in {downloaded}s",
            format_args={
                "position": index,
                "count": total_models,
                "http_status": "?",
                "state": "loading",
                "elapsed": 0,
            },
        )
        reporter.start()
        # Keep the one trigger request open: cancelling it also cancels a
        # still-loading llama-swap worker. `/running` is the authoritative state.
        trigger_status = _trigger_warmup(
            base_url, model_id, headers=headers, timeout=warmup_timeout
        )
        while time.monotonic() < deadline:
            item = model_status(settings, model_id, headers=headers, timeout=warmup_poll_interval)
            if item and item.get("state") == "ready":
                _update_warmup_progress(
                    reporter,
                    started_at=started_at,
                    warmup_timeout=warmup_timeout,
                    status=(trigger_status[0] if isinstance(trigger_status[0], int) else None),
                    item=item,
                )
                reporter.finish()
                break
            if item:
                state = str(item.get("state", "unknown")).lower()
                error = str(item.get("error") or item.get("message") or "").strip()
                if state in failure_states or error:
                    detail_suffix = f": {error}" if error else ""
                    raise SystemExit(
                        f"failed to warm model {model_id}: state={state}{detail_suffix}"
                    )
            trigger_code = trigger_status[0]
            if item is None and isinstance(trigger_code, int) and trigger_code not in (200, 429):
                detail = str(trigger_status[1] or "").strip()
                detail_suffix = f": {detail}" if detail else ""
                raise SystemExit(
                    f"failed to warm model {model_id}: HTTP {trigger_code}{detail_suffix}"
                )
            _update_warmup_progress(
                reporter,
                started_at=started_at,
                warmup_timeout=warmup_timeout,
                status=(trigger_status[0] if isinstance(trigger_status[0], int) else None),
                item=item,
            )
            time.sleep(warmup_poll_interval)
        else:
            raise SystemExit(
                f"failed to warm model {model_id} within {warmup_timeout}s "
                f"(last HTTP {trigger_status[0]})"
            )
    return 0


class DockerRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client: Any = from_env()
        self.api: Any = self.client.api

    def ensure_daemon(self) -> None:
        try:
            self.client.ping()
        except DockerException as exc:
            raise SystemExit("docker daemon is not reachable (start docker and retry)") from exc

    def ensure_nvidia_runtime(self) -> None:
        runtimes = self.client.info().get("Runtimes", {})
        if "nvidia" not in runtimes:
            raise SystemExit("nvidia container runtime is not available in docker")

    def get_container(self):
        for container in self.client.containers.list(all=True):
            if container.name == self.settings.container_name:
                return container
        return None

    def get_legacy_default_container(self):
        if self.settings.container_name != "easyllama-server-swap" or any(
            name in os.environ for name in ("EASYLLAMA_CONTAINER_NAME",)
        ):
            return None
        for container in self.client.containers.list(all=True):
            if container.name == "llamacpp-server-swap":
                return container
        return None

    def get_running_container_count(self) -> int:
        return sum(
            1
            for container in self.client.containers.list()
            if container.name == self.settings.container_name
        )

    def is_running(self) -> bool:
        container = self.get_container()
        return bool(container and container.status == "running")

    def image_exists(self, image_name: str | None = None) -> bool:
        try:
            self.client.images.get(image_name or self.settings.image_name)
        except ImageNotFound:
            return False
        return True

    def _build_cmd(self, target: str, build_args: dict[str, str]) -> list[str]:
        docker_bin = shutil.which("docker")
        if docker_bin is None:
            raise SystemExit("docker CLI is required for image builds")
        cmd = [
            docker_bin,
            "buildx",
            "build",
            "--load",
            "--progress=plain",
            "--pull",
            "--tag",
            self.settings.image_name,
            "--target",
            target,
            "--file",
            str(self.settings.root_dir / "Dockerfile"),
        ]
        for key, value in build_args.items():
            cmd.extend(("--build-arg", f"{key}={value}"))
        cmd.append(str(self.settings.root_dir))
        return cmd

    def _ensure_buildx(self) -> None:
        docker_bin = shutil.which("docker")
        if docker_bin is None:
            raise SystemExit("docker CLI is required for image builds")
        result = subprocess.run(
            [docker_bin, "buildx", "inspect", "--bootstrap"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            message = "docker buildx with BuildKit is required for image builds"
            if detail:
                message = f"{message}: {detail}"
            raise SystemExit(message)

    def build_image(self) -> int:
        self.ensure_daemon()
        self._ensure_buildx()
        mode_metadata = server_mode_def(self.settings.mode)
        target = mode_metadata.docker_target
        build_args = {
            "BUILD_MODE": self.settings.mode,
            "DEBIAN_FRONTEND": "noninteractive",
            "HOST_TZ": self.settings.host_tz,
            "HOST_LANG": self.settings.host_lang,
            "HOST_LC_ALL": self.settings.host_lc_all,
            "CMAKE_CUDA_ARCHITECTURES": compute_cuda_architectures(self.settings),
        }
        build_args.update(mode_metadata.build_args(self.settings))
        LOGGER.info(_build_summary(self.settings, target))
        proc = subprocess.Popen(
            self._build_cmd(target, build_args),
            cwd=str(self.settings.root_dir),
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            message = line.rstrip()
            if message:
                LOGGER.debug(message)
        if proc.wait() != 0:
            raise SystemExit(f"docker buildx build failed for {self.settings.image_name}")
        LOGGER.info("build complete: %s", self.settings.image_name)
        return 0

    def _remove_container(self, container) -> None:
        try:
            if container.status == "running":
                container.stop(timeout=10)
            container.remove()
        except APIError:
            container.remove(force=True)
        LOGGER.info("removed container %s", container.name)

    def remove_container(self) -> None:
        container = self.get_container()
        if container is not None:
            self._remove_container(container)
        legacy = self.get_legacy_default_container()
        if legacy is not None:
            LOGGER.info("migrating legacy default container to easyllama project naming")
            self._remove_container(legacy)
        if container is None and legacy is None:
            LOGGER.warning("container %s does not exist", self.settings.container_name)

    def run_container(self) -> int:
        self.ensure_daemon()
        self.ensure_nvidia_runtime()
        auth = load_auth(self.settings)
        self.settings.models_dir.mkdir(parents=True, exist_ok=True)
        self.settings.mmproj_dir.mkdir(parents=True, exist_ok=True)

        running_count = self.get_running_container_count()
        if running_count > 1:
            raise SystemExit(
                f"refusing to start {self.settings.container_name}: found "
                f"{running_count} running containers with same name"
            )

        legacy = self.get_legacy_default_container()
        if legacy is not None:
            LOGGER.info("migrating legacy default container to easyllama project naming")
            self._remove_container(legacy)

        container = self.get_container()
        if container is not None and container.status == "running":
            running_mode = container.labels.get("easyllama.mode", "unknown")
            LOGGER.warning(
                "container %s is already running in %s mode; use restart to replace it",
                self.settings.container_name,
                running_mode,
            )
            return 0
        if container is not None:
            self.remove_container()

        if not self.image_exists():
            LOGGER.info("image %s is missing; building it first", self.settings.image_name)
            self.build_image()

        config_path, container_config_path_value = effective_config_path(self.settings, auth)
        mmproj_argument = mmproj_arg(self.settings, auth)
        volumes = {
            str(self.settings.models_dir): {"bind": MODELS_DIR_CONTAINER, "mode": "rw"},
            str(self.settings.mmproj_dir): {"bind": MMPROJ_DIR_CONTAINER, "mode": "rw"},
            str(config_path): {"bind": container_config_path_value, "mode": "ro"},
        }
        if self.settings.chat_template_dir.is_dir():
            volumes[str(self.settings.chat_template_dir)] = {
                "bind": CHAT_TEMPLATE_DIR_CONTAINER,
                "mode": "ro",
            }
        if Path("/etc/localtime").is_file():
            volumes["/etc/localtime"] = {"bind": "/etc/localtime", "mode": "ro"}
        if Path("/etc/timezone").is_file():
            volumes["/etc/timezone"] = {"bind": "/etc/timezone", "mode": "ro"}

        environment = {
            "EASYLLAMA_RUNTIME_MODE": RUNTIME_CONTAINER,
            "EASYLLAMA_MODE": self.settings.mode,
            "CONTAINER_PORT": str(self.settings.container_port),
            "EASYLLAMA_MMPROJ_ARG": mmproj_argument,
            "TZ": self.settings.host_tz,
            "LANG": self.settings.host_lang,
            "LC_ALL": self.settings.host_lc_all,
            "EASYLLAMA_ROOT": "/app",
        }
        if auth.hf_token:
            environment["HF_TOKEN"] = auth.hf_token

        backend = server_mode_def(self.settings.mode).backend
        self.client.containers.run(
            self.settings.image_name,
            command=["serve"],
            detach=True,
            init=True,
            name=self.settings.container_name,
            restart_policy={"Name": "unless-stopped"},
            security_opt=["no-new-privileges"],
            pids_limit=4096 if backend == "vllm" else self.settings.pids_limit,
            runtime="nvidia",
            device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
            ports={f"{self.settings.container_port}/tcp": ("127.0.0.1", self.settings.host_port)},
            volumes=volumes,
            environment=environment,
            labels={
                "easyllama.mode": self.settings.mode,
                "easyllama.backend": backend,
                "easyllama.image": self.settings.image_name,
            },
            **({"shm_size": "8g"} if backend == "vllm" else {}),
        )
        LOGGER.info(
            "started %s (%s mode) on http://localhost:%s",
            self.settings.container_name,
            self.settings.mode,
            self.settings.host_port,
        )
        return 0

    def _remove_effective_configs(self) -> None:
        if not self.settings.runtime_dir.is_dir():
            return
        for path in self.settings.runtime_dir.glob("*.effective.yaml"):
            path.unlink(missing_ok=True)

    def stop_container(self) -> int:
        self.ensure_daemon()
        self.remove_container()
        self._remove_effective_configs()
        return 0

    def restart_container(self) -> int:
        self.stop_container()
        return self.run_container()

    def print_logs(self, *, tail: int | None = None) -> int:
        self.ensure_daemon()
        container = self.get_container()
        if container is None:
            raise SystemExit(f"container {self.settings.container_name} does not exist")
        if tail is not None:
            print(container.logs(tail=tail).decode("utf-8", errors="replace"), end="")
            return 0
        try:
            for chunk in container.logs(stream=True, follow=True):
                print(chunk.decode("utf-8", errors="replace"), end="")
        except KeyboardInterrupt:
            LOGGER.info("log follow interrupted")
            return 0
        return 0

    def status(self) -> int:
        self.ensure_daemon()
        container = self.get_container()
        if container is None:
            LOGGER.info("container %s is not present", self.settings.container_name)
        else:
            image = container.image
            image_name = image.tags[0] if image and image.tags else "<untagged>"
            LOGGER.info(
                "container %s status=%s image=%s", container.name, container.status, image_name
            )
        available_images = []
        for mode_metadata in server_mode_defs():
            mode_settings = self.settings.with_mode(mode_metadata.mode)
            if self.image_exists(mode_settings.image_name):
                available_images.append(mode_settings.image_name)
        if available_images:
            LOGGER.info("available mode images: %s", ", ".join(available_images))
        return 0

    def clean(self, *, all_images: bool = False) -> int:
        self.ensure_daemon()
        container = self.get_container()
        if container is not None:
            self.remove_container()
        self._remove_effective_configs()
        image_names = [self.settings.image_name]
        if all_images:
            image_names = [
                self.settings.with_mode(mode_metadata.mode).image_name
                for mode_metadata in server_mode_defs()
            ]
        for image_name in image_names:
            try:
                self.client.images.remove(image_name, force=True)
                LOGGER.info("removed image %s", image_name)
            except ImageNotFound:
                LOGGER.warning("image %s does not exist", image_name)
        return 0


def serve(settings: Settings) -> int:
    config_path = container_config_path(settings)
    llama_swap_bin = Path(LLAMA_SWAP_BIN)
    if not llama_swap_bin.is_file():
        raise SystemExit(f"llama-swap binary not found at {llama_swap_bin}")
    LOGGER.info(
        "starting llama-swap (%s mode, config=%s, listen=:%s)",
        settings.mode,
        config_path,
        settings.container_port,
    )
    proc = subprocess.Popen(
        [
            str(llama_swap_bin),
            "-config",
            str(config_path),
            "-listen",
            f"0.0.0.0:{settings.container_port}",
        ],
        start_new_session=True,
    )
    saved: dict[signal.Signals, object] = {}

    def handle(signum: int, _frame: types.FrameType | None) -> None:
        LOGGER.info("received signal %s, stopping llama-swap", signum)
        _stop_proc(proc)

    for sig in (signal.SIGINT, signal.SIGTERM):
        saved[sig] = signal.getsignal(sig)
        signal.signal(sig, handle)

    try:
        return proc.wait()
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)  # pyright: ignore[reportArgumentType]
        _stop_proc(proc)
