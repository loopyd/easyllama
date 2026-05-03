from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request

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
    resolved_api_key,
)
from .logger import get_logger
from .servers import mode_def as server_mode_def, mode_defs as server_mode_defs

LOGGER = get_logger(__name__)


def _build_summary(settings: Settings, target: str) -> str:
    return server_mode_def(settings.mode).build_summary(
        settings,
        image_name=settings.image_name,
        target=target,
    )


def _stop_proc(proc: subprocess.Popen[str]) -> None:
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


def _http_json(url: str, *, headers: dict[str, str] | None = None) -> dict[str, object]:
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def _http_response(url: str, *, headers: dict[str, str] | None = None) -> tuple[int, str]:
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request) as response:
            body = response.read().decode("utf-8", errors="replace")
            return response.status, body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return exc.code, body


def _http_status(url: str, *, headers: dict[str, str] | None = None) -> int:
    status, _ = _http_response(url, headers=headers)
    return status


def model_status(
    settings: Settings, model_id: str, *, headers: dict[str, str]
) -> dict[str, object] | None:
    payload = _http_json(f"{listen_url(settings)}/running", headers=headers)
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


def warmup_models(settings: Settings, model_ids: list[str]) -> int:
    auth = load_auth(settings)
    api_key = resolved_api_key(settings, auth)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    base_url = listen_url(settings)
    warmup_timeout = int(os.environ.get("LLAMACPP_WARMUP_TIMEOUT", "1800"))
    warmup_poll_interval = float(os.environ.get("LLAMACPP_WARMUP_POLL_INTERVAL", "2"))
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
    failure_states = {"error", "failed", "stopped", "terminated"}
    for model_id in selected_ids:
        LOGGER.info("Warming model %s", model_id)
        deadline = time.monotonic() + warmup_timeout
        last_status: int | None = None
        while time.monotonic() < deadline:
            status, detail = _http_response(
                f"{base_url}/upstream/{model_id}/health", headers=headers
            )
            last_status = status
            if 200 <= status < 400:
                LOGGER.info("Warmed %s", model_id)
                break
            item = model_status(settings, model_id, headers=headers)
            if item and item.get("state") == "ready":
                LOGGER.warning(
                    "warmup health request returned HTTP %s for %s, "
                    "but the model is ready; treating it as warmed",
                    status,
                    model_id,
                )
                break
            if item:
                state = str(item.get("state", "unknown")).lower()
                error = str(item.get("error") or item.get("message") or "").strip()
                if state in failure_states or error:
                    detail_suffix = f": {error}" if error else ""
                    raise SystemExit(
                        f"failed to warm model {model_id}: state={state}{detail_suffix}"
                    )
            detail = detail.strip()
            if status >= 500 and detail:
                raise SystemExit(
                    f"failed to warm model {model_id}: upstream health returned "
                    f"HTTP {status}: {detail}"
                )
            LOGGER.debug(
                "model %s is not ready yet (HTTP %s); waiting %.1fs before retrying",
                model_id,
                status,
                warmup_poll_interval,
            )
            time.sleep(warmup_poll_interval)
        else:
            raise SystemExit(
                f"failed to warm model {model_id} within {warmup_timeout}s "
                f"(last HTTP {last_status})"
            )
    return 0


class DockerRuntime:
    def __init__(self, settings: Settings) -> None:
        import docker

        self.settings = settings
        self.client = docker.from_env()
        self.api = self.client.api
        self.docker = docker

    def ensure_daemon(self) -> None:
        try:
            self.client.ping()
        except self.docker.errors.DockerException as exc:
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

    def is_running(self) -> bool:
        container = self.get_container()
        return bool(container and container.status == "running")

    def image_exists(self, image_name: str | None = None) -> bool:
        try:
            self.client.images.get(image_name or self.settings.image_name)
        except self.docker.errors.ImageNotFound:
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

    def remove_container(self) -> None:
        container = self.get_container()
        if container is None:
            LOGGER.warning("container %s does not exist", self.settings.container_name)
            return
        try:
            if container.status == "running":
                container.stop(timeout=10)
            container.remove()
        except self.docker.errors.APIError:
            container.remove(force=True)
        LOGGER.info("removed container %s", self.settings.container_name)

    def run_container(self) -> int:
        self.ensure_daemon()
        self.ensure_nvidia_runtime()
        auth = load_auth(self.settings)
        self.settings.models_dir.mkdir(parents=True, exist_ok=True)
        self.settings.mmproj_dir.mkdir(parents=True, exist_ok=True)

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
            "LLAMACPP_RUNTIME_MODE": RUNTIME_CONTAINER,
            "LLAMACPP_MODE": self.settings.mode,
            "CONTAINER_PORT": str(self.settings.container_port),
            "LLAMACPP_MMPROJ_ARG": mmproj_argument,
            "TZ": self.settings.host_tz,
            "LANG": self.settings.host_lang,
            "LC_ALL": self.settings.host_lc_all,
            "EASYLLAMA_ROOT": "/app",
        }
        if auth.hf_token:
            environment["HF_TOKEN"] = auth.hf_token

        self.client.containers.run(
            self.settings.image_name,
            command=["serve"],
            detach=True,
            init=True,
            name=self.settings.container_name,
            restart_policy={"Name": "unless-stopped"},
            security_opt=["no-new-privileges"],
            runtime="nvidia",
            device_requests=[self.docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])],
            ports={f"{self.settings.container_port}/tcp": self.settings.host_port},
            volumes=volumes,
            environment=environment,
            labels={
                "easyllama.mode": self.settings.mode,
                "easyllama.image": self.settings.image_name,
            },
        )
        LOGGER.info(
            "started %s (%s mode) on http://localhost:%s",
            self.settings.container_name,
            self.settings.mode,
            self.settings.host_port,
        )
        return 0

    def stop_container(self) -> int:
        self.ensure_daemon()
        self.remove_container()
        return 0

    def restart_container(self) -> int:
        self.stop_container()
        return self.run_container()

    def print_logs(self) -> int:
        self.ensure_daemon()
        container = self.get_container()
        if container is None:
            raise SystemExit(f"container {self.settings.container_name} does not exist")
        for chunk in container.logs(stream=True, follow=True):
            print(chunk.decode("utf-8", errors="replace"), end="")
        return 0

    def status(self) -> int:
        self.ensure_daemon()
        container = self.get_container()
        if container is None:
            LOGGER.info("container %s is not present", self.settings.container_name)
        else:
            LOGGER.info(
                "container %s status=%s image=%s",
                container.name,
                container.status,
                container.image.tags[0] if container.image.tags else "<untagged>",
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
            except self.docker.errors.ImageNotFound:
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

    def handle(signum: int, _frame: object) -> None:
        LOGGER.info("received signal %s, stopping llama-swap", signum)
        _stop_proc(proc)

    for sig in (signal.SIGINT, signal.SIGTERM):
        saved[sig] = signal.getsignal(sig)
        signal.signal(sig, handle)

    try:
        return proc.wait()
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)
        _stop_proc(proc)
