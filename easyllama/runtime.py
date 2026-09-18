"""Warm configured models and run llama-swap in a container."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
import types

from .config import CONTAINERPATH, RUNTIME, Config
from .helpers.hf import HuggingFace
from .helpers.http import Http
from .helpers.logger import LOG as APP_LOG
from .helpers.progress import ProgressReporter

LOGGER = APP_LOG.get(__name__)


def _stop_proc(proc: subprocess.Popen) -> None:
    """Perform the internal stop proc operation.

    Args:
        proc: The proc."""
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


def model_status(
    settings: Config,
    model_id: str,
    *,
    headers: dict[str, str],
    timeout: float | None = None,
) -> dict[str, object] | None:
    """Return a model entry from llama-swap running state.

    Args:
        settings: The settings.
        model_id: The model id.
        headers: The headers.
        timeout: The timeout.

    Returns:
        dict[str, object] | None: The model status result."""
    try:
        payload = Http(settings.listen_url(), headers=headers, timeout=timeout).at("running").json()
    except OSError:
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


def _update_warmup_progress(
    reporter: ProgressReporter,
    *,
    started_at: float,
    warmup_timeout: int,
    status: int | None,
    item: dict[str, object] | None,
) -> None:
    """Perform the internal update warmup progress operation.

    Args:
        reporter: The reporter.
        started_at: The started at.
        warmup_timeout: The warmup timeout.
        status: The status.
        item: The item."""
    elapsed = min(int(time.monotonic() - started_at), warmup_timeout)
    reporter.format_args["http_status"] = status if status is not None else "?"
    state = str(item.get("state") or "unknown").strip().lower() if item else "loading"
    reporter.format_args["state"] = state or "unknown"
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
    """Perform the internal trigger warmup operation.

    Args:
        base_url: The base url.
        model_id: The model id.
        headers: The headers.
        timeout: The timeout.

    Returns:
        list[int | str | None]: The trigger warmup result."""
    result: list[int | str | None] = [None, None]

    def request() -> None:
        """Build a standard-library request for the bound URL."""
        result[:] = (
            Http(base_url, headers=headers, timeout=timeout)
            .at(f"upstream/{model_id}/health")
            .response()
        )

    threading.Thread(target=request, daemon=True).start()
    return result


HUB_CACHE_PREFIX = CONTAINERPATH.MODELS


def _expand_macros(text: str, macros: dict[str, object]) -> str:
    """Expand ${MACRO} references until the text is stable.

    Args:
        text: The text.
        macros: The macros.

    Returns:
        str: The expanded text result."""
    for _ in range(len(macros)):
        expanded = re.sub(
            r"\$\{([A-Za-z0-9_-]+)\}",
            lambda item: str(macros.get(item.group(1), item.group(0))),
            text,
        )
        if expanded == text:
            break
        text = expanded
    return text


def _model_file_refs(command: str) -> list[str]:
    """Return every --model argument value from an expanded server command.

    Args:
        command: The command.

    Returns:
        list[str]: The model file refs result."""
    return re.findall(r"(?:^|\s)--model\s+(\S+)", command)


def _hub_cache_ref(path: str) -> tuple[str, str, str] | None:
    """Decode a hub-cache file path into (repo, commit, file).

    Args:
        path: The path under the container Hugging Face hub cache.

    Returns:
        tuple[str, str, str] | None: The (repo, commit, file) result, or None
        when the path does not use the models--<owner>--<repo>/snapshots layout."""
    parts = Path(path.removeprefix(f"{HUB_CACHE_PREFIX}/")).parts
    if len(parts) < 4 or parts[0].removeprefix("models--") == parts[0] or parts[1] != "snapshots":
        return None
    owner, _, repo = parts[0].removeprefix("models--").partition("--")
    if not owner or not repo:
        return None
    return f"{owner}/{repo}", parts[2], "/".join(parts[3:])


def _ensure_model_file(settings: Config, model_id: str, model_ref: str, label: str) -> None:
    """Ensure a --model file exists in the host model cache, downloading if needed.

    Hub-cache paths resolve to an authenticated host-side download pinned to the
    snapshot commit the container command expects, so a wiped or fresh model
    cache cannot fail the container upstream with a missing model file.

    Args:
        settings: The settings.
        model_id: The model id.
        model_ref: The model ref.
        label: The label."""
    if model_ref.startswith(f"{HUB_CACHE_PREFIX}/"):
        ref = _hub_cache_ref(model_ref)
        if ref is None:
            LOGGER.warning("%s: cannot resolve hub cache path %s", label, model_ref)
            return
        repo, commit, file = ref
        HuggingFace(model_id, repo).get(
            file,
            cache_dir=settings.dirs.models,
            warmup=label,
            revision=commit,
        )
        return
    if model_ref.startswith("/") and not Path(model_ref).is_file():
        LOGGER.warning(
            "%s: model file %s is missing on the host; warmup may fail", label, model_ref
        )


def _prefetch_models(settings: Config, model_ids: list[str]) -> None:
    """Perform the internal prefetch models operation.

    Args:
        settings: The settings.
        model_ids: The model ids."""
    import yaml

    config = yaml.safe_load(settings.resolve_ls_config().read_text(encoding="utf-8")) or {}
    macros = config.get("macros", {})
    models = config.get("models", {})
    total_models = len(model_ids)
    for position, model_id in enumerate(model_ids, start=1):
        command = _expand_macros(str(models.get(model_id, {}).get("cmd", "")), macros)
        label = f"Warming model {position}/{total_models}: {model_id}"
        hf_match = re.search(r"(?:^|\s)-hf\s+(\S+)", command)
        if hf_match:
            repo, selector = HuggingFace.parse_spec(hf_match.group(1))
            if repo:
                hf = HuggingFace(model_id, repo)
                if selector:
                    hf.get(
                        hf.file(selector, suffixes=(".gguf",)),
                        cache_dir=settings.dirs.models,
                        warmup=label,
                    )
                else:
                    # Bare repo specs (for example GLM-5.3-Flash) resolve to a
                    # filtered snapshot; prefetching here keeps the download
                    # authenticated and off the container warmup deadline, in
                    # the same host cache the mounted container reads from.
                    hf.snapshot(cache_dir=settings.dirs.models, warmup=label)
        for model_ref in _model_file_refs(command):
            _ensure_model_file(settings, model_id, model_ref, label)


def warmup_models(settings: Config, model_ids: list[str]) -> int:
    """Load selected models and wait until each is ready.

    Args:
        settings: The settings.
        model_ids: The model ids.

    Returns:
        int: The warmup models result.

    Raises:
        SystemExit: If the warmup models operation cannot be completed."""
    auth = settings.load_auth()
    if auth.hf_token:
        os.environ["HF_TOKEN"] = auth.hf_token
    api_key = settings.resolved_api_key(auth)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    base_url = settings.listen_url()
    warmup_timeout = settings.warmup.timeout
    warmup_poll_interval = settings.warmup.poll_interval
    if settings.runtime.environment == RUNTIME.HOST:
        from .helpers.docker import DockerRuntime

        runtime = DockerRuntime(settings)
        runtime.ensure_daemon()
        if not runtime.is_running():
            raise SystemExit(
                f"container {settings.docker.container_name} is not running; start it first"
            )
    http = Http(base_url, headers=headers)
    health_status, _ = http.at("health").response()
    if health_status >= 400:
        raise SystemExit(f"llama-swap is not reachable at {base_url}")
    selected_ids = list(model_ids)
    if not selected_ids:
        payload = http.at("v1/models").json()
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


def serve(settings: Config) -> int:
    """Run llama-swap directly in the container.

    Args:
        settings: The settings.

    Returns:
        int: The serve result.

    Raises:
        SystemExit: If the serve operation cannot be completed."""
    config_path = settings.container_config_path()
    llama_swap_bin = Path(CONTAINERPATH.LLAMA_SWAP)
    if not llama_swap_bin.is_file():
        raise SystemExit(f"llama-swap binary not found at {llama_swap_bin}")
    LOGGER.info(
        "starting llama-swap (%s mode, config=%s, listen=:%s)",
        settings.runtime.mode,
        config_path,
        settings.runtime.container_port,
    )
    proc = subprocess.Popen(
        [
            str(llama_swap_bin),
            "-config",
            str(config_path),
            "-listen",
            f"0.0.0.0:{settings.runtime.container_port}",
        ],
        start_new_session=True,
    )
    saved: dict[signal.Signals, object] = {}

    def handle(signum: int, _frame: types.FrameType | None) -> None:
        """Perform the handle operation.

        Args:
            signum: The signum.
            _frame: The frame."""
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
