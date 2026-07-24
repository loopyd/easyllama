from __future__ import annotations

import logging
import os
from pathlib import Path
import subprocess
import time
import tomllib
from typing import Any
import urllib.error
import urllib.request

import pycurl

from .logger import get_logger
from .servers import mode_names as server_mode_names

LOGGER = get_logger(__name__)


# Base URL for Hugging Face resources
HF_URL_BASE = "https://huggingface.co"


class ProgressReporter:
    """Rate-limited log progress for non-download tasks."""

    def __init__(
        self,
        name: str,
        total: int | None = None,
        log_threshold: int = 1,
        level: int = logging.INFO,
        start_template: str | None = None,
        update_template: str = "{name}: {downloaded}/{total}",
        finish_template: str = "Completed {name}: {downloaded}/{total}",
        format_args: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.total = total
        self.log_threshold = log_threshold
        self.level = level
        self.start_template = start_template
        self.update_template = update_template
        self.finish_template = finish_template
        self.format_args = dict(format_args or {})
        self.downloaded = 0
        self.next_log = log_threshold

    def _render(self, template: str) -> str:
        values = {
            "name": self.name,
            "downloaded": self.downloaded,
            "total": self.total or 0,
            "percent": self.downloaded * 100 / self.total if self.total else 0,
            "remaining": max((self.total or 0) - self.downloaded, 0),
            **self.format_args,
        }
        return template.format(**values)

    def start(self) -> None:
        if self.start_template:
            LOGGER.log(self.level, self._render(self.start_template))

    def update(self, n: int) -> None:
        self.downloaded += n
        if self.downloaded >= self.next_log:
            LOGGER.log(self.level, self._render(self.update_template))
            self.next_log += self.log_threshold

    def finish(self) -> None:
        LOGGER.log(self.level, self._render(self.finish_template))


def hf_mmproj_url(spec: str) -> str:
    parts = spec.split("/", 2)
    if len(parts) != 3 or not all(parts):
        raise SystemExit(f"LLAMACPP_HF_MMPROJ must be <owner>/<repo>/<file.gguf>; got: {spec}")
    owner, repo, filename = parts
    return f"{HF_URL_BASE}/{owner}/{repo}/blob/main/{filename}"


def _fetch_content_length(url: str, hf_token: str | None) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    if hf_token:
        request.add_header("Authorization", f"Bearer {hf_token}")
    try:
        with urllib.request.urlopen(request) as response:
            length = response.headers.get("Content-Length")
    except urllib.error.URLError:
        return None
    return int(length) if length and length.isdigit() else None


def _format_bytes(value: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def _format_duration(seconds: float) -> str:
    return f"{max(0, round(seconds))}s"


def _download_file(url: str, destination: Path, hf_token: str | None) -> None:
    """Download *url* and log byte progress, current rate, and ETA every five seconds."""
    LOGGER.info("Starting download %s -> %s", url, destination.name)
    started_at = last_log = time.monotonic()
    downloaded = last_downloaded = 0
    total_bytes = 0

    with destination.open("wb") as file_handle:
        curl = pycurl.Curl()
        try:
            curl.setopt(pycurl.URL, url)
            curl.setopt(pycurl.WRITEDATA, file_handle)
            curl.setopt(pycurl.FAILONERROR, True)
            curl.setopt(pycurl.FOLLOWLOCATION, True)
            if hf_token:
                curl.setopt(pycurl.HTTPHEADER, [f"Authorization: Bearer {hf_token}"])

            def report(
                total: float, current: float, _upload_total: float, _upload_current: float
            ) -> int:
                nonlocal downloaded, last_downloaded, last_log, total_bytes
                now = time.monotonic()
                downloaded, total_bytes = int(current), int(total)
                elapsed = now - last_log
                if elapsed < 5:
                    return 0
                speed = (downloaded - last_downloaded) / elapsed
                if total_bytes and speed > 0:
                    eta = _format_duration((total_bytes - downloaded) / speed)
                    LOGGER.info(
                        "Downloading %s: %s/%s (%.1f%%, %s/s, ETA %s)",
                        destination.name,
                        _format_bytes(downloaded),
                        _format_bytes(total_bytes),
                        downloaded * 100 / total_bytes,
                        _format_bytes(speed),
                        eta,
                    )
                else:
                    LOGGER.info(
                        "Downloading %s: %s (%s/s, ETA unknown)",
                        destination.name,
                        _format_bytes(downloaded),
                        _format_bytes(speed),
                    )
                last_downloaded, last_log = downloaded, now
                return 0

            curl.setopt(pycurl.XFERINFOFUNCTION, report)
            curl.setopt(pycurl.NOPROGRESS, False)
            curl.perform()
        finally:
            curl.close()
    elapsed = time.monotonic() - started_at
    speed = downloaded / elapsed if elapsed else 0
    LOGGER.info(
        "Download complete %s: %s in %s (%s/s)",
        destination.name,
        _format_bytes(downloaded),
        _format_duration(elapsed),
        _format_bytes(speed),
    )


def project_root() -> Path:
    env_root = os.environ.get("EASYLLAMA_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parents[1]


def load_pyproject(root_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    pyproject_path = root_dir / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    tool_config = data.get("tool", {}).get("easyllama", {})
    defaults = dict(tool_config.get("defaults", {}))
    configs = dict(tool_config.get("configs", {}))
    return defaults, configs


def known_modes() -> tuple[str, ...]:
    return server_mode_names()


def normalize_mode(value: str | None) -> str:
    selected = (value or "basic").strip().lower()
    if selected not in known_modes():
        allowed = ", ".join(known_modes())
        raise SystemExit(f"unsupported mode: {selected}; allowed: {allowed}")
    return selected


def detect_runtime_mode(value: str | None = None) -> str:
    selected = value or os.environ.get("LLAMACPP_RUNTIME_MODE")
    if not selected:
        return "container" if Path("/.dockerenv").exists() else "host"
    if selected not in {"host", "container"}:
        allowed = ", ".join(("host", "container"))
        raise SystemExit(f"unsupported LLAMACPP_RUNTIME_MODE={selected}; allowed: {allowed}")
    return selected


def absolute_path(root_dir: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (root_dir / candidate).resolve()


def shutil_which(executable: str) -> str | None:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / executable
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def detect_timezone() -> str:
    if os.environ.get("TZ"):
        return os.environ["TZ"]
    timezone_path = Path("/etc/timezone")
    if timezone_path.is_file():
        return timezone_path.read_text(encoding="utf-8").strip()
    timedatectl = shutil_which("timedatectl")
    if timedatectl:
        result = subprocess.run(
            [timedatectl, "show", "-p", "Timezone", "--value"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return "UTC"


def image_name_for_mode(image_name_base: str, image_tag_base: str, mode: str) -> str:
    explicit = os.environ.get("LLAMACPP_IMAGE_NAME")
    base = explicit or image_name_base
    if ":" in base:
        repository, tag = base.rsplit(":", 1)
        return f"{repository}:{tag}-{mode}"
    return f"{base}:{image_tag_base}-{mode}"
