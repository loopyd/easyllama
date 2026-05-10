from __future__ import annotations

from typing import Any, Dict
import logging
import urllib.error
import urllib.request
from pathlib import Path
import os
import subprocess
import tomllib

from .logger import get_logger

LOGGER = get_logger(__name__)


# Base URL for Hugging Face resources
HF_URL_BASE = "https://huggingface.co"


class ProgressReporter:
    """Generic progress reporter for long-running tasks.

    Can be used for downloads or any process that reports incremental progress.

    Parameters:
      name: short name for the task (for logs)
      total: optional total units (bytes/items)
      log_threshold: how many units between update logs
      level: logging level (int)
      start_template: optional format string for a start message
      update_template: format string for periodic updates
      finish_template: format string for final message
      format_args: extra named values available to templates

    Templates support these fields by default: {name}, {downloaded}, {total}, {percent}.
    Additional values from `format_args` are also available (for example {url}).
    """

    def __init__(
        self,
        name: str,
        total: int | None = None,
        log_threshold: int = 2 * 1024 * 1024,
        level: int = logging.DEBUG,
        start_template: str | None = None,
        update_template: str | None = None,
        finish_template: str | None = None,
        format_args: Dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.total = total
        self.log_threshold = log_threshold
        self.level = level
        self.start_template = start_template
        self.update_template = (
            update_template
            or "Downloading {name}: {downloaded}/{total} bytes ({percent:.1f}%)"
        )
        self.finish_template = (
            finish_template or "Download complete {name}: {downloaded}/{total} bytes"
        )
        self.format_args = dict(format_args or {})

        self.downloaded = 0
        self.next_log = log_threshold

    def _render(self, template: str) -> str:
        percent = 0.0
        if self.total and self.total > 0:
            percent = self.downloaded * 100.0 / self.total
        values = dict(
            name=self.name,
            downloaded=self.downloaded,
            total=self.total or 0,
            percent=percent,
        )
        values.update(self.format_args)
        try:
            return template.format(**values)
        except Exception:
            # Fallback to a simple join if formatting fails
            return f"{self.name} {self.downloaded}/{self.total or '?'}"

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


def _download_file(url: str, destination: Path, hf_token: str | None) -> None:
    request = urllib.request.Request(url)
    if hf_token:
        request.add_header("Authorization", f"Bearer {hf_token}")
    with urllib.request.urlopen(request) as response, destination.open("wb") as file_handle:
        total_header = response.getheader("Content-Length")
        total_size: int | None = int(total_header) if total_header and total_header.isdigit() else None
        reporter = ProgressReporter(
            destination.name,
            total_size,
            format_args={"url": url},
            start_template="Starting download {url} -> {name}",
            update_template=(
                "Downloading {name}: {downloaded}/{total} bytes ({percent:.1f}%)"
                if total_size
                else "Downloading {name}: {downloaded} bytes downloaded"
            ),
            finish_template=(
                "Download complete {name}: {downloaded}/{total} bytes"
                if total_size
                else "Download complete {name}: {downloaded} bytes"
            ),
        )
        reporter.start()
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            file_handle.write(chunk)
            reporter.update(len(chunk))
        reporter.finish()


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


from .servers import mode_names as server_mode_names


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
