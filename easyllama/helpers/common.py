"""Provide general filesystem, environment, and progress utilities."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tomllib

from .logger import LOG as APP_LOG

LOGGER = APP_LOG.get(__name__)


def project_root() -> Path:
    """Perform the project root operation.

    Returns:
        Path: The project root result."""
    env_root = os.environ.get("EASYLLAMA_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parents[1]


def load_pyproject(root_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Perform the load pyproject operation.

    Args:
        root_dir: The root dir.

    Returns:
        tuple[dict[str, object], dict[str, object]]: The load pyproject result."""
    pyproject_path = root_dir / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    tool_config = data.get("tool", {}).get("easyllama", {})
    defaults = dict(tool_config.get("defaults", {}))
    configs = dict(tool_config.get("configs", {}))
    return defaults, configs


def normalize_mode(value: str | None) -> str:
    """Normalize and validate a server mode name.

    Args:
        value: User-selected mode, or ``None`` for the default mode.

    Returns:
        The normalized lowercase mode name.

    Raises:
        SystemExit: If the selected mode is not registered.
    """
    from ..servers import mode_names

    selected = (value or "basic").strip().lower()
    if selected not in mode_names():
        allowed = ", ".join(mode_names())
        raise SystemExit(f"unsupported mode: {selected}; allowed: {allowed}")
    return selected


def absolute_path(root_dir: Path, value: str) -> Path:
    """Perform the absolute path operation.

    Args:
        root_dir: The root dir.
        value: The value.

    Returns:
        Path: The absolute path result."""
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (root_dir / candidate).resolve()


def shutil_which(executable: str) -> str | None:
    """Perform the shutil which operation.

    Args:
        executable: The executable.

    Returns:
        str | None: The shutil which result."""
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / executable
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def detect_timezone() -> str:
    """Perform the detect timezone operation.

    Returns:
        str: The detect timezone result."""
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
    """Perform the image name for mode operation.

    Args:
        image_name_base: The image name base.
        image_tag_base: The image tag base.
        mode: The mode.

    Returns:
        str: The image name for mode result."""
    base = os.environ.get("EASYLLAMA_IMAGE_NAME", image_name_base) or image_name_base
    if ":" in base:
        repository, tag = base.rsplit(":", 1)
        return f"{repository}:{tag}-{mode}"
    return f"{base}:{image_tag_base}-{mode}"


def format_bytes(value: float) -> str:
    """Perform the format bytes operation.

    Args:
        value: The value.

    Returns:
        str: The format bytes result.

    Raises:
        AssertionError: If the format bytes operation cannot be completed."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def format_duration(seconds: float) -> str:
    """Perform the format duration operation.

    Args:
        seconds: The seconds.

    Returns:
        str: The format duration result."""
    return f"{max(0, round(seconds))}s"
