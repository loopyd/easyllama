from __future__ import annotations

import logging
import os

try:
    from colorama import Fore, Style, init as colorama_init
except ModuleNotFoundError:  # pragma: no cover - exercised only before deps are installed

    class _ColorFallback:
        MAGENTA = ""
        RED = ""
        YELLOW = ""
        CYAN = ""
        GREEN = ""
        RESET_ALL = ""

    Fore = Style = _ColorFallback()

    def colorama_init() -> None:
        return None


DEFAULT_LEVEL = "DEBUG"
LEVEL_NAMES = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}
LEVEL_COLORS = {
    logging.CRITICAL: Fore.MAGENTA,
    logging.ERROR: Fore.RED,
    logging.WARNING: Fore.YELLOW,
    logging.INFO: Fore.CYAN,
    logging.DEBUG: Fore.GREEN,
}


class ColorFormatter(logging.Formatter):
    def __init__(self, *, use_color: bool) -> None:
        super().__init__("%(levelname)s %(message)s")
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        level_name = record.levelname
        if self.use_color:
            color = LEVEL_COLORS.get(record.levelno, "")
            if color:
                record.levelname = f"{color}{level_name}{Style.RESET_ALL}"
        try:
            return super().format(record)
        finally:
            record.levelname = level_name


def resolve_level(verbosity: str | None = None, *, quiet: bool = False) -> int:
    if quiet:
        return logging.WARNING
    env_level = os.environ.get("LLAMACPP_LOG_LEVEL") or os.environ.get("EASYLLAMA_LOG_LEVEL")
    selected = (verbosity or env_level or DEFAULT_LEVEL).lower()
    return LEVEL_NAMES.get(selected, logging.DEBUG)


def configure_logging(
    *,
    verbosity: str | None = None,
    quiet: bool = False,
    no_color: bool = False,
) -> None:
    colorama_init()
    handler = logging.StreamHandler()
    use_color = not no_color and not os.environ.get("LLAMACPP_NO_COLOR")
    handler.setFormatter(ColorFormatter(use_color=use_color))

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(resolve_level(verbosity, quiet=quiet))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
