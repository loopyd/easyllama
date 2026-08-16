"""Configure and expose process-wide easyllama logging."""

from __future__ import annotations

import logging
import os

try:
    from colorama import Fore, Style, init as colorama_init
except ModuleNotFoundError:  # pragma: no cover - exercised only before deps are installed

    class _ColorFallback:
        """Represent ColorFallback state and behavior.

        Attributes:
            MAGENTA: The MAGENTA.
            RED: The RED.
            YELLOW: The YELLOW.
            CYAN: The CYAN.
            GREEN: The GREEN.
            RESET_ALL: The RESET ALL."""

        MAGENTA = ""
        RED = ""
        YELLOW = ""
        CYAN = ""
        GREEN = ""
        RESET_ALL = ""

    Fore = Style = _ColorFallback()

    def colorama_init() -> None:
        """Provide a no-op fallback when colorama is unavailable."""
        return None  # noqa: RET501


DEFAULT_LEVEL = "INFO"
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
    """Format log records with optional level colors.

    Attributes:
        use_color: The use color."""

    def __init__(self, *, use_color: bool) -> None:
        """Initialize the instance.

        Args:
            use_color: The use color."""
        super().__init__("%(levelname)s %(message)s")
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        """Perform the format operation.

        Args:
            record: The record.

        Returns:
            str: The format result."""
        level_name = record.levelname
        if self.use_color:
            color = LEVEL_COLORS.get(record.levelno, "")
            if color:
                record.levelname = f"{color}{level_name}{Style.RESET_ALL}"
        try:
            return super().format(record)
        finally:
            record.levelname = level_name


class Logger:
    """Manage process-wide logging through a singleton.

    Attributes:
        _instance: The instance."""

    _instance: Logger | None = None

    def __new__(cls) -> Logger:
        """Return the process-wide singleton instance.

        Returns:
            Logger: The new result."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def configure(
        self,
        *,
        verbosity: str | None = None,
        quiet: bool = False,
        no_color: bool = False,
    ) -> None:
        """Configure the process-wide root logger.

        Args:
            verbosity: The verbosity.
            quiet: The quiet.
            no_color: The no color."""
        colorama_init()
        handler = logging.StreamHandler()
        handler.setFormatter(
            ColorFormatter(use_color=not no_color and not os.environ.get("EASYLLAMA_NO_COLOR"))
        )
        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(self.resolve_level(verbosity, quiet=quiet))

    def get(self, name: str) -> logging.Logger:
        """Download or return a cached repository file.

        Args:
            name: The name.

        Returns:
            logging.Logger: The get result."""
        return logging.getLogger(name)

    def resolve_level(self, verbosity: str | None = None, *, quiet: bool = False) -> int:
        """Resolve command-line and environment verbosity to a logging level.

        Args:
            verbosity: The verbosity.
            quiet: The quiet.

        Returns:
            int: The resolve level result."""
        if quiet:
            return logging.WARNING
        selected = (verbosity or os.environ.get("EASYLLAMA_LOG_LEVEL") or DEFAULT_LEVEL).lower()
        return LEVEL_NAMES.get(selected, logging.DEBUG)

    def set_level(self, name: str, level: int) -> int:
        """Set a named logger level and return its previous level.

        Args:
            name: The name.
            level: The level.

        Returns:
            int: The set level result."""
        logger = self.get(name)
        previous = logger.level
        logger.setLevel(level)
        return previous


LOG = Logger()
