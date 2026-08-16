"""Report rate-limited progress for easyllama operations."""

from __future__ import annotations

import logging
from typing import Any

from .logger import LOG as APP_LOG

LOGGER = APP_LOG.get(__name__)


class ProgressReporter:
    """Report rate-limited progress for non-download operations.

    Attributes:
        name: The name.
        total: The total.
        log_threshold: The log threshold.
        level: The level.
        start_template: The start template.
        update_template: The update template.
        finish_template: The finish template.
        format_args: The format args.
        downloaded: The downloaded.
        next_log: The next log."""

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
        """Initialize the instance.

        Args:
            name: The name.
            total: The total.
            log_threshold: The log threshold.
            level: The level.
            start_template: The start template.
            update_template: The update template.
            finish_template: The finish template.
            format_args: The format args."""
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
        """Perform the internal render operation.

        Args:
            template: The template.

        Returns:
            str: The render result."""
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
        """Perform the start operation."""
        if self.start_template:
            LOGGER.log(self.level, self._render(self.start_template))

    def update(self, n: int) -> None:
        """Perform the update operation.

        Args:
            n: The n."""
        self.downloaded += n
        if self.downloaded >= self.next_log:
            LOGGER.log(self.level, self._render(self.update_template))
            self.next_log += self.log_threshold

    def finish(self) -> None:
        """Perform the finish operation."""
        LOGGER.log(self.level, self._render(self.finish_template))
