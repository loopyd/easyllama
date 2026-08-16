"""Provide URL-bound HTTP request and download operations."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

import pycurl

from easyllama.helpers import format_bytes, format_duration

from .logger import LOG as APP_LOG

LOGGER = APP_LOG.get(__name__)


class Http:
    """Perform HTTP operations against a bound URL.

    Attributes:
        url: The url.
        headers: The headers.
        timeout: The timeout."""

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> None:
        """Initialize the instance.

        Args:
            url: The url.
            headers: The headers.
            timeout: The timeout."""
        self.url = url
        self.headers = dict(headers or {})
        self.timeout = timeout

    @property
    def path(self) -> str:
        """Return the parsed URL path.

        Returns:
            str: The path result."""
        return urllib.parse.urlsplit(self.url).path

    @property
    def filename(self) -> str:
        """Return the final filename component of the URL.

        Returns:
            str: The filename result."""
        return Path(self.path).name

    def at(self, path: str) -> Http:
        """Return an HTTP client bound to a child URL.

        Args:
            path: The path.

        Returns:
            Http: The at result."""
        return Http(
            f"{self.url.rstrip('/')}/{path.lstrip('/')}",
            headers=self.headers,
            timeout=self.timeout,
        )

    def request(self, method: str = "GET") -> urllib.request.Request:
        """Build a standard-library request for the bound URL.

        Args:
            method: The method.

        Returns:
            urllib.request.Request: The request result."""
        return urllib.request.Request(self.url, headers=self.headers, method=method)

    def response(self, method: str = "GET") -> tuple[int, str]:
        """Perform a request and return its status and body.

        Args:
            method: The method.

        Returns:
            tuple[int, str]: The response result."""
        try:
            with urllib.request.urlopen(self.request(method), timeout=self.timeout) as response:
                return response.status, response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", errors="replace")
        except (TimeoutError, urllib.error.URLError) as exc:
            return 0, str(exc)

    def json(self) -> dict[str, object]:
        """Fetch and decode a JSON object.

        Returns:
            dict[str, object]: The json result.

        Raises:
            ValueError: If the json operation cannot be completed."""
        with urllib.request.urlopen(self.request(), timeout=self.timeout) as response:
            payload: Any = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"expected JSON object from {self.url}")
        return payload

    def content_length(self) -> int | None:
        """Fetch the remote content length when available.

        Returns:
            int | None: The content length result."""
        try:
            with urllib.request.urlopen(self.request("HEAD"), timeout=self.timeout) as response:
                length = response.headers.get("Content-Length")
        except urllib.error.URLError:
            return None
        return int(length) if length and length.isdigit() else None

    def download(self, destination: Path) -> None:
        """Download the bound URL to a local path.

        Args:
            destination: The destination."""
        LOGGER.info("Starting download %s -> %s", self.url, destination.name)
        started_at = last_log = time.monotonic()
        downloaded = last_downloaded = 0
        total_bytes = 0

        with destination.open("wb") as file_handle:
            curl = pycurl.Curl()
            try:
                curl.setopt(pycurl.URL, self.url)
                curl.setopt(pycurl.WRITEDATA, file_handle)
                curl.setopt(pycurl.FAILONERROR, True)
                curl.setopt(pycurl.FOLLOWLOCATION, True)
                if self.headers:
                    curl.setopt(
                        pycurl.HTTPHEADER,
                        [f"{key}: {value}" for key, value in self.headers.items()],
                    )

                def report(
                    total: float, current: float, _upload_total: float, _upload_current: float
                ) -> int:
                    """Perform the report operation.

                    Args:
                        total: The total.
                        current: The current.
                        _upload_total: The upload total.
                        _upload_current: The upload current.

                    Returns:
                        int: The report result."""
                    nonlocal downloaded, last_downloaded, last_log, total_bytes
                    now = time.monotonic()
                    downloaded, total_bytes = int(current), int(total)
                    elapsed = now - last_log
                    if elapsed < 5:
                        return 0
                    speed = (downloaded - last_downloaded) / elapsed
                    if total_bytes and speed > 0:
                        LOGGER.info(
                            "Downloading %s: %s/%s (%.1f%%, %s/s, ETA %s)",
                            destination.name,
                            format_bytes(downloaded),
                            format_bytes(total_bytes),
                            downloaded * 100 / total_bytes,
                            format_bytes(speed),
                            format_duration((total_bytes - downloaded) / speed),
                        )
                    else:
                        LOGGER.info(
                            "Downloading %s: %s (%s/s, ETA unknown)",
                            destination.name,
                            format_bytes(downloaded),
                            format_bytes(speed),
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
            format_bytes(downloaded),
            format_duration(elapsed),
            format_bytes(speed),
        )
