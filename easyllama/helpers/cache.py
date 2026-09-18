"""Manage persistent host cache directories."""

from __future__ import annotations

from pathlib import Path
import shutil

from pydantic.dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HostCache:
    """Base functionality shared by persistent host caches.

    Attributes:
        name: The name.
        host: The host.
        container: The container."""

    name: str
    host: Path
    container: str

    def exists(self) -> bool:
        """Return whether the cache directory exists."""
        return self.host.is_dir()

    def ensure(self) -> None:
        """Create the cache directory and tracked placeholder."""
        self.host.mkdir(parents=True, exist_ok=True)
        (self.host / ".gitkeep").touch()

    def clean(self) -> None:
        """Remove cache contents while preserving the directory placeholder."""
        self.ensure()
        for path in self.host.iterdir():
            if path.name == ".gitkeep":
                continue
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()

    def volume(self) -> dict[str, dict[str, str]]:
        """Return this cache's Docker bind mapping."""
        return {str(self.host): {"bind": self.container, "mode": "rw"}}


class RootCache(HostCache):
    """Persist the container root user's general cache."""


class PackageCache(HostCache):
    """Persist the container system package-manager cache."""


class PythonCache(HostCache):
    """Persist the container Python package cache."""


class ModelCache(HostCache):
    """Persist downloaded Hugging Face model repositories."""
