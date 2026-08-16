"""Expose the public easyllama package metadata."""

from __future__ import annotations

from dataclasses import dataclass

from .base import BuildSource, RuntimeModeMetadata, ServerBase, Spec, server_metadata
from .basic import BasicServer
from .lucebox import LuceboxServer
from .spiritbuun import SpiritbuunServer


@dataclass(frozen=True, slots=True)
class ServerDef:
    """Describe a server exposed to CLI discovery.

    Attributes:
        name: The name (str).
        help: The help (str)."""

    name: str
    help: str


_SERVERS: dict[str, type[ServerBase]] = {
    BasicServer.name: BasicServer,
    LuceboxServer.name: LuceboxServer,
    SpiritbuunServer.name: SpiritbuunServer,
}


def _mode_registry() -> dict[str, RuntimeModeMetadata]:
    """Perform the internal mode registry operation.

    Returns:
        dict[str, RuntimeModeMetadata]: The mode registry result.

    Raises:
        RuntimeError: If the mode registry operation cannot be completed."""
    modes: dict[str, RuntimeModeMetadata] = {}
    for server_cls in _SERVERS.values():
        for mode in server_cls.runtime_modes:
            existing = modes.get(mode.mode)
            if existing is not None:
                raise RuntimeError(f"duplicate runtime mode metadata for {mode.mode}")
            modes[mode.mode] = mode
    return modes


_MODES = _mode_registry()


def defs() -> tuple[ServerDef, ...]:
    """Perform the defs operation.

    Returns:
        tuple[ServerDef, ...]: The defs result."""
    return tuple(ServerDef(name=name, help=cls.help) for name, cls in _SERVERS.items())


def mode_defs() -> tuple[RuntimeModeMetadata, ...]:
    """Perform the mode defs operation.

    Returns:
        tuple[RuntimeModeMetadata, ...]: The mode defs result."""
    return tuple(_MODES.values())


def mode_names() -> tuple[str, ...]:
    """Perform the mode names operation.

    Returns:
        tuple[str, ...]: The mode names result."""
    return tuple(_MODES)


def mode_def(mode: str) -> RuntimeModeMetadata:
    """Perform the mode def operation.

    Args:
        mode: The mode.

    Returns:
        RuntimeModeMetadata: The mode def result.

    Raises:
        SystemExit: If the mode def operation cannot be completed."""
    try:
        return _MODES[mode]
    except KeyError as exc:
        supported = ", ".join(sorted(_MODES))
        raise SystemExit(f"unknown runtime mode: {mode}; supported: {supported}") from exc


def make(name: str) -> ServerBase:
    """Perform the make operation.

    Args:
        name: The name.

    Returns:
        ServerBase: The make result.

    Raises:
        SystemExit: If the make operation cannot be completed."""
    try:
        server_cls = _SERVERS[name]
    except KeyError as exc:
        supported = ", ".join(sorted(_SERVERS))
        raise SystemExit(f"unknown server: {name}; supported: {supported}") from exc
    return server_cls()


def run(name: str, argv: list[str] | None = None) -> int:
    """Run a built server process specification.

    Args:
        name: The name.
        argv: The argv.

    Returns:
        int: The run result."""
    return make(name).main(argv)


__all__ = [
    "BuildSource",
    "RuntimeModeMetadata",
    "ServerBase",
    "ServerDef",
    "Spec",
    "defs",
    "make",
    "mode_def",
    "mode_defs",
    "mode_names",
    "run",
    "server_metadata",
]
