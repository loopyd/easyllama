from __future__ import annotations

from dataclasses import dataclass

from .basic import BasicServer
from .lucebox import LuceboxServer
from .server_base import BuildSource, RuntimeModeMetadata, ServerBase, Spec, server_metadata
from .spiritbuun import SpiritbuunServer


@dataclass(frozen=True, slots=True)
class ServerDef:
    name: str
    help: str


_SERVERS: dict[str, type[ServerBase]] = {
    BasicServer.name: BasicServer,
    LuceboxServer.name: LuceboxServer,
    SpiritbuunServer.name: SpiritbuunServer,
}


def _mode_registry() -> dict[str, RuntimeModeMetadata]:
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
    return tuple(ServerDef(name=name, help=cls.help) for name, cls in _SERVERS.items())


def mode_defs() -> tuple[RuntimeModeMetadata, ...]:
    return tuple(_MODES.values())


def mode_names() -> tuple[str, ...]:
    return tuple(_MODES)


def mode_def(mode: str) -> RuntimeModeMetadata:
    try:
        return _MODES[mode]
    except KeyError as exc:
        supported = ", ".join(sorted(_MODES))
        raise SystemExit(f"unknown runtime mode: {mode}; supported: {supported}") from exc


def make(name: str) -> ServerBase:
    try:
        server_cls = _SERVERS[name]
    except KeyError as exc:
        supported = ", ".join(sorted(_SERVERS))
        raise SystemExit(f"unknown server: {name}; supported: {supported}") from exc
    return server_cls()


def run(name: str, argv: list[str] | None = None) -> int:
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
