from __future__ import annotations

from dataclasses import dataclass

from .basic import BasicServer
from .lucebox import LuceboxServer
from .server_base import ServerBase, Spec
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


def defs() -> tuple[ServerDef, ...]:
    return tuple(ServerDef(name=name, help=cls.help) for name, cls in _SERVERS.items())


def make(name: str) -> ServerBase:
    try:
        server_cls = _SERVERS[name]
    except KeyError as exc:
        supported = ", ".join(sorted(_SERVERS))
        raise SystemExit(f"unknown server: {name}; supported: {supported}") from exc
    return server_cls()


def run(name: str, argv: list[str] | None = None) -> int:
    return make(name).main(argv)


__all__ = ["ServerBase", "ServerDef", "Spec", "defs", "make", "run"]
