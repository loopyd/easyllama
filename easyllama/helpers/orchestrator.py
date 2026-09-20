"""Compile llama-swap process configs into networked container contracts."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, ClassVar

from pydantic.dataclasses import dataclass
import yaml

from ..config import IMAGE, MODE
from .images import ModeImages

_PORT = re.compile(r"\$\{PORT\}")
_ENV = re.compile(r"\$\{env\.([A-Za-z_][A-Za-z0-9_]*)\}")
_MACRO = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_-]*)\}")
_NAME = re.compile(r"[^a-z0-9]+")

# Each vLLM model gets its own LMCache server: the connector's chunk size must be
# a multiple of that model's vLLM block size, which is model-specific, so one
# server cannot serve multiple models.
LMCACHE_PORT_BASE = 5555
LMCACHE_HTTP_PORT_BASE = 18080
LMCACHE_PROMETHEUS_PORT_BASE = 19090


@dataclass(frozen=True, slots=True)
class ContainerContract:
    """Explicit command, port, health, and stop contract for one container."""

    name: str
    image: IMAGE
    command: tuple[str, ...]
    port: int | None = None
    health_path: str | None = None
    stop_signal: str = "SIGTERM"
    gpu: bool = False
    environment: tuple[str, ...] = ()
    lifecycle_port: int | None = None
    http_port: int | None = None
    publish: bool = False

    @property
    def endpoint(self) -> str | None:
        """Return this contract's private-network endpoint."""
        if self.port is None:
            return None
        return f"http://{self.name}:{self.port}"


@dataclass(frozen=True, slots=True)
class OrchestrationPlan:
    """Generated proxy configuration and its dependency containers."""

    mode: MODE
    config: dict[str, Any]
    containers: tuple[ContainerContract, ...]

    def write(self, path: Path) -> Path:
        """Write the generated llama-swap proxy configuration."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.config, sort_keys=False))
        return path


class ProxyConfigCompiler:
    """Convert concise model configs into complete remote container proxies."""

    ROOT_DEFAULTS: ClassVar[dict[str, Any]] = {
        "healthCheckTimeout": 1800,
        "logLevel": "info",
        "sendLoadingState": False,
        "globalTTL": 0,
        "routing": {
            "router": {
                "use": "group",
                "settings": {
                    "groups": {
                        "gpu": {
                            "swap": True,
                            "exclusive": True,
                            "members": ["qwen3-chat", "qwen3-embeddings"],
                        }
                    }
                },
            }
        },
    }
    MACRO_DEFAULTS: ClassVar[dict[str, Any]] = {"server_bin": "/opt/venv/bin/easyllama"}
    ENV_DEFAULTS: ClassVar[tuple[str, ...]] = ("HF_TOKEN=${HF_TOKEN}",)

    def __init__(self, mode: MODE, start_port: int = 9000, settings: Any = None) -> None:
        self.mode = mode
        self.start_port = start_port
        self.settings = settings
        images = ModeImages.for_mode(mode)
        self.images = images.select(settings.modes[mode].services) if settings else images

    def image_for_model(self, model: str, command: str) -> IMAGE:
        """Select the required backend image for a configured model."""
        if self.mode is MODE.QWEN and "vllm" in command:
            return IMAGE.VLLM
        if self.mode is MODE.GLM53FLASH:
            return IMAGE.FREETOKEN
        return IMAGE.LLAMACPP

    @staticmethod
    def _environment(values: list[str]) -> tuple[str, ...]:
        return tuple(_ENV.sub(lambda match: f"${{{match.group(1)}}}", value) for value in values)

    @staticmethod
    def _expand(command: str, macros: dict[str, Any]) -> str:
        """Expand llama-swap config macros before moving a command out of its config."""
        command = _ENV.sub(lambda match: f"${{{match.group(1)}}}", command)
        for _ in range(len(macros) + 1):
            expanded = _MACRO.sub(
                lambda match: str(macros.get(match.group(1), match.group(0))), command
            )
            if expanded == command:
                return _ENV.sub(lambda match: f"${{{match.group(1)}}}", command)
            command = expanded
        raise SystemExit("recursive llama-swap macros cannot be compiled")

    def _start_port(self, payload: dict[str, Any], source: Path) -> int:
        """Resolve the port base for this profile from its optional ``startPort`` key.

        Args:
            payload: The parsed config payload.
            source: The source file, for diagnostics.

        Returns:
            int: The base port for this profile.

        Raises:
            SystemExit: If the startPort value cannot be parsed."""
        raw = payload.pop("startPort", None)
        if raw is None:
            return self.start_port
        try:
            return int(raw)
        except (TypeError, ValueError):
            raise SystemExit(f"invalid startPort {raw!r} in {source}") from None

    def compile(self, source: Path, api_key: str | None = None) -> OrchestrationPlan:
        """Compile one local process config into proxy and container contracts."""
        source_payload = yaml.safe_load(source.read_text()) or {}
        payload = {**self.ROOT_DEFAULTS, **source_payload}
        contracts: list[ContainerContract] = []
        lmcache_servers: list[tuple[str, int, int]] = []
        macros = {**self.MACRO_DEFAULTS, **(payload.get("macros") or {})}
        models = payload.get("models", {})
        base = self._start_port(payload, source)
        host_network = self.settings is not None and self.settings.docker.network_mode == "host"
        claimed_ports: dict[int, str] = {}

        def claim(value: int, model_id: str, *, reserve_next: bool = False) -> None:
            """Reserve a port for one model.

            API ports also reserve ``port + 1``: multi-socket backends
            (FreeToken) bind the server port plus one for their
            torch.distributed store, so that slot must stay clear of every
            other model's address. Lifecycle ports are single listeners.
            """
            slots = (value, value + 1) if reserve_next else (value,)
            for port in slots:
                owner = claimed_ports.get(port)
                if owner is not None:
                    raise SystemExit(
                        f"model {model_id} port collision: {port} is already "
                        f"assigned to {owner}; set distinct port/lifecyclePort values"
                    )
                claimed_ports[port] = model_id

        for index, (model_id, model) in enumerate(models.items()):
            command = self._expand(str(model["cmd"]).strip(), macros)
            model.pop("cmdStop", None)
            environment = tuple(
                dict.fromkeys((*self.ENV_DEFAULTS, *self._environment(model.pop("env", []) or [])))
            )
            image = self.image_for_model(model_id, command)
            # Two ports per model: the API port, plus a reserved backend slot
            # at port + 1 for multi-socket backends (FreeToken binds
            # server_port + 1 for its torch.distributed store). Lifecycle
            # listeners start after the reserved band so they never land on a
            # backend slot or on another model's API port.
            #
            # Positions shift whenever the model set changes, so a model may
            # pin explicit ``port``/``lifecyclePort`` values: pinned addresses
            # keep captured container commands (and any deployment pinned to
            # them) valid when other models are added or removed.
            port = int(model.get("port") or 0) or base + 2 * index
            lifecycle_port = int(model.get("lifecyclePort") or 0) or base + 2 * len(models) + index
            claim(port, model_id, reserve_next=True)
            claim(lifecycle_port, model_id)
            model.pop("port", None)
            model.pop("lifecyclePort", None)
            # A model may opt into a host-loopback port binding so host-network
            # consumers (for example Hindsight) can reach its API directly
            # instead of going through the swap proxy.
            publish = bool(model.pop("publish", False))
            model_name = _NAME.sub("-", model_id.lower()).strip("-")
            name = f"easyllama-{self.mode}-{image}-{model_name}"
            command = _PORT.sub(str(port), command)
            bind_host = "127.0.0.1" if host_network else "0.0.0.0"
            command = re.sub(
                r"--host\s+(?:127\.0\.0\.1|localhost|0\.0\.0\.0)", f"--host {bind_host}", command
            )
            if image is IMAGE.VLLM:
                for obsolete in ("/app/bin/log-exec", "/app/bin/qwen-lmcache-vllm"):
                    command = command.replace(obsolete, "")
                # A model opts into LMCache by declaring the MP connector in its
                # command; each such model gets its own server because the chunk
                # size must be a multiple of that model's own vLLM block size.
                lmcache_chunk = int(model.pop("lmcacheChunkSize", 0) or 0)
                model.pop("lmcachePort", None)
                if '"lmcache.mp.host"' in command:
                    lmcache_port = LMCACHE_PORT_BASE + len(lmcache_servers)
                    lmcache_name = f"easyllama-{self.mode}-{IMAGE.LMCACHE}-{model_name}"
                    claim(lmcache_port, model_id)
                    command = command.replace(
                        '"lmcache.mp.host":"127.0.0.1"',
                        f'"lmcache.mp.host":"{lmcache_name}"',
                    )
                    command = command.replace(
                        '"lmcache.mp.port":5555',
                        f'"lmcache.mp.port":{lmcache_port}',
                    )
                    lmcache_servers.append((lmcache_name, lmcache_port, lmcache_chunk))
            command = " ".join(command.splitlines())
            contracts.append(
                ContainerContract(
                    name=name,
                    image=image,
                    command=("/bin/sh", "-lc", f"exec {command}"),
                    port=port,
                    health_path="/health" if image is IMAGE.VLLM else "/v1/models",
                    gpu=True,
                    environment=environment,
                    lifecycle_port=lifecycle_port,
                    publish=publish,
                )
            )
            endpoint_host = "127.0.0.1" if host_network else name
            model.update(
                {
                    "cmd": (
                        f"curl --fail --silent --show-error --request POST "
                        f"http://{endpoint_host}:{lifecycle_port}/run"
                    ),
                    "cmdStop": (
                        f"curl --fail --silent --show-error --request POST "
                        f"http://{endpoint_host}:{lifecycle_port}/sleep"
                    ),
                    "proxy": f"http://{endpoint_host}:{port}",
                    "checkEndpoint": contracts[-1].health_path,
                    "useModelName": model_id,
                }
            )
        if lmcache_servers:
            if host_network:
                raise SystemExit("host networking is not supported for LMCache dependencies")
            dependency = self.images.dependency(IMAGE.LMCACHE)
            for offset, (server_name, server_port, chunk_size) in enumerate(lmcache_servers):
                http_port = LMCACHE_HTTP_PORT_BASE + offset
                command = list(dependency.command)
                for flag, value in (
                    ("--instance-id", server_name),
                    ("--port", server_port),
                    ("--http-port", http_port),
                    ("--prometheus-port", LMCACHE_PROMETHEUS_PORT_BASE + offset),
                ):
                    command[command.index(flag) + 1] = str(value)
                if chunk_size:
                    command[command.index("--chunk-size") + 1] = str(chunk_size)
                command = tuple(item.replace("{mode}", str(self.mode)) for item in command)
                if self.settings is not None:
                    command = tuple(item.format(lmcache=self.settings.lmcache) for item in command)
                contracts.insert(
                    0,
                    ContainerContract(
                        name=server_name,
                        image=dependency.image,
                        command=command,
                        port=server_port,
                        http_port=http_port,
                        health_path=dependency.health_path,
                        stop_signal=dependency.stop_signal,
                        gpu=dependency.gpu,
                    ),
                )
        payload.pop("macros", None)
        configured = set(models)
        members = payload["routing"]["router"]["settings"]["groups"]["gpu"]["members"]
        payload["routing"]["router"]["settings"]["groups"]["gpu"]["members"] = [
            model for model in members if model in configured
        ]
        if api_key:
            payload["apiKeys"] = [api_key]
        # Ensure generated values remain ordinary YAML scalars, not enum objects.
        json.dumps(payload)
        return OrchestrationPlan(self.mode, payload, tuple(contracts))
