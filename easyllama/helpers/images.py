"""Define mode-specific Docker image dependencies."""

from __future__ import annotations

from typing import ClassVar

from pydantic.dataclasses import dataclass

from ..config import IMAGE, MODE


@dataclass(frozen=True, slots=True)
class ImageDependency:
    """One image role required by a mode."""

    image: IMAGE
    gpu: bool = False
    command: tuple[str, ...] = ()
    port: int | None = None
    health_path: str | None = None
    stop_signal: str = "SIGTERM"

    def name(self, mode: MODE) -> str:
        """Return this dependency's stable network identity."""
        return f"easyllama-{mode}-{self.image}"


@dataclass(frozen=True, slots=True)
class ModeImages:
    """Ordered Docker image dependency set for one mode."""

    mode: MODE
    dependencies: tuple[ImageDependency, ...]

    _REGISTRY: ClassVar[dict[MODE, ModeImages]] = {}

    def __post_init__(self) -> None:
        self._REGISTRY.setdefault(self.mode, self)

    @property
    def primary(self) -> IMAGE:
        """Return the externally exposed orchestrator role."""
        return self.dependencies[0].image

    @property
    def backend(self) -> IMAGE:
        """Return the mode's primary inference runtime."""
        return self.dependencies[1].image

    def requires(self, image: IMAGE) -> bool:
        """Return whether this mode uses an image role."""
        return any(dependency.image is image for dependency in self.dependencies)

    def dependency(self, image: IMAGE) -> ImageDependency:
        """Return one required image role."""
        for dependency in self.dependencies:
            if dependency.image is image:
                return dependency
        raise KeyError(f"{self.mode} mode does not use a {image} image")

    def select(self, services: tuple[IMAGE, ...]) -> ModeImages:
        """Return dependencies in the configured service order."""
        return ModeImages(self.mode, tuple(self.dependency(image) for image in services))

    @classmethod
    def for_mode(cls, mode: MODE) -> ModeImages:
        """Return registered dependencies for a mode."""
        return cls._REGISTRY[mode]


for _mode in MODE:
    ModeImages(
        _mode,
        (
            ImageDependency(IMAGE.LLAMASWAP),
            *(
                (
                    ImageDependency(IMAGE.VLLM, gpu=True),
                    ImageDependency(
                        IMAGE.LMCACHE,
                        gpu=True,
                        command=(
                            "/opt/venv/bin/lmcache",
                            "server",
                            "--instance-id",
                            "easyllama-{mode}",
                            "--host",
                            "0.0.0.0",
                            "--port",
                            "5555",
                            "--http-port",
                            "18080",
                            "--prometheus-port",
                            "19090",
                            "--chunk-size",
                            "{lmcache.chunk_size}",
                            "--l1-size-gb",
                            "{lmcache.l1_size_gb}",
                            "--eviction-policy",
                            "LRU",
                            "--separate-object-groups",
                        ),
                        port=5555,
                        health_path="/health",
                    ),
                    ImageDependency(IMAGE.LLAMACPP, gpu=True),
                )
                if _mode is MODE.QWEN
                else (ImageDependency(IMAGE.LLAMACPP, gpu=True),)
            ),
        ),
    )
