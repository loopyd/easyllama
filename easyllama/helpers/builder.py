"""Compile and build mode-specific Docker images from stage files."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
import shutil
import subprocess
from typing import TYPE_CHECKING, Any, ClassVar

from docker.errors import ImageNotFound
from pydantic.dataclasses import dataclass

from ..config import IMAGE, MODE
from .images import ModeImages

if TYPE_CHECKING:
    from ..config import Config


@dataclass(frozen=True, slots=True)
class Dockerfile:
    """A compiled Dockerfile ready for BuildKit."""

    path: Path
    target: str
    stages: tuple[str, ...]


class DockerfileCompiler:
    """Stitch dependency-ordered Docker stage files."""

    _MODE_BUILDERS: ClassVar[dict[MODE, tuple[MODE, ...]]] = {
        **{mode: (mode,) for mode in MODE},
        MODE.LUCEBOX: (MODE.LLAMACPP, MODE.LUCEBOX),
    }

    def __init__(self, root: Path) -> None:
        self.root = root
        self.stage_dir = root / "docker"

    @staticmethod
    def target(mode: MODE, image_type: IMAGE) -> str:
        """Return the final stage name for a mode and image role."""
        suffix = f"-{mode}" if image_type is IMAGE.LLAMACPP and mode is not MODE.LLAMACPP else ""
        return f"runtime-{image_type}{suffix}"

    def stages(self, mode: MODE, image_type: IMAGE) -> tuple[str, ...]:
        """Return the minimal ordered stage set for a mode and image role."""
        if not ModeImages.for_mode(mode).requires(image_type):
            raise SystemExit(f"{mode} mode does not use a {image_type} image")
        stages = ["runtime-base", "runtime-python"]
        if image_type is IMAGE.LLAMASWAP:
            stages.append("vllm-wrapper-build")
        elif image_type is IMAGE.LLAMACPP:
            stages += [
                "builder-base",
                *(f"{builder_mode}-builder" for builder_mode in self._MODE_BUILDERS[mode]),
            ]
        elif image_type is IMAGE.VLLM:
            stages += ["builder-base", "vllm-builder"]
        elif image_type is IMAGE.FREETOKEN:
            stages += ["freetoken-builder"]
        stages.append(self.target(mode, image_type))
        return tuple(dict.fromkeys(stages))

    def compile(self, mode: MODE, image_type: IMAGE) -> Dockerfile:
        """Compile selected stages into a generated Dockerfile."""
        stages = self.stages(mode, image_type)
        target = self.target(mode, image_type)
        name = (
            str(image_type)
            if mode is MODE.LLAMACPP and image_type is IMAGE.LLAMACPP
            else f"{mode}-{image_type}"
        )
        output = self.root / ".runtime/docker" / f"{name}.Dockerfile"
        output.parent.mkdir(parents=True, exist_ok=True)
        parts = [(self.stage_dir / "header.Dockerfile").read_text().rstrip()]
        for stage in stages:
            path = self.stage_dir / f"{stage}.Dockerfile"
            if not path.is_file():
                raise SystemExit(f"Docker stage does not exist: {path}")
            parts.append(path.read_text().strip())
        output.write_text("\n\n".join(parts) + "\n")
        return Dockerfile(output, target, stages)


class DockerBuilder:
    """Build and track one mode/type Docker image."""

    def __init__(self, settings: Config, client: Any, mode: MODE, image_type: IMAGE) -> None:
        self.settings = settings
        self.client = client
        self.mode = mode
        self.image_type = image_type
        self.compiler = DockerfileCompiler(settings.dirs.root)

    @property
    def name(self) -> str:
        """Return a deterministic image tag for this role."""
        repository = self.settings.docker.image_name
        base_tag = self.settings.docker.image_tag
        if self.image_type in {IMAGE.LLAMASWAP, IMAGE.LMCACHE, IMAGE.FREETOKEN} or (
            self.mode is MODE.LLAMACPP and self.image_type is IMAGE.LLAMACPP
        ):
            return f"{repository}:{base_tag}-{self.image_type}"
        return f"{repository}:{base_tag}-{self.mode}-{self.image_type}"

    def exists(self) -> bool:
        """Return whether the built image is present."""
        try:
            self.client.images.get(self.name)
        except ImageNotFound:
            return False
        return True

    def command(self, build_args: dict[str, str]) -> list[str]:
        """Return the BuildKit command for this image."""
        docker = shutil.which("docker")
        if docker is None:
            raise SystemExit("docker CLI is required for image builds")
        dockerfile = self.compiler.compile(self.mode, self.image_type)
        command = [
            docker,
            "buildx",
            "build",
            "--load",
            "--progress=plain",
            "--pull",
            "--tag",
            self.name,
            "--label",
            "easyllama.managed=true",
            "--label",
            f"easyllama.mode={self.mode}"
            if self.image_type not in {IMAGE.LLAMASWAP, IMAGE.LMCACHE}
            else "easyllama.mode=shared",
            "--label",
            f"easyllama.type={self.image_type}",
            "--target",
            dockerfile.target,
            "--file",
            str(dockerfile.path),
        ]
        for key, value in build_args.items():
            command.extend(("--build-arg", f"{key}={value}"))
        command.append(str(self.settings.dirs.root))
        return command

    def build(self, build_args: dict[str, str]) -> None:
        """Build this image and label it for discovery and cleanup."""
        subprocess.run(self.command(build_args), cwd=self.settings.dirs.root, check=True)

    def remove(self) -> None:
        """Remove this image if present."""
        with suppress(ImageNotFound):
            self.client.images.remove(self.name, force=True)

    @classmethod
    def managed_images(cls, client: Any) -> list[Any]:
        """Return images built by DockerBuilder."""
        return client.images.list(filters={"label": "easyllama.managed=true"})
