"""Build and manage easyllama Docker images and containers."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

from docker import from_env
from docker.errors import APIError, DockerException, ImageNotFound
from docker.types import DeviceRequest

from ..config import (
    CHAT_TEMPLATE_DIR_CONTAINER,
    MMPROJ_DIR_CONTAINER,
    MODELS_DIR_CONTAINER,
    RUNTIME_CONTAINER,
    Config,
)
from ..servers import mode_def as server_mode_def, mode_defs as server_mode_defs
from .logger import LOG as APP_LOG

LOGGER = APP_LOG.get(__name__)


def detect_runtime_mode(value: str | None = None) -> str:
    """Return the explicit runtime mode or detect host versus container execution."""
    selected = value or os.environ.get("EASYLLAMA_RUNTIME_MODE")
    if not selected:
        return "container" if Path("/.dockerenv").exists() else "host"
    selected = selected.strip().lower()
    if selected not in {"host", "container"}:
        raise SystemExit(f"unsupported runtime mode: {selected}; allowed: host, container")
    return selected


def cuda_architectures(self) -> str:
    """Resolve CUDA architectures from configuration or detected GPUs.

    Args:
        settings: The settings.

    Returns:
        str: The compute cuda architectures result."""
    if self.settings.cmake_cuda_architectures != "auto":
        return self.settings.cmake_cuda_architectures
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        LOGGER.warning(
            "nvidia-smi not found or failed; using fallback CUDA arch %s",
            self.settings.default_cuda_architectures,
        )
        return self.settings.default_cuda_architectures
    values: set[str] = set()
    for line in result.stdout.splitlines():
        match = re.findall(r"\d+", line)
        if not match:
            continue
        major = match[0]
        minor = match[1][0] if len(match) > 1 and match[1] else "0"
        values.add(f"{major}{minor}")
    if not values:
        LOGGER.warning(
            "failed to detect compute capability; using fallback CUDA arch %s",
            self.settings.default_cuda_architectures,
        )
        return self.settings.default_cuda_architectures
    return ";".join(sorted(values, key=int))


class DockerRuntime:
    """Manage the Docker image and container for one configuration.

    Attributes:
        settings: The settings.
        client: The client.
        api: The api."""

    def __init__(self, settings: Config) -> None:
        """Initialize the instance.

        Args:
            settings: The settings."""
        self.settings = settings
        self.client: Any = from_env()
        self.api: Any = self.client.api

    def ensure_daemon(self) -> None:
        """Perform the ensure daemon operation.

        Raises:
            SystemExit: If the ensure daemon operation cannot be completed."""
        try:
            self.client.ping()
        except DockerException as exc:
            raise SystemExit("docker daemon is not reachable (start docker and retry)") from exc

    def ensure_nvidia_runtime(self) -> None:
        """Perform the ensure nvidia runtime operation.

        Raises:
            SystemExit: If the ensure nvidia runtime operation cannot be completed."""
        runtimes = self.client.info().get("Runtimes", {})
        if "nvidia" not in runtimes:
            raise SystemExit("nvidia container runtime is not available in docker")

    def get_container(self):
        """Perform the get container operation."""
        for container in self.client.containers.list(all=True):
            if container.name == self.settings.container_name:
                return container
        return None

    def get_legacy_default_container(self):
        """Perform the get legacy default container operation."""
        if self.settings.container_name != "easyllama-server-swap" or any(
            name in os.environ for name in ("EASYLLAMA_CONTAINER_NAME",)
        ):
            return None
        for container in self.client.containers.list(all=True):
            if container.name == "llamacpp-server-swap":
                return container
        return None

    def get_running_container_count(self) -> int:
        """Perform the get running container count operation.

        Returns:
            int: The get running container count result."""
        return sum(
            1
            for container in self.client.containers.list()
            if container.name == self.settings.container_name
        )

    def is_running(self) -> bool:
        """Perform the is running operation.

        Returns:
            bool: The is running result."""
        container = self.get_container()
        return bool(container and container.status == "running")

    def image_exists(self, image_name: str | None = None) -> bool:
        """Perform the image exists operation.

        Args:
            image_name: The image name.

        Returns:
            bool: The image exists result."""
        try:
            self.client.images.get(image_name or self.settings.image_name)
        except ImageNotFound:
            return False
        return True

    def _build_cmd(self, target: str, build_args: dict[str, str]) -> list[str]:
        """Perform the internal build cmd operation.

        Args:
            target: The target.
            build_args: The build args.

        Returns:
            list[str]: The build cmd result.

        Raises:
            SystemExit: If the build cmd operation cannot be completed."""
        docker_bin = shutil.which("docker")
        if docker_bin is None:
            raise SystemExit("docker CLI is required for image builds")
        cmd = [
            docker_bin,
            "buildx",
            "build",
            "--load",
            "--progress=plain",
            "--pull",
            "--tag",
            self.settings.image_name,
            "--target",
            target,
            "--file",
            str(self.settings.root_dir / "Dockerfile"),
        ]
        for key, value in build_args.items():
            cmd.extend(("--build-arg", f"{key}={value}"))
        cmd.append(str(self.settings.root_dir))
        return cmd

    def _ensure_buildx(self) -> None:
        """Perform the internal ensure buildx operation.

        Raises:
            SystemExit: If the ensure buildx operation cannot be completed."""
        docker_bin = shutil.which("docker")
        if docker_bin is None:
            raise SystemExit("docker CLI is required for image builds")
        result = subprocess.run(
            [docker_bin, "buildx", "inspect", "--bootstrap"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            message = "docker buildx with BuildKit is required for image builds"
            if detail:
                message = f"{message}: {detail}"
            raise SystemExit(message)

    def build_image(self) -> int:
        """Build the selected runtime image.

        Returns:
            int: The build image result.

        Raises:
            SystemExit: If the build image operation cannot be completed."""
        self.ensure_daemon()
        self._ensure_buildx()
        mode_metadata = server_mode_def(self.settings.mode)
        target = mode_metadata.docker_target
        build_args = {
            "BUILD_MODE": self.settings.mode,
            "DEBIAN_FRONTEND": "noninteractive",
            "HOST_TZ": self.settings.host_tz,
            "HOST_LANG": self.settings.host_lang,
            "HOST_LC_ALL": self.settings.host_lc_all,
            "CMAKE_CUDA_ARCHITECTURES": self.cuda_architectures(),
        }
        build_args.update(mode_metadata.build_args(self.settings))
        LOGGER.info(
            mode_metadata.build_summary(
                self.settings,
                image_name=self.settings.image_name,
                target=target,
            )
        )
        proc = subprocess.Popen(
            self._build_cmd(target, build_args),
            cwd=str(self.settings.root_dir),
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            message = line.rstrip()
            if message:
                LOGGER.debug(message)
        if proc.wait() != 0:
            raise SystemExit(f"docker buildx build failed for {self.settings.image_name}")
        LOGGER.info("build complete: %s", self.settings.image_name)
        return 0

    def _remove_container(self, container) -> None:
        """Perform the internal remove container operation.

        Args:
            container: The container."""
        try:
            if container.status == "running":
                container.stop(timeout=10)
            container.remove()
        except APIError:
            container.remove(force=True)
        LOGGER.info("removed container %s", container.name)

    def remove_container(self) -> None:
        """Perform the remove container operation."""
        container = self.get_container()
        if container is not None:
            self._remove_container(container)
        legacy = self.get_legacy_default_container()
        if legacy is not None:
            LOGGER.info("migrating legacy default container to easyllama project naming")
            self._remove_container(legacy)
        if container is None and legacy is None:
            LOGGER.warning("container %s does not exist", self.settings.container_name)

    def run_container(self) -> int:
        """Start the selected runtime container.

        Returns:
            int: The run container result.

        Raises:
            SystemExit: If the run container operation cannot be completed."""
        self.ensure_daemon()
        self.ensure_nvidia_runtime()
        auth = self.settings.load_auth()
        self.settings.models_dir.mkdir(parents=True, exist_ok=True)
        self.settings.mmproj_dir.mkdir(parents=True, exist_ok=True)

        running_count = self.get_running_container_count()
        if running_count > 1:
            raise SystemExit(
                f"refusing to start {self.settings.container_name}: found "
                f"{running_count} running containers with same name"
            )

        legacy = self.get_legacy_default_container()
        if legacy is not None:
            LOGGER.info("migrating legacy default container to easyllama project naming")
            self._remove_container(legacy)

        container = self.get_container()
        if container is not None and container.status == "running":
            running_mode = container.labels.get("easyllama.mode", "unknown")
            LOGGER.warning(
                "container %s is already running in %s mode; use restart to replace it",
                self.settings.container_name,
                running_mode,
            )
            return 0
        if container is not None:
            self.remove_container()

        if not self.image_exists():
            LOGGER.info("image %s is missing; building it first", self.settings.image_name)
            self.build_image()

        config_path, container_config_path_value = self.settings.effective_config_path(auth)
        mmproj_argument = self.settings.mmproj_arg(auth)
        volumes = {
            str(self.settings.models_dir): {"bind": MODELS_DIR_CONTAINER, "mode": "rw"},
            str(self.settings.mmproj_dir): {"bind": MMPROJ_DIR_CONTAINER, "mode": "rw"},
            str(config_path): {"bind": container_config_path_value, "mode": "ro"},
        }
        if self.settings.chat_template_dir.is_dir():
            volumes[str(self.settings.chat_template_dir)] = {
                "bind": CHAT_TEMPLATE_DIR_CONTAINER,
                "mode": "ro",
            }
        if Path("/etc/localtime").is_file():
            volumes["/etc/localtime"] = {"bind": "/etc/localtime", "mode": "ro"}
        if Path("/etc/timezone").is_file():
            volumes["/etc/timezone"] = {"bind": "/etc/timezone", "mode": "ro"}

        environment = {
            "EASYLLAMA_RUNTIME_MODE": RUNTIME_CONTAINER,
            "EASYLLAMA_MODE": self.settings.mode,
            "CONTAINER_PORT": str(self.settings.container_port),
            "EASYLLAMA_MMPROJ_ARG": mmproj_argument,
            "TZ": self.settings.host_tz,
            "LANG": self.settings.host_lang,
            "LC_ALL": self.settings.host_lc_all,
            "EASYLLAMA_ROOT": "/app",
        }
        if auth.hf_token:
            environment["HF_TOKEN"] = auth.hf_token

        backend = server_mode_def(self.settings.mode).backend
        self.client.containers.run(
            self.settings.image_name,
            command=["serve"],
            detach=True,
            init=True,
            name=self.settings.container_name,
            restart_policy={"Name": "unless-stopped"},
            security_opt=["no-new-privileges"],
            pids_limit=4096 if backend == "vllm" else self.settings.pids_limit,
            runtime="nvidia",
            device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
            ports={f"{self.settings.container_port}/tcp": ("127.0.0.1", self.settings.host_port)},
            volumes=volumes,
            environment=environment,
            labels={
                "easyllama.mode": self.settings.mode,
                "easyllama.backend": backend,
                "easyllama.image": self.settings.image_name,
            },
            **({"shm_size": "32g"} if backend == "vllm" else {}),
        )
        LOGGER.info(
            "started %s (%s mode) on http://localhost:%s",
            self.settings.container_name,
            self.settings.mode,
            self.settings.host_port,
        )
        return 0

    def _remove_effective_configs(self) -> None:
        """Perform the internal remove effective configs operation."""
        if not self.settings.runtime_dir.is_dir():
            return
        for path in self.settings.runtime_dir.glob("*.effective.yaml"):
            path.unlink(missing_ok=True)

    def stop_container(self) -> int:
        """Stop and remove the runtime container.

        Returns:
            int: The stop container result."""
        self.ensure_daemon()
        self.remove_container()
        self._remove_effective_configs()
        return 0

    def restart_container(self) -> int:
        """Replace the runtime container.

        Returns:
            int: The restart container result."""
        self.stop_container()
        return self.run_container()

    def print_logs(self, *, tail: int | None = None) -> int:
        """Print or follow runtime container logs.

        Args:
            tail: The tail.

        Returns:
            int: The print logs result.

        Raises:
            SystemExit: If the print logs operation cannot be completed."""
        self.ensure_daemon()
        container = self.get_container()
        if container is None:
            raise SystemExit(f"container {self.settings.container_name} does not exist")
        if tail is not None:
            print(container.logs(tail=tail).decode("utf-8", errors="replace"), end="")
            return 0
        try:
            for chunk in container.logs(stream=True, follow=True):
                print(chunk.decode("utf-8", errors="replace"), end="")
        except KeyboardInterrupt:
            LOGGER.info("log follow interrupted")
            return 0
        return 0

    def status(self) -> int:
        """Log runtime container and image status.

        Returns:
            int: The status result."""
        self.ensure_daemon()
        container = self.get_container()
        if container is None:
            LOGGER.info("container %s is not present", self.settings.container_name)
        else:
            image = container.image
            image_name = image.tags[0] if image and image.tags else "<untagged>"
            LOGGER.info(
                "container %s status=%s image=%s", container.name, container.status, image_name
            )
        available_images = []
        for mode_metadata in server_mode_defs():
            image_name = self.settings.image_for_mode(mode_metadata.mode)
            if self.image_exists(image_name):
                available_images.append(image_name)
        if available_images:
            LOGGER.info("available mode images: %s", ", ".join(available_images))
        return 0

    def clean(self, *, all_images: bool = False) -> int:
        """Remove runtime containers, generated configuration, and images.

        Args:
            all_images: The all images.

        Returns:
            int: The clean result."""
        self.ensure_daemon()
        container = self.get_container()
        if container is not None:
            self.remove_container()
        self._remove_effective_configs()
        image_names = [self.settings.image_name]
        if all_images:
            image_names = [
                self.settings.image_for_mode(mode_metadata.mode)
                for mode_metadata in server_mode_defs()
            ]
        for image_name in image_names:
            try:
                self.client.images.remove(image_name, force=True)
                LOGGER.info("removed image %s", image_name)
            except ImageNotFound:
                LOGGER.warning("image %s does not exist", image_name)
        return 0
