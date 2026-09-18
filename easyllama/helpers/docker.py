"""Build and manage easyllama Docker images and containers."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from queue import Queue
import re
import shutil
import subprocess
from threading import Thread
import time
from typing import Any

from docker import from_env
from docker.errors import APIError, DockerException, ImageNotFound
from docker.types import DeviceRequest, Ulimit

from ..config import CONTAINERPATH, IMAGE, RUNTIME, Config
from ..servers import mode_def as server_mode_def, mode_defs as server_mode_defs
from .builder import DockerBuilder
from .cache import HostCache, ModelCache, PackageCache, PythonCache, RootCache
from .images import ModeImages
from .logger import LOG as APP_LOG
from .orchestrator import ContainerContract, ProxyConfigCompiler

LOGGER = APP_LOG.get(__name__)


def detect_runtime_mode(value: str | None = None) -> RUNTIME:
    """Return the explicit runtime mode or detect host versus container execution."""
    selected = value or os.environ.get("EASYLLAMA_RUNTIME_MODE")
    if not selected:
        return RUNTIME.CONTAINER if Path("/.dockerenv").exists() else RUNTIME.HOST
    try:
        return RUNTIME(selected.strip().lower())
    except ValueError as error:
        allowed = ", ".join(RUNTIME)
        raise SystemExit(f"unsupported runtime mode: {selected}; allowed: {allowed}") from error


def cuda_architectures(settings: Config) -> str:
    """Resolve CUDA architectures from configuration or detected GPUs."""
    if settings.docker.cuda.cmake != "auto":
        return settings.docker.cuda.cmake
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        LOGGER.warning(
            "nvidia-smi not found or failed; using fallback CUDA arch %s",
            settings.docker.cuda.default,
        )
        return settings.docker.cuda.default
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
            settings.docker.cuda.default,
        )
        return settings.docker.cuda.default
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

    def _host_capacity(self) -> tuple[int, float, float]:
        """Return configured or detected host CPU, RAM, and swap capacity."""
        docker_info = self.client.info()
        host = self.settings.resources.host
        cpus = min(host.cpus, int(docker_info.get("NCPU") or 1))
        ram_gib = host.ram_gib
        if ram_gib is None:
            ram_gib = int(docker_info.get("MemTotal") or 0) / 1024**3
        swap_gib = host.swap_gib
        if swap_gib is None:
            swap_gib = 0.0
            try:
                fields = Path("/proc/meminfo").read_text().splitlines()
                swap_kib = next(
                    int(line.split()[1]) for line in fields if line.startswith("SwapTotal:")
                )
                swap_gib = swap_kib / 1024**2
            except (OSError, StopIteration, ValueError):
                pass
        return cpus, ram_gib, swap_gib

    def _resources(self, image: IMAGE) -> dict[str, int]:
        """Return Docker CPU, RAM, and host-swap limits for an image role."""
        profile = self.settings.resources.profile(image)
        cpus, ram_gib, swap_gib = self._host_capacity()
        profiles = self.settings.resources.profiles
        memory = profile.ram.allocate(ram_gib, profiles.ram.floor(profile.ram))
        resources = {
            "nano_cpus": profile.cpu.allocate(cpus, profiles.cpu.floor(profile.cpu)) * 1_000_000_000
        }
        if memory:
            resources["mem_limit"] = memory
            resources["memswap_limit"] = memory + profile.swap.allocate(
                swap_gib, profiles.swap.floor(profile.swap)
            )
        return resources

    def ensure_nvidia_runtime(self) -> None:
        """Perform the ensure nvidia runtime operation.

        Raises:
            SystemExit: If the ensure nvidia runtime operation cannot be completed."""
        runtimes = self.client.info().get("Runtimes", {})
        if "nvidia" not in runtimes:
            raise SystemExit("nvidia container runtime is not available in docker")

    @property
    def network_name(self) -> str:
        """Return the private network shared by one mode's containers."""
        return f"easyllama-{self.settings.runtime.mode}"

    def mode_networks(self) -> list[Any]:
        """Return managed private networks for the selected mode."""
        return self.client.networks.list(
            filters={
                "label": ["easyllama.managed=true", f"easyllama.mode={self.settings.runtime.mode}"]
            }
        )

    def ensure_network(self):
        """Return the selected mode's managed network, creating it when absent."""
        networks = self.mode_networks()
        for network in networks:
            if network.name == self.network_name:
                return network
        same_name = self.client.networks.list(names=[self.network_name])
        if same_name:
            raise SystemExit(f"network {self.network_name} exists but is not managed by easyllama")
        return self.client.networks.create(
            self.network_name,
            driver="bridge",
            labels={"easyllama.managed": "true", "easyllama.mode": self.settings.runtime.mode},
        )

    def remove_networks(self) -> None:
        """Remove every managed network owned by the selected mode."""
        for network in self.mode_networks():
            network.remove()
            LOGGER.info("removed network %s", network.name)

    def get_container(self):
        """Return the externally exposed llama-swap container."""
        for container in self.client.containers.list(all=True):
            if container.name == self.settings.docker.container_name:
                return container
        return None

    def mode_containers(self) -> list[Any]:
        """Return all managed containers for the selected mode."""
        return self.client.containers.list(
            all=True,
            filters={
                "label": ["easyllama.managed=true", f"easyllama.mode={self.settings.runtime.mode}"]
            },
        )

    def get_legacy_default_container(self):
        """Perform the get legacy default container operation."""
        if self.settings.docker.container_name != "easyllama-server-swap" or any(
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
            if container.name == self.settings.docker.container_name
        )

    def is_running(self) -> bool:
        """Perform the is running operation.

        Returns:
            bool: The is running result."""
        container = self.get_container()
        return bool(container and container.status == "running")

    def builders(self) -> tuple[DockerBuilder, ...]:
        """Return builders for all images required by the selected mode."""
        return tuple(
            DockerBuilder(self.settings, self.client, self.settings.runtime.mode, dependency.image)
            for dependency in ModeImages.for_mode(self.settings.runtime.mode)
            .select(self.settings.modes[self.settings.runtime.mode].services)
            .dependencies
        )

    def image_exists(self, image_name: str | None = None) -> bool:
        """Perform the image exists operation.

        Args:
            image_name: The image name.

        Returns:
            bool: The image exists result."""
        try:
            self.client.images.get(
                image_name or self.settings.image_for_mode(self.settings.runtime.mode)
            )
        except ImageNotFound:
            return False
        return True

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

    def build_image(self, image_type: IMAGE | None = None) -> int:
        """Build the selected runtime image.

        Returns:
            int: The build image result.

        Raises:
            SystemExit: If the build image operation cannot be completed."""
        self.ensure_daemon()
        self._ensure_buildx()
        mode_metadata = server_mode_def(self.settings.runtime.mode)
        dependencies = ModeImages.for_mode(self.settings.runtime.mode).select(
            self.settings.modes[self.settings.runtime.mode].services
        )
        if image_type is not None and not dependencies.requires(image_type):
            raise SystemExit(f"{self.settings.runtime.mode} mode does not use a {image_type} image")
        builders = (
            (DockerBuilder(self.settings, self.client, self.settings.runtime.mode, image_type),)
            if image_type is not None
            else self.builders()
        )
        common_build_args = {
            "BUILD_MODE": str(self.settings.runtime.mode),
            "BUILD_JOBS": str(max(1, int(self.settings.resources.host.cpus * 0.75))),
            "DEBIAN_FRONTEND": "noninteractive",
            "HOST_TZ": self.settings.locale.timezone,
            "HOST_LANG": self.settings.locale.lang,
            "HOST_LC_ALL": self.settings.locale.lc_all,
        }
        for builder in builders:
            build_args = common_build_args.copy()
            if builder.image_type in {IMAGE.LLAMACPP, IMAGE.FREETOKEN}:
                if builder.image_type is IMAGE.LLAMACPP:
                    build_args["CMAKE_CUDA_ARCHITECTURES"] = cuda_architectures(self.settings)
                build_args.update(mode_metadata.build_args(self.settings))
                summary = mode_metadata.build_summary(
                    self.settings,
                    image_name=builder.name,
                    target=builder.compiler.target(self.settings.runtime.mode, builder.image_type),
                )
            else:
                target = builder.compiler.target(self.settings.runtime.mode, builder.image_type)
                summary = f"building {builder.name} (type={builder.image_type} target={target})"
            LOGGER.info(summary)
            proc = subprocess.Popen(
                builder.command(build_args),
                cwd=str(self.settings.dirs.root),
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
                raise SystemExit(f"docker buildx build failed for {builder.name}")
            LOGGER.info("build complete: %s", builder.name)
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
        """Remove every managed container in the selected mode stack."""
        container = self.get_container()
        removed = self.mode_containers()
        for managed in removed:
            self._remove_container(managed)
        if container is not None and all(managed.id != container.id for managed in removed):
            self._remove_container(container)
        legacy = self.get_legacy_default_container()
        if legacy is not None:
            LOGGER.info("migrating legacy default container to easyllama project naming")
            self._remove_container(legacy)
        if container is None and legacy is None and not removed:
            LOGGER.warning("container %s does not exist", self.settings.docker.container_name)

    def _container_environment(self, auth, contract: ContainerContract) -> dict[str, str]:
        """Return environment values for one dependency container.

        Contract ``${ENV}`` placeholders expand against the host environment. A
        placeholder that cannot be resolved on the host must not leak the literal
        ``${ENV}`` into the container: a value already resolved here (for example
        the configured HF_TOKEN) is kept instead, and a fully resolved expansion
        takes precedence over it (host environment > configured credential).
        """
        environment = {
            "EASYLLAMA_RUNTIME_MODE": RUNTIME.CONTAINER,
            "EASYLLAMA_MODE": self.settings.runtime.mode,
            "TZ": self.settings.locale.timezone,
            "LANG": self.settings.locale.lang,
            "LC_ALL": self.settings.locale.lc_all,
            "EASYLLAMA_ROOT": "/app",
            "EASYLLAMA_MMPROJ_ARG": self.settings.mmproj_arg(auth),
            # Pin every in-container Hugging Face client (easyllama helper, vLLM,
            # FreeToken, transformers) to the mounted host cache so model downloads
            # persist across image rebuilds and container restarts instead of
            # re-downloading into an ephemeral image layer.
            "HF_HOME": str(Path(CONTAINERPATH.MODELS).parent),
            "HF_HUB_CACHE": str(CONTAINERPATH.MODELS),
        }
        if auth.hf_token:
            environment["HF_TOKEN"] = auth.hf_token
        for item in contract.environment:
            key, _, value = item.partition("=")
            resolved = os.path.expandvars(value)
            if "${" in resolved:
                continue
            if resolved or key not in environment:
                environment[key] = resolved
        return environment

    def _run_dependency(
        self,
        contract: ContainerContract,
        auth,
        network,
        volumes: dict[str, dict[str, str]],
    ) -> Any:
        """Start one backend dependency from its lifecycle contract."""
        builder = DockerBuilder(
            self.settings, self.client, self.settings.runtime.mode, contract.image
        )
        if not builder.exists():
            self.build_image(contract.image)
        command = list(contract.command)
        if contract.lifecycle_port is not None:
            command = [
                "/opt/venv/bin/easyllama",
                "lifecycle",
                *(["--host", "127.0.0.1"] if network is None else []),
                "--port",
                str(contract.lifecycle_port),
                "--",
                *command,
            ]
        kwargs: dict[str, Any] = {
            "command": command,
            "entrypoint": [],
            "detach": True,
            "init": True,
            "name": contract.name,
            "hostname": contract.name,
            **({"network_mode": "host"} if network is None else {"network": network.name}),
            "restart_policy": {"Name": "unless-stopped"},
            "security_opt": ["no-new-privileges"],
            "pids_limit": 4096
            if contract.image is IMAGE.VLLM
            else self.settings.runtime.pids_limit,
            "volumes": volumes,
            "environment": self._container_environment(auth, contract),
            **self._resources(contract.image),
            "labels": {
                "easyllama.managed": "true",
                "easyllama.mode": self.settings.runtime.mode,
                "easyllama.type": contract.image,
                "easyllama.port": str(contract.port or ""),
                "easyllama.stop-signal": contract.stop_signal,
            },
        }
        if contract.health_path:
            health_port = contract.lifecycle_port or (
                18080 if contract.image is IMAGE.LMCACHE else contract.port
            )
            health_path = (
                "/health"
                if contract.lifecycle_port
                else ("/" if contract.image is IMAGE.LMCACHE else contract.health_path)
            )
            kwargs["healthcheck"] = {
                "test": [
                    "CMD",
                    "curl",
                    "--fail",
                    "--silent",
                    f"http://127.0.0.1:{health_port}{health_path}",
                ],
                "interval": 30_000_000_000,
                "timeout": 5_000_000_000,
                "start_period": 20_000_000_000,
                "retries": 3,
            }
        if contract.gpu:
            kwargs.update(
                runtime="nvidia",
                device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
            )
        if contract.image in {IMAGE.VLLM, IMAGE.LMCACHE}:
            # ponytail: LMCacheMPConnector requires CUDA IPC; isolate per mode when it
            # supports a named IPC namespace instead of Docker's host namespace.
            kwargs["ipc_mode"] = "host"
        if contract.image is IMAGE.VLLM:
            kwargs.update(
                shm_size="32g",
                ulimits=[Ulimit(name="memlock", soft=-1, hard=-1)],
            )
        return self.client.containers.run(builder.name, **kwargs)

    def _wait_for_dependency(self, container: Any) -> None:
        """Wait until one dependency is healthy before starting the next."""
        deadline = time.monotonic() + self.settings.warmup.timeout
        while time.monotonic() < deadline:
            container.reload()
            state = container.attrs["State"]
            if state.get("Status") != "running":
                raise SystemExit(f"dependency {container.name} stopped during startup")
            if state.get("Health", {}).get("Status") == "healthy":
                return
            time.sleep(self.settings.warmup.poll_interval)
        raise SystemExit(f"dependency {container.name} did not become healthy")

    def run_container(self) -> int:
        """Start the selected runtime container.

        Returns:
            int: The run container result.

        Raises:
            SystemExit: If the run container operation cannot be completed."""
        self.ensure_daemon()
        self.ensure_nvidia_runtime()
        host_network = self.settings.docker.network_mode == "host"
        network = None if host_network else self.ensure_network()
        auth = self.settings.load_auth()
        caches = self.host_caches()
        for cache in caches:
            cache.ensure()
        self.settings.dirs.mmproj.mkdir(parents=True, exist_ok=True)

        running_count = self.get_running_container_count()
        if running_count > 1:
            raise SystemExit(
                f"refusing to start {self.settings.docker.container_name}: found "
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
                self.settings.docker.container_name,
                running_mode,
            )
            return 0
        # Sweep stale containers from earlier runs (stopped orchestrator or
        # dependency containers left behind by a crash) so start stays
        # idempotent instead of failing with a container-name conflict.
        if container is not None or self.mode_containers():
            self.remove_container()

        source_config = self.settings.resolve_ls_config()
        plan = ProxyConfigCompiler(self.settings.runtime.mode, settings=self.settings).compile(
            source_config, auth.api_key
        )
        config_path = plan.write(
            self.settings.dirs.runtime / f"{self.settings.runtime.mode}.proxy.effective.yaml"
        )
        container_config_path_value = f"/app/config.d/{config_path.name}"
        mmproj_argument = self.settings.mmproj_arg(auth)
        shared_volumes = {
            str(self.settings.dirs.mmproj): {"bind": CONTAINERPATH.MMPROJ, "mode": "rw"},
        }
        volumes = {
            **shared_volumes,
            str(config_path): {"bind": container_config_path_value, "mode": "ro"},
        }
        for cache in caches:
            volumes.update(cache.volume())
        if self.settings.dirs.chat_template.is_dir():
            volumes[str(self.settings.dirs.chat_template)] = {
                "bind": CONTAINERPATH.CHAT_TEMPLATE,
                "mode": "ro",
            }
        if Path("/etc/localtime").is_file():
            volumes["/etc/localtime"] = {"bind": "/etc/localtime", "mode": "ro"}
        if Path("/etc/timezone").is_file():
            volumes["/etc/timezone"] = {"bind": "/etc/timezone", "mode": "ro"}

        dependency_volumes = dict(shared_volumes)
        for cache in caches:
            dependency_volumes.update(cache.volume())
        if self.settings.dirs.chat_template.is_dir():
            dependency_volumes[str(self.settings.dirs.chat_template)] = {
                "bind": CONTAINERPATH.CHAT_TEMPLATE,
                "mode": "ro",
            }
        for contract in plan.containers:
            dependency = self._run_dependency(contract, auth, network, dependency_volumes)
            if contract.health_path:
                self._wait_for_dependency(dependency)

        orchestrator = DockerBuilder(
            self.settings, self.client, self.settings.runtime.mode, IMAGE.LLAMASWAP
        )
        if not orchestrator.exists():
            self.build_image(IMAGE.LLAMASWAP)
        environment = self._container_environment(
            auth,
            ContainerContract(
                name=self.settings.docker.container_name,
                image=IMAGE.LLAMASWAP,
                command=("serve",),
            ),
        )
        environment.update(
            CONTAINER_PORT=str(self.settings.runtime.container_port),
            EASYLLAMA_MMPROJ_ARG=mmproj_argument,
        )
        if host_network:
            connection: dict[str, object] = {"network_mode": "host"}
        elif network is None:
            raise SystemExit(f"network {self.network_name} is required for container networking")
        else:
            connection = {
                "network": network.name,
                "ports": {
                    f"{self.settings.runtime.container_port}/tcp": (
                        str(self.settings.runtime.host),
                        self.settings.runtime.host_port,
                    ),
                },
            }
        process = {"command": ["serve"]}
        if host_network:
            process = {
                "entrypoint": [str(CONTAINERPATH.LLAMA_SWAP)],
                "command": [
                    "-config",
                    container_config_path_value,
                    "-listen",
                    f"{self.settings.runtime.host}:{self.settings.runtime.host_port}",
                ],
            }
        self.client.containers.run(
            orchestrator.name,
            **process,
            detach=True,
            init=True,
            name=self.settings.docker.container_name,
            restart_policy={"Name": "unless-stopped"},
            security_opt=["no-new-privileges"],
            pids_limit=self.settings.runtime.pids_limit,
            **self._resources(IMAGE.LLAMASWAP),
            volumes=volumes,
            environment=environment,
            **connection,
            labels={
                "easyllama.managed": "true",
                "easyllama.mode": self.settings.runtime.mode,
                "easyllama.image": orchestrator.name,
                "easyllama.type": IMAGE.LLAMASWAP,
            },
        )
        LOGGER.info(
            "started %s (%s mode) on http://%s:%s",
            self.settings.docker.container_name,
            self.settings.runtime.mode,
            self.settings.runtime.host,
            self.settings.runtime.host_port,
        )
        return 0

    def _remove_effective_configs(self) -> None:
        """Perform the internal remove effective configs operation."""
        if not self.settings.dirs.runtime.is_dir():
            return
        for path in self.settings.dirs.runtime.glob("*.effective.yaml"):
            path.unlink(missing_ok=True)

    def stop_container(self) -> int:
        """Stop and remove the complete selected-mode stack and its networks."""
        self.ensure_daemon()
        self.remove_container()
        self.remove_networks()
        self._remove_effective_configs()
        return 0

    def restart_container(self) -> int:
        """Replace the runtime container.

        Returns:
            int: The restart container result."""
        self.stop_container()
        return self.run_container()

    @staticmethod
    def _log_lines(container: Any, chunks: Any, queue: Queue[tuple[str, str] | None]) -> None:
        """Decode one container's stream into tagged complete lines."""
        pending = ""
        for chunk in chunks:
            pending += chunk.decode("utf-8", errors="replace")
            while "\n" in pending:
                line, pending = pending.split("\n", 1)
                queue.put((container.name, line))
        if pending:
            queue.put((container.name, pending))
        queue.put(None)

    def print_logs(self, *, tail: int | None = None) -> int:
        """Aggregate selected mode container logs through the application logger."""
        self.ensure_daemon()
        containers = sorted(self.mode_containers(), key=lambda container: container.name)
        if not containers:
            raise SystemExit(f"no containers exist for {self.settings.runtime.mode} mode")
        if tail is not None:
            history: list[tuple[str, str, str]] = []
            for container in containers:
                text = container.logs(tail=tail, timestamps=True).decode("utf-8", errors="replace")
                for line in text.splitlines():
                    timestamp, separator, message = line.partition(" ")
                    if not separator:
                        continue
                    history.append((timestamp, container.name, message))
            # Docker emits fixed-width RFC3339Nano UTC timestamps, so lexical order
            # preserves nanosecond chronology without datetime's microsecond truncation.
            for _timestamp, container_name, line in sorted(history):
                LOGGER.info(line, extra={"container": container_name})
            return 0

        queue: Queue[tuple[str, str] | None] = Queue()
        threads = [
            Thread(
                target=self._log_lines,
                args=(container, container.logs(stream=True, follow=True), queue),
                daemon=True,
            )
            for container in containers
        ]
        for thread in threads:
            thread.start()
        remaining = len(threads)
        try:
            while remaining:
                item = queue.get()
                if item is None:
                    remaining -= 1
                    continue
                container_name, line = item
                LOGGER.log(logging.INFO, line, extra={"container": container_name})
        except KeyboardInterrupt:
            LOGGER.info("log follow interrupted")
        return 0

    def status(self) -> int:
        """Log runtime container and image status.

        Returns:
            int: The status result."""
        self.ensure_daemon()
        container = self.get_container()
        if container is None:
            LOGGER.info("container %s is not present", self.settings.docker.container_name)
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

    def host_caches(self) -> tuple[HostCache, ...]:
        """Return persistent host caches in mount order."""
        return (
            RootCache("root", self.settings.dirs.root_cache, CONTAINERPATH.ROOT_CACHE),
            PackageCache("pkg", self.settings.dirs.pkg_cache, CONTAINERPATH.PKG_CACHE),
            PythonCache("python", self.settings.dirs.python_cache, CONTAINERPATH.PYTHON_CACHE),
            ModelCache("models", self.settings.dirs.models, CONTAINERPATH.MODELS),
        )

    def clean(self, *, all_images: bool = False, wipe: frozenset[str] | None = None) -> int:
        """Remove runtime containers, generated configuration, images, and host caches.

        Args:
            all_images: The all images.
            wipe: The wipe.

        Returns:
            int: The clean result."""
        self.ensure_daemon()
        self.remove_container()
        self.remove_networks()
        self._remove_effective_configs()
        kept: list[str] = []
        for cache in self.host_caches():
            if wipe is not None and cache.name in wipe:
                cache.clean()
                LOGGER.info("cleaned host cache %s (%s)", cache.name, cache.host)
            else:
                kept.append(cache.name)
        if kept:
            LOGGER.info("kept host caches: %s (clean --wipe-cache to remove them)", ", ".join(kept))
        image_names = [
            image.tags[0] for image in DockerBuilder.managed_images(self.client) if image.tags
        ]
        if not all_images:
            selected = {builder.name for builder in self.builders()}
            image_names = [name for name in image_names if name in selected]
        for image_name in image_names:
            try:
                self.client.images.remove(image_name, force=True)
                LOGGER.info("removed image %s", image_name)
            except ImageNotFound:
                LOGGER.warning("image %s does not exist", image_name)
        if all_images:
            for network in self.client.networks.list(filters={"label": "easyllama.managed=true"}):
                network.remove()
        return 0
