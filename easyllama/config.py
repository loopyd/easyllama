"""Load and manage process-wide easyllama configuration."""

from __future__ import annotations

from contextlib import suppress
from enum import IntEnum, StrEnum
from ipaddress import ip_address
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    ValidationInfo,
    field_serializer,
    field_validator,
)

from .helpers.common import (
    absolute_path,
    detect_timezone,
    image_name_for_mode,
    normalize_mode,
    project_root,
)
from .helpers.hf import HuggingFace
from .helpers.http import Http
from .helpers.logger import LOG as APP_LOG

LOGGER = APP_LOG.get(__name__)


class RUNTIME(StrEnum):
    """Supported execution environments."""

    HOST = "host"
    CONTAINER = "container"


class MODE(StrEnum):
    """Supported server modes."""

    LLAMACPP = "llamacpp"
    TURBOQUANT = "turboquant"
    QWEN = "qwen"
    SPIRITBUUN = "spiritbuun"
    LUCEBOX = "lucebox"


class IMAGE(StrEnum):
    """Supported isolated Docker image roles."""

    LLAMASWAP = "llamaswap"
    VLLM = "vllm"
    LLAMACPP = "llamacpp"
    LMCACHE = "lmcache"


class CPU_WEIGHT(IntEnum):
    """Relative CPU allocation within the host's 75% container budget."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    XHIGH = 4

    @classmethod
    def from_string(cls, value: str | CPU_WEIGHT) -> CPU_WEIGHT:
        """Resolve a JSON profile name."""
        if isinstance(value, cls):
            return value
        try:
            return cls[str(value).strip().upper()]
        except KeyError as error:
            raise ValueError(f"invalid CPU weight: {value}") from error

    def allocate(self, available: int, floor: int | None = None) -> int:
        """Return weighted CPUs, subject to the configured profile floor."""
        budget = max(1, int(available * 0.75))
        floors = {
            CPU_WEIGHT.LOW: 2,
            CPU_WEIGHT.MEDIUM: 4,
            CPU_WEIGHT.HIGH: 8,
            CPU_WEIGHT.XHIGH: 16,
        }
        floor = floors[self] if floor is None else floor
        divisors = {
            CPU_WEIGHT.LOW: 12,
            CPU_WEIGHT.MEDIUM: 3,
            CPU_WEIGHT.HIGH: 1.5,
            CPU_WEIGHT.XHIGH: 1,
        }
        return max(floor, int(budget / divisors[self]))


class RAM_WEIGHT(IntEnum):
    """Relative RAM allocation while reserving at least 16 GiB for the host."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    XHIGH = 4

    @classmethod
    def from_string(cls, value: str | RAM_WEIGHT) -> RAM_WEIGHT:
        """Resolve a JSON profile name."""
        if isinstance(value, cls):
            return value
        try:
            return cls[str(value).strip().upper()]
        except KeyError as error:
            raise ValueError(f"invalid RAM weight: {value}") from error

    def allocate(self, available_gib: float, floor: int | None = None) -> int:
        """Return weighted RAM bytes, subject to the configured profile floor."""
        gib = 1024**3
        budget = max(0, min(available_gib * 0.75, available_gib - 16))
        floors = {
            RAM_WEIGHT.LOW: 8,
            RAM_WEIGHT.MEDIUM: 16,
            RAM_WEIGHT.HIGH: 32,
            RAM_WEIGHT.XHIGH: 64,
        }
        floor = floors[self] if floor is None else floor
        divisors = {
            RAM_WEIGHT.LOW: 12,
            RAM_WEIGHT.MEDIUM: 3,
            RAM_WEIGHT.HIGH: 1.5,
            RAM_WEIGHT.XHIGH: 1,
        }
        return int(max(floor, budget / divisors[self]) * gib)


class SWAP_WEIGHT(IntEnum):
    """Relative swap allocation independent of RAM allocation."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    XHIGH = 4

    @classmethod
    def from_string(cls, value: str | SWAP_WEIGHT) -> SWAP_WEIGHT:
        """Resolve a JSON profile name."""
        if isinstance(value, cls):
            return value
        try:
            return cls[str(value).strip().upper()]
        except KeyError as error:
            raise ValueError(f"invalid swap weight: {value}") from error

    def allocate(self, available_gib: float, floor: int | None = None) -> int:
        """Return weighted swap bytes, subject to the configured profile floor."""
        gib = 1024**3
        floors = {
            SWAP_WEIGHT.LOW: 4,
            SWAP_WEIGHT.MEDIUM: 8,
            SWAP_WEIGHT.HIGH: 16,
            SWAP_WEIGHT.XHIGH: 32,
        }
        floor = floors[self] if floor is None else floor
        return int(max(floor, available_gib * int(self) / int(SWAP_WEIGHT.XHIGH)) * gib)


class CONTAINERPATH(StrEnum):
    """Paths shared with the runtime container."""

    ROOT_CACHE = "/root/.cache"
    PKG_CACHE = "/var/cache/apt"
    PYTHON_CACHE = f"{ROOT_CACHE}/pip"
    MODELS = f"{ROOT_CACHE}/huggingface/hub"
    CHAT_TEMPLATE = "/chat_template"
    MMPROJ = "/mmproj"
    LLAMA_SWAP = "/app/bin/llama-swap"


class DataModel(BaseModel):
    """Serializable, validated, immutable configuration data."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)


def config_field(default: object, *, env: str | None = None) -> Any:
    """Declare the matching environment variable suffix for a configuration field."""
    return Field(default, json_schema_extra={"env": env} if env else {})


def _merge(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Recursively merge JSON configuration into defaults."""
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = value


def _apply_env(model: type[DataModel], values: dict[str, Any]) -> None:
    """Recursively apply field-declared EASYLLAMA environment overrides."""
    for name, field in model.model_fields.items():
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, DataModel):
            _apply_env(annotation, values.setdefault(name, {}))
            continue
        meta = cast(dict[str, object], field.json_schema_extra or {})
        env_name = meta.get("env")
        if env_name and (value := os.environ.get(f"EASYLLAMA_{env_name}")) is not None:
            values[name] = value


class ProfileValues(DataModel):
    """Configurable floors for the four resource weights."""

    low: PositiveInt
    medium: PositiveInt
    high: PositiveInt
    xhigh: PositiveInt

    def floor(self, weight: IntEnum) -> int:
        """Return the configured floor for a weight."""
        return cast(int, getattr(self, weight.name.lower()))


class CPUProfiles(ProfileValues):
    low: PositiveInt = 2
    medium: PositiveInt = 4
    high: PositiveInt = 8
    xhigh: PositiveInt = 16

    @field_validator("low", "medium", "high", "xhigh")
    @classmethod
    def validate_floor(cls, value: int, info: ValidationInfo) -> int:
        minimum = {"low": 2, "medium": 4, "high": 8, "xhigh": 16}[cast(str, info.field_name)]
        if value < minimum:
            raise ValueError(f"{info.field_name} CPU floor must be at least {minimum}")
        return value


class RAMProfiles(ProfileValues):
    low: PositiveInt = 8
    medium: PositiveInt = 16
    high: PositiveInt = 32
    xhigh: PositiveInt = 64

    @field_validator("low", "medium", "high", "xhigh")
    @classmethod
    def validate_floor(cls, value: int, info: ValidationInfo) -> int:
        minimum = {"low": 8, "medium": 16, "high": 32, "xhigh": 64}[cast(str, info.field_name)]
        if value < minimum:
            raise ValueError(f"{info.field_name} RAM floor must be at least {minimum} GiB")
        return value


class SwapProfiles(ProfileValues):
    low: PositiveInt = 4
    medium: PositiveInt = 8
    high: PositiveInt = 16
    xhigh: PositiveInt = 32

    @field_validator("low", "medium", "high", "xhigh")
    @classmethod
    def validate_floor(cls, value: int, info: ValidationInfo) -> int:
        minimum = {"low": 4, "medium": 8, "high": 16, "xhigh": 32}[cast(str, info.field_name)]
        if value < minimum:
            raise ValueError(f"{info.field_name} swap floor must be at least {minimum} GiB")
        return value


class ProfilesConfig(DataModel):
    """Configurable CPU, RAM, and swap floors."""

    cpu: CPUProfiles = CPUProfiles()
    ram: RAMProfiles = RAMProfiles()
    swap: SwapProfiles = SwapProfiles()


class HostResources(DataModel):
    """Configured host capacity available to containers."""

    cpus: PositiveInt = config_field(os.cpu_count() or 1, env="AVAILABLE_CPUS")
    ram_gib: float | None = config_field(None, env="AVAILABLE_RAM_GIB")
    swap_gib: float | None = config_field(None, env="AVAILABLE_SWAP_GIB")

    @field_validator("ram_gib", "swap_gib")
    @classmethod
    def positive_capacity(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("resource capacity cannot be negative")
        return value


class HardwareProfile(DataModel):
    """Independent container CPU, RAM, and swap profiles."""

    cpu: CPU_WEIGHT
    ram: RAM_WEIGHT
    swap: SWAP_WEIGHT

    @field_validator("cpu", mode="before")
    @classmethod
    def parse_cpu(cls, value: object) -> CPU_WEIGHT:
        return CPU_WEIGHT.from_string(cast(str | CPU_WEIGHT, value))

    @field_validator("ram", mode="before")
    @classmethod
    def parse_ram(cls, value: object) -> RAM_WEIGHT:
        return RAM_WEIGHT.from_string(cast(str | RAM_WEIGHT, value))

    @field_validator("swap", mode="before")
    @classmethod
    def parse_swap(cls, value: object) -> SWAP_WEIGHT:
        return SWAP_WEIGHT.from_string(cast(str | SWAP_WEIGHT, value))

    @field_serializer("cpu", "ram", "swap")
    def serialize_weight(self, value: IntEnum) -> str:
        return value.name.lower()


class ResourcesConfig(DataModel):
    """Host capacity, profile floors, and service-role assignments."""

    host: HostResources = HostResources()
    profiles: ProfilesConfig = ProfilesConfig()
    roles: dict[IMAGE, HardwareProfile]

    def profile(self, image: IMAGE) -> HardwareProfile:
        """Return the configured profile for an image role."""
        try:
            return self.roles[image]
        except KeyError as error:
            raise ValueError(f"missing resource profile for {image}") from error


class CredentialsConfig(DataModel):
    """External-service credentials."""

    hf_token: str | None = config_field(None, env="HF_TOKEN")
    api_key: str | None = config_field(None, env="API_KEY")


class ConfigDirs(DataModel):
    """Project paths."""

    root: Path = config_field(".", env="ROOT")
    models: Path = config_field("cache/models", env="MODELS_DIR")
    root_cache: Path = config_field("cache/root", env="ROOT_CACHE_DIR")
    pkg_cache: Path = config_field("cache/pkg", env="PKG_CACHE_DIR")
    python_cache: Path = config_field("cache/python", env="PYTHON_CACHE_DIR")
    mmproj: Path = config_field("mmproj", env="MMPROJ_DIR")
    chat_template: Path = config_field("chat_template", env="CHAT_TEMPLATE_DIR")
    runtime: Path = config_field(".runtime", env="RUNTIME_DIR")

    @field_validator("*", mode="after")
    @classmethod
    def absolute_paths(cls, value: Path, info: ValidationInfo) -> Path:
        """Resolve all project paths against the configured root."""
        if info.field_name == "root":
            return value.expanduser().resolve()
        return absolute_path(info.data.get("root", project_root()), str(value))


class ConfigRepo(DataModel):
    """Source repository URL and revision."""

    url: str
    ref: str


class ModeSettings(DataModel):
    """Repository and service roles for one runtime mode."""

    repo: ConfigRepo
    services: tuple[IMAGE, ...]
    hub: ConfigRepo | None = None


class LMCacheConfig(DataModel):
    """LMCache server settings."""

    chunk_size: PositiveInt = config_field(1600, env="LMCACHE_CHUNK_SIZE")
    l1_size_gb: PositiveInt = config_field(16, env="LMCACHE_L1_SIZE_GB")


class RuntimeConfig(DataModel):
    """Server process and network settings."""

    environment: RUNTIME = config_field(RUNTIME.HOST, env="RUNTIME_MODE")
    mode: MODE = config_field(MODE.LLAMACPP, env="MODE")
    host: str = config_field("127.0.0.1", env="HOST")
    host_port: int = config_field(8080, env="HOST_PORT")
    container_port: int = config_field(8080, env="CONTAINER_PORT")
    pids_limit: PositiveInt = config_field(256, env="PIDS_LIMIT")

    @field_validator("environment", mode="before")
    @classmethod
    def validate_environment(cls, value: object) -> RUNTIME:
        from .helpers.docker import detect_runtime_mode

        return detect_runtime_mode(str(value) if value is not None else None)

    @field_validator("mode", mode="before")
    @classmethod
    def validate_mode(cls, value: object) -> MODE:
        return MODE(normalize_mode(str(value) if value is not None else None))

    @field_validator("host", mode="before")
    @classmethod
    def validate_host(cls, value: object) -> str:
        """Normalize an IPv4, IPv6, or RFC-compliant hostname."""
        host = str(value).strip()
        try:
            return ip_address(host.removeprefix("[").removesuffix("]")).compressed
        except ValueError:
            pass
        try:
            ascii_host = host.rstrip(".").encode("idna").decode("ascii").lower()
        except UnicodeError as error:
            raise ValueError("host must be a valid IPv4, IPv6, or hostname") from error
        labels = ascii_host.split(".")
        if (
            not ascii_host
            or len(ascii_host) > 253
            or any(
                len(label) > 63 or not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", label)
                for label in labels
            )
        ):
            raise ValueError("host must be a valid IPv4, IPv6, or hostname")
        return ascii_host

    @field_validator("host_port", "container_port")
    @classmethod
    def validate_port(cls, value: int) -> int:
        if not 1 <= value <= 65535:
            raise ValueError("port must be between 1 and 65535")
        return value


class CudaConfig(DataModel):
    """CUDA architecture build settings."""

    default: str = config_field("120", env="DEFAULT_CUDA_ARCHITECTURES")
    cmake: str = config_field("auto", env="CMAKE_CUDA_ARCHITECTURES")


class DockerConfig(DataModel):
    """Docker image, container, and build settings."""

    image_name: str = config_field("easyllama", env="IMAGE_NAME")
    image_tag: str = "cuda13"
    container_name: str = config_field("easyllama-server-swap", env="CONTAINER_NAME")
    network_mode: Literal["bridge", "host"] = config_field("bridge", env="NETWORK_MODE")
    cuda: CudaConfig = CudaConfig()


class LocaleConfig(DataModel):
    """Host locale forwarded to containers."""

    timezone: str = config_field("UTC", env="HOST_TZ")
    lang: str = config_field("C.UTF-8", env="HOST_LANG")
    lc_all: str = config_field("C.UTF-8", env="HOST_LC_ALL")


class WarmupConfig(DataModel):
    """Model warmup polling settings."""

    timeout: PositiveInt = config_field(1800, env="WARMUP_TIMEOUT")
    poll_interval: float = config_field(2.0, env="WARMUP_POLL_INTERVAL")

    @field_validator("poll_interval")
    @classmethod
    def positive_poll_interval(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("poll interval must be positive")
        return value


class Config(DataModel):
    """Store and operate on process-wide easyllama configuration.

    Attributes:
        root_dir: The root dir (Path).
        runtime_mode: The runtime mode (str).
        mode: The mode (str).
        image_name: The image name (str).
        container_name: The container name (str).
        host: The host bind address (str).
        host_port: The host port (int).
        container_port: The container port (int).
        pids_limit: The pids limit (int).
        models_dir: The models cache dir (Path).
        root_cache_dir: The root cache dir (Path).
        pkg_cache_dir: The package-manager cache dir (Path).
        python_cache_dir: The Python cache dir (Path).
        mmproj_dir: The mmproj dir (Path).
        chat_template_dir: The chat template dir (Path).
        auth_file: The auth file (Path).
        auth_example_file: The auth example file (Path).
        runtime_dir: The runtime dir (Path).
        config_override: The config override (Path | None).
        modes: Mode repositories and service roles.
        resources: Host capacity and resource profiles.
        host_tz: The host tz (str).
        host_lang: The host lang (str).
        host_lc_all: The host lc all (str).
    _instance: The instance."""

    dirs: ConfigDirs
    runtime: RuntimeConfig
    docker: DockerConfig
    resources: ResourcesConfig
    modes: dict[MODE, ModeSettings]
    locale: LocaleConfig
    lmcache: LMCacheConfig
    llama_swap_override: Path | None = config_field(None, env="LS_CONFIG_FILE")
    warmup: WarmupConfig = WarmupConfig()
    credentials: CredentialsConfig = CredentialsConfig()

    @field_validator("llama_swap_override", mode="after")
    @classmethod
    def resolve_override(cls, value: Path | None, info: ValidationInfo) -> Path | None:
        root = (info.context or {}).get("root")
        return absolute_path(Path(root), str(value)) if value and root else value

    @classmethod
    def load(
        cls,
        *,
        config_file: str | Path | None = None,
        mode_override: MODE | str | None = None,
        runtime_mode_override: RUNTIME | str | None = None,
        host_override: str | None = None,
    ) -> Config:
        """Load defaults, optional JSON, environment, then CLI overrides."""
        root = project_root()
        values: dict[str, Any] = {
            "dirs": {"root": root},
            "runtime": {},
            "docker": {},
            "resources": {
                "host": {},
                "profiles": {},
                "roles": {
                    "llamaswap": {"cpu": "low", "ram": "low", "swap": "low"},
                    "llamacpp": {"cpu": "xhigh", "ram": "medium", "swap": "xhigh"},
                    "vllm": {"cpu": "xhigh", "ram": "medium", "swap": "medium"},
                    "lmcache": {"cpu": "medium", "ram": "medium", "swap": "medium"},
                },
            },
            "locale": {
                "timezone": detect_timezone(),
                "lang": os.environ.get("LANG", "C.UTF-8"),
                "lc_all": os.environ.get("LC_ALL", os.environ.get("LANG", "C.UTF-8")),
            },
            "lmcache": {},
            "modes": {
                "llamacpp": {
                    "repo": {
                        "url": "https://github.com/Luce-Org/llama.cpp.git",
                        "ref": "luce-dflash",
                    },
                    "services": ["llamaswap", "llamacpp"],
                },
                "turboquant": {
                    "repo": {
                        "url": "https://github.com/TheTom/llama-cpp-turboquant.git",
                        "ref": "feature/turboquant-kv-cache",
                    },
                    "services": ["llamaswap", "llamacpp"],
                },
                "qwen": {
                    "repo": {"url": "https://github.com/ggml-org/llama.cpp.git", "ref": "master"},
                    "services": ["llamaswap", "vllm", "lmcache", "llamacpp"],
                },
                "spiritbuun": {
                    "repo": {
                        "url": "https://github.com/spiritbuun/buun-llama-cpp.git",
                        "ref": "master",
                    },
                    "services": ["llamaswap", "llamacpp"],
                },
                "lucebox": {
                    "repo": {
                        "url": "https://github.com/Luce-Org/llama.cpp.git",
                        "ref": "luce-dflash",
                    },
                    "services": ["llamaswap", "llamacpp"],
                    "hub": {"url": "https://github.com/Luce-Org/lucebox-hub.git", "ref": "main"},
                },
            },
            "warmup": {},
            "credentials": {},
        }
        path = Path(config_file) if config_file else root / "config.json"
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise SystemExit(f"invalid configuration file {path}: {error}") from error
            if not isinstance(payload, dict):
                raise SystemExit(f"invalid configuration file {path}: expected a JSON object")
            _merge(values, payload)
        elif config_file:
            raise SystemExit(f"configuration file not found: {path}")
        _apply_env(cls, values)
        credentials = values.setdefault("credentials", {})
        if value := os.environ.get("HF_TOKEN"):
            credentials["hf_token"] = value
        if value := os.environ.get("API_KEY"):
            credentials["api_key"] = value
        for mode in values.get("modes", {}).values():
            repo = mode.get("repo", {})
            repo["url"] = os.environ.get("EASYLLAMA_LLAMA_CPP_REPO", repo.get("url", ""))
            repo["ref"] = os.environ.get("EASYLLAMA_LLAMA_CPP_REF", repo.get("ref", ""))
        lucebox = values.get("modes", {}).get("lucebox", {})
        hub = lucebox.get("hub", {})
        hub["url"] = os.environ.get("EASYLLAMA_LUCEBOX_HUB_REPO", hub.get("url", ""))
        hub["ref"] = os.environ.get("EASYLLAMA_LUCEBOX_HUB_REF", hub.get("ref", ""))
        runtime = values.setdefault("runtime", {})
        if mode_override is not None:
            runtime["mode"] = mode_override
        if runtime_mode_override is not None:
            runtime["environment"] = runtime_mode_override
        if host_override is not None:
            runtime["host"] = host_override
        try:
            return cls.model_validate(values, context={"root": root})
        except ValueError as error:
            raise SystemExit(f"invalid configuration: {error}") from error

    def image_for_mode(self, mode: MODE | str) -> str:
        """Return the image name for a server mode."""
        return image_name_for_mode(self.docker.image_name, self.docker.image_tag, str(mode))

    def load_auth(self) -> CredentialsConfig:
        """Return credentials centralized in the configuration model."""
        return self.credentials

    def resolve_ls_config(self) -> Path:
        """Resolve the active llama-swap configuration file.

        Returns:
            Path: The resolve ls config result.

        Raises:
            SystemExit: If the resolve ls config operation cannot be completed."""
        if self.llama_swap_override is not None:
            if not self.llama_swap_override.is_file():
                raise SystemExit(
                    "no llama-swap config found at "
                    f"{self.llama_swap_override}; set EASYLLAMA_LS_CONFIG_FILE "
                    "to a readable file"
                )
            return self.llama_swap_override

        active = self.dirs.root / f"config/config.{self.runtime.mode}.yml"
        example = active.with_suffix(".yml.example")
        if active.is_file():
            return active
        if example.is_file():
            LOGGER.info("Using %s; create %s for local overrides", example.name, active.name)
            return example
        raise SystemExit(
            f"no llama-swap config found for {self.runtime.mode} mode; "
            f"set EASYLLAMA_LS_CONFIG_FILE or create {active} from {example}"
        )

    def effective_config_path(self, auth: CredentialsConfig) -> tuple[Path, str]:
        """Create an authenticated runtime configuration when needed.

        Args:
            auth: The auth.

        Returns:
            tuple[Path, str]: The effective config path result."""
        config_path = self.resolve_ls_config()
        if not auth.api_key:
            return config_path, f"/app/config.d/{config_path.name}"

        self.dirs.runtime.mkdir(parents=True, exist_ok=True)
        effective_path = self.dirs.runtime / f"{config_path.name}.effective.yaml"
        payload = f"apiKeys:\n  - {json.dumps(auth.api_key)}\n" + config_path.read_text(
            encoding="utf-8"
        )
        fd, temporary_name = tempfile.mkstemp(dir=self.dirs.runtime, prefix=".effective-")
        temporary_path = Path(temporary_name)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as effective_file:
                effective_file.write(payload)
            temporary_path.replace(effective_path)
        except BaseException:
            with suppress(OSError):
                os.close(fd)
            temporary_path.unlink(missing_ok=True)
            raise
        return effective_path, f"/app/config.d/{effective_path.name}"

    def container_config_path(self) -> Path:
        """Resolve the llama-swap configuration visible inside the container.

        Returns:
            Path: The container config path result.

        Raises:
            SystemExit: If the container config path operation cannot be completed."""
        if self.llama_swap_override is not None:
            if not self.llama_swap_override.is_file():
                raise SystemExit(f"container config not found at {self.llama_swap_override}")
            return self.llama_swap_override

        config_dir = Path("/app/config.d")
        if config_dir.is_dir():
            matches = sorted(list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml")))
            if matches:
                return matches[0]
        raise SystemExit(
            "no container config found under /app/config.d; "
            "mount one with run.sh start or set "
            "EASYLLAMA_LS_CONFIG_FILE inside the container"
        )

    def listen_url(self) -> str:
        """Return the llama-swap base URL for the active runtime.

        Returns:
            str: The listen url result."""
        port = (
            self.runtime.host_port
            if self.runtime.environment == RUNTIME.HOST
            else self.runtime.container_port
        )
        host = self.runtime.host
        url_host = f"[{host}]" if ":" in host else host
        return f"http://{url_host}:{port}"

    def resolved_api_key(self, auth: CredentialsConfig) -> str | None:
        """Resolve the API key from credentials or llama-swap configuration.

        Args:
            auth: The auth.

        Returns:
            str | None: The resolved api key result."""
        if auth.api_key:
            return auth.api_key
        config_path = (
            self.container_config_path()
            if self.runtime.environment == RUNTIME.CONTAINER
            else self.resolve_ls_config()
        )
        if not config_path.is_file():
            return None
        in_api_keys = False
        for raw_line in config_path.read_text(encoding="utf-8").splitlines():
            if not in_api_keys:
                if raw_line.strip() == "apiKeys:":
                    in_api_keys = True
                continue
            if raw_line and not raw_line.startswith((" ", "\t", "-")):
                break
            stripped = raw_line.strip()
            if stripped.startswith("-"):
                value = stripped[1:].strip().split(" #", 1)[0].strip().strip('"')
                env_match = re.fullmatch(r"\$\{env\.([A-Za-z_][A-Za-z0-9_]*)\}", value)
                if env_match:
                    return os.environ.get(env_match.group(1))
                return value or None
        return None

    def map_mmproj(self, auth: CredentialsConfig, source: str) -> str:
        """Resolve or download a multimodal projector into its container path.

        Args:
            auth: The auth.
            source: The source.

        Returns:
            str: The map mmproj result.

        Raises:
            SystemExit: If the map mmproj operation cannot be completed."""
        if not source:
            return ""
        if re.match(r"^https?://", source):
            url = source
            if re.match(r"^https?://huggingface\.co/.*/blob/", url):
                url = url.replace("/blob/", "/resolve/", 1)
            headers = {"Authorization": f"Bearer {auth.hf_token}"} if auth.hf_token else {}
            http = Http(url, headers=headers)
            filename = http.filename
            if not filename:
                raise SystemExit(f"could not infer mmproj filename from URL: {source}")
            self.dirs.mmproj.mkdir(parents=True, exist_ok=True)
            output_path = self.dirs.mmproj / filename
            expected_size = http.content_length()
            if not output_path.is_file() or (
                expected_size is not None and output_path.stat().st_size != expected_size
            ):
                LOGGER.info("Downloading mmproj from %s", source)
                temp_path = output_path.with_suffix(output_path.suffix + ".part")
                http.download(temp_path)
                if expected_size is not None and temp_path.stat().st_size != expected_size:
                    raise SystemExit(
                        f"mmproj download incomplete for {source}: "
                        f"got {temp_path.stat().st_size} bytes, "
                        f"expected {expected_size}"
                    )
                temp_path.replace(output_path)
                LOGGER.info("Downloaded mmproj to %s", output_path)
            return f"{CONTAINERPATH.MMPROJ}/{filename}"
        if source.startswith(f"{CONTAINERPATH.MMPROJ}/"):
            return source
        if source.startswith(str(self.dirs.mmproj) + "/"):
            return f"{CONTAINERPATH.MMPROJ}/{Path(source).relative_to(self.dirs.mmproj).as_posix()}"
        if source.startswith("mmproj/"):
            return f"{CONTAINERPATH.MMPROJ}/{source.removeprefix('mmproj/')}"
        if "/" not in source and "\\" not in source:
            return f"{CONTAINERPATH.MMPROJ}/{source}"
        if Path(source).is_absolute():
            raise SystemExit(
                "EASYLLAMA_MMPROJ_FILE must be in "
                f"{self.dirs.mmproj}, use mmproj/<file>, or provide a URL"
            )
        return f"{CONTAINERPATH.MMPROJ}/{source.removeprefix('./')}"

    def mmproj_arg(self, auth: CredentialsConfig) -> str:
        """Build the optional llama.cpp multimodal projector argument.

        Args:
            auth: The auth.

        Returns:
            str: The mmproj arg result."""
        source = os.environ.get("EASYLLAMA_MMPROJ_FILE")
        hf_mmproj = os.environ.get("EASYLLAMA_HF_MMPROJ")
        if not source and hf_mmproj:
            source = HuggingFace.mmproj_url(hf_mmproj)
        if not source:
            return ""
        return f"--mmproj {self.map_mmproj(auth, source)}"
