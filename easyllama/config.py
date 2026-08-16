"""Load and manage process-wide easyllama configuration."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile
from typing import ClassVar

from helpers.common import (
    absolute_path,
    detect_timezone,
    image_name_for_mode,
    load_pyproject,
    normalize_mode,
    project_root,
)
from helpers.hf import HuggingFace
from helpers.http import Http
from helpers.logger import LOG as APP_LOG
from servers import mode_names

LOGGER = APP_LOG.get(__name__)
RUNTIME_HOST = "host"
RUNTIME_CONTAINER = "container"
MODE_BASIC = "basic"
MODE_TURBOQUANT = "turboquant"
MODE_MTP = "mtp"
MODE_SPIRITBUUN = "spiritbuun"
MODE_LUCEBOX = "lucebox"
MODELS_DIR_CONTAINER = "/root/.cache/huggingface/hub"
CHAT_TEMPLATE_DIR_CONTAINER = "/chat_template"
MMPROJ_DIR_CONTAINER = "/mmproj"
LLAMA_SWAP_BIN = "/app/bin/llama-swap"


@dataclass(frozen=True)
class ModeConfig:
    """Pair active and example configuration paths.

    Attributes:
        active: The active (Path).
        example: The example (Path)."""

    active: Path
    example: Path


@dataclass(frozen=True)
class ResolvedAuth:
    """Hold resolved external-service credentials.

    Attributes:
        hf_token: The hf token (str | None).
        api_key: The api key (str | None)."""

    hf_token: str | None
    api_key: str | None


@dataclass(init=False)
class Config:
    """Store and operate on process-wide easyllama configuration.

    Attributes:
        root_dir: The root dir (Path).
        runtime_mode: The runtime mode (str).
        mode: The mode (str).
        image_name: The image name (str).
        container_name: The container name (str).
        host_port: The host port (int).
        container_port: The container port (int).
        pids_limit: The pids limit (int).
        models_dir: The models dir (Path).
        mmproj_dir: The mmproj dir (Path).
        chat_template_dir: The chat template dir (Path).
        auth_file: The auth file (Path).
        auth_example_file: The auth example file (Path).
        runtime_dir: The runtime dir (Path).
        config_override: The config override (Path | None).
        configs: The configs (dict[str, ModeConfig]).
        default_cuda_architectures: The default cuda architectures (str).
        cmake_cuda_architectures: The cmake cuda architectures (str).
        llama_cpp_repo: The llama cpp repo (str).
        llama_cpp_ref: The llama cpp ref (str).
        lucebox_hub_repo: The lucebox hub repo (str).
        lucebox_hub_ref: The lucebox hub ref (str).
        host_tz: The host tz (str).
        host_lang: The host lang (str).
        host_lc_all: The host lc all (str).
    _instance: The instance."""

    _instance: ClassVar[Config | None] = None

    root_dir: Path
    runtime_mode: str
    mode: str
    image_name: str
    container_name: str
    host_port: int
    container_port: int
    pids_limit: int
    models_dir: Path
    mmproj_dir: Path
    chat_template_dir: Path
    auth_file: Path
    auth_example_file: Path
    runtime_dir: Path
    config_override: Path | None
    configs: dict[str, ModeConfig]
    default_cuda_architectures: str
    cmake_cuda_architectures: str
    llama_cpp_repo: str
    llama_cpp_ref: str
    lucebox_hub_repo: str
    lucebox_hub_ref: str
    host_tz: str
    host_lang: str
    host_lc_all: str

    def __new__(
        cls,
        *,
        mode_override: str | None = None,
        runtime_mode_override: str | None = None,
    ) -> Config:
        """Return the process-wide singleton instance.

        Args:
            mode_override: The mode override.
            runtime_mode_override: The runtime mode override.

        Returns:
            Config: The new result."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        *,
        mode_override: str | None = None,
        runtime_mode_override: str | None = None,
    ) -> None:
        """Initialize the instance.

        Args:
            mode_override: The mode override.
            runtime_mode_override: The runtime mode override."""
        self.load(mode_override=mode_override, runtime_mode_override=runtime_mode_override)

    @staticmethod
    def env(name: str, default: str | None = None) -> str | None:
        """Read an EASYLLAMA-prefixed environment setting.

        Args:
            name: The name.
            default: The default.

        Returns:
            str | None: The env result."""
        return os.environ.get(f"EASYLLAMA_{name}", default)

    def load(
        self,
        *,
        mode_override: str | None = None,
        runtime_mode_override: str | None = None,
    ) -> Config:
        """Reload configuration fields from project defaults and environment overrides.

        Args:
            mode_override: The mode override.
            runtime_mode_override: The runtime mode override.

        Returns:
            Config: The load result."""
        root_dir = project_root()
        defaults, config_defaults = load_pyproject(root_dir)
        from helpers.docker import detect_runtime_mode

        runtime_mode = detect_runtime_mode(runtime_mode_override)
        mode = normalize_mode(mode_override or self.env("MODE"))

        image_name_base = str(defaults["image_name_base"])
        image_tag_base = str(defaults["image_tag_base"])
        configs = {
            mode_name: ModeConfig(
                active=absolute_path(root_dir, str(config_defaults[mode_name])),
                example=absolute_path(root_dir, str(config_defaults[f"{mode_name}_example"])),
            )
            for mode_name in mode_names()
        }
        config_override = self.env("LS_CONFIG_FILE")
        values = dict(
            root_dir=root_dir,
            runtime_mode=runtime_mode,
            mode=mode,
            image_name=image_name_for_mode(image_name_base, image_tag_base, mode),
            container_name=self.env("CONTAINER_NAME", str(defaults["container_name"]))
            or str(defaults["container_name"]),
            host_port=int(
                self.env("HOST_PORT", str(defaults["host_port"])) or str(defaults["host_port"])
            ),
            container_port=int(
                self.env("CONTAINER_PORT", str(defaults["container_port"]))
                or str(defaults["container_port"])
            ),
            pids_limit=int(str(defaults.get("pids_limit", 256))),
            models_dir=absolute_path(
                root_dir,
                self.env("MODELS_DIR", str(defaults["models_dir"])) or str(defaults["models_dir"]),
            ),
            mmproj_dir=absolute_path(
                root_dir,
                self.env("MMPROJ_DIR", str(defaults["mmproj_dir"])) or str(defaults["mmproj_dir"]),
            ),
            chat_template_dir=absolute_path(
                root_dir,
                self.env("CHAT_TEMPLATE_DIR", str(defaults["chat_template_dir"]))
                or str(defaults["chat_template_dir"]),
            ),
            auth_file=absolute_path(
                root_dir,
                self.env("AUTH_FILE", str(defaults["auth_file"])) or str(defaults["auth_file"]),
            ),
            auth_example_file=absolute_path(root_dir, str(defaults["auth_example_file"])),
            runtime_dir=(root_dir / ".runtime").resolve(),
            config_override=absolute_path(root_dir, config_override) if config_override else None,
            configs=configs,
            default_cuda_architectures=self.env(
                "DEFAULT_CUDA_ARCHITECTURES", str(defaults["cuda_default_architectures"])
            )
            or str(defaults["cuda_default_architectures"]),
            cmake_cuda_architectures=self.env("CMAKE_CUDA_ARCHITECTURES", "auto") or "auto",
            llama_cpp_repo=self.env("LLAMA_CPP_REPO", str(defaults["llama_cpp_repo"]))
            or str(defaults["llama_cpp_repo"]),
            llama_cpp_ref=self.env("LLAMA_CPP_REF", str(defaults["llama_cpp_ref"]))
            or str(defaults["llama_cpp_ref"]),
            lucebox_hub_repo=self.env("LUCEBOX_HUB_REPO", str(defaults["lucebox_hub_repo"]))
            or str(defaults["lucebox_hub_repo"]),
            lucebox_hub_ref=self.env("LUCEBOX_HUB_REF", str(defaults["lucebox_hub_ref"]))
            or str(defaults["lucebox_hub_ref"]),
            host_tz=self.env("HOST_TZ", detect_timezone()) or detect_timezone(),
            host_lang=self.env("HOST_LANG", os.environ.get("LANG", "C.UTF-8")) or "C.UTF-8",
            host_lc_all=self.env(
                "HOST_LC_ALL", os.environ.get("LC_ALL", os.environ.get("LANG", "C.UTF-8"))
            )
            or "C.UTF-8",
        )

        for name, value in values.items():
            setattr(self, name, value)
        return self

    def image_for_mode(self, mode: str) -> str:
        """Return the image name for a server mode.

        Args:
            mode: The mode.

        Returns:
            str: The image for mode result."""
        defaults, _ = load_pyproject(self.root_dir)
        return image_name_for_mode(
            str(defaults["image_name_base"]), str(defaults["image_tag_base"]), mode
        )

    def load_auth(self) -> ResolvedAuth:
        """Load Hugging Face and API credentials.

        Returns:
            ResolvedAuth: The load auth result.

        Raises:
            SystemExit: If the load auth operation cannot be completed."""
        hf_token = os.environ.get("HF_TOKEN") or self.env("HF_TOKEN")
        api_key = self.env("API_KEY") or os.environ.get("API_KEY")
        if hf_token and api_key:
            return ResolvedAuth(hf_token=hf_token, api_key=api_key)

        source_path: Path | None = None
        if self.auth_file.is_file():
            source_path = self.auth_file
        elif self.auth_example_file.is_file():
            source_path = self.auth_example_file
            LOGGER.info(
                "Using %s; create %s for local credentials",
                self.auth_example_file.name,
                self.auth_file.name,
            )
        else:
            LOGGER.warning(
                "No auth file found at %s; private Hugging Face downloads may fail", self.auth_file
            )
            return ResolvedAuth(hf_token=hf_token, api_key=api_key)

        try:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid JSON auth file: {source_path}: {exc}") from exc

        return ResolvedAuth(
            hf_token=hf_token or payload.get("hf_token") or None,
            api_key=api_key or payload.get("api_key") or None,
        )

    def resolve_ls_config(self) -> Path:
        """Resolve the active llama-swap configuration file.

        Returns:
            Path: The resolve ls config result.

        Raises:
            SystemExit: If the resolve ls config operation cannot be completed."""
        if self.config_override is not None:
            if not self.config_override.is_file():
                raise SystemExit(
                    "no llama-swap config found at "
                    f"{self.config_override}; set EASYLLAMA_LS_CONFIG_FILE "
                    "to a readable file"
                )
            return self.config_override

        config_pair = self.configs[self.mode]
        if config_pair.active.is_file():
            return config_pair.active
        if config_pair.example.is_file():
            LOGGER.info(
                "Using %s; create %s for local overrides",
                config_pair.example.name,
                config_pair.active.name,
            )
            return config_pair.example
        raise SystemExit(
            f"no llama-swap config found for {self.mode} mode; "
            "set EASYLLAMA_LS_CONFIG_FILE or create "
            f"{config_pair.active} from {config_pair.example}"
        )

    def effective_config_path(self, auth: ResolvedAuth) -> tuple[Path, str]:
        """Create an authenticated runtime configuration when needed.

        Args:
            auth: The auth.

        Returns:
            tuple[Path, str]: The effective config path result."""
        config_path = self.resolve_ls_config()
        if not auth.api_key:
            return config_path, f"/app/config.d/{config_path.name}"

        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        effective_path = self.runtime_dir / f"{config_path.name}.effective.yaml"
        payload = f"apiKeys:\n  - {json.dumps(auth.api_key)}\n" + config_path.read_text(
            encoding="utf-8"
        )
        fd, temporary_name = tempfile.mkstemp(dir=self.runtime_dir, prefix=".effective-")
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
        if self.config_override is not None:
            if not self.config_override.is_file():
                raise SystemExit(f"container config not found at {self.config_override}")
            return self.config_override

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
        port = self.host_port if self.runtime_mode == RUNTIME_HOST else self.container_port
        return f"http://127.0.0.1:{port}"

    def resolved_api_key(self, auth: ResolvedAuth) -> str | None:
        """Resolve the API key from credentials or llama-swap configuration.

        Args:
            auth: The auth.

        Returns:
            str | None: The resolved api key result."""
        if auth.api_key:
            return auth.api_key
        config_path = (
            self.container_config_path()
            if self.runtime_mode == RUNTIME_CONTAINER
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

    def map_mmproj(self, auth: ResolvedAuth, source: str) -> str:
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
            self.mmproj_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.mmproj_dir / filename
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
            return f"{MMPROJ_DIR_CONTAINER}/{filename}"
        if source.startswith(f"{MMPROJ_DIR_CONTAINER}/"):
            return source
        if source.startswith(str(self.mmproj_dir) + "/"):
            return f"{MMPROJ_DIR_CONTAINER}/{Path(source).relative_to(self.mmproj_dir).as_posix()}"
        if source.startswith("mmproj/"):
            return f"{MMPROJ_DIR_CONTAINER}/{source.removeprefix('mmproj/')}"
        if "/" not in source:
            return f"{MMPROJ_DIR_CONTAINER}/{source}"
        if Path(source).is_absolute():
            raise SystemExit(
                "EASYLLAMA_MMPROJ_FILE must be in "
                f"{self.mmproj_dir}, use mmproj/<file>, or provide a URL"
            )
        return f"{MMPROJ_DIR_CONTAINER}/{source.removeprefix('./')}"

    def mmproj_arg(self, auth: ResolvedAuth) -> str:
        """Build the optional llama.cpp multimodal projector argument.

        Args:
            auth: The auth.

        Returns:
            str: The mmproj arg result."""
        source = self.env("MMPROJ_FILE")
        hf_mmproj = self.env("HF_MMPROJ")
        if not source and hf_mmproj:
            source = HuggingFace.mmproj_url(hf_mmproj)
        if not source:
            return ""
        return f"--mmproj {self.map_mmproj(auth, source)}"
