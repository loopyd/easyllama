from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import tomllib
import urllib.error
import urllib.parse
import urllib.request

from .logger import get_logger

LOGGER = get_logger(__name__)
RUNTIME_HOST = "host"
RUNTIME_CONTAINER = "container"
MODE_BASIC = "basic"
MODE_TURBOQUANT = "turboquant"
MODE_LUCEBOX = "lucebox"
VALID_MODES = {MODE_BASIC, MODE_TURBOQUANT, MODE_LUCEBOX}
MODELS_DIR_CONTAINER = "/root/.cache/huggingface/hub"
CHAT_TEMPLATE_DIR_CONTAINER = "/chat_template"
MMPROJ_DIR_CONTAINER = "/mmproj"
LLAMA_SWAP_BIN = "/app/bin/llama-swap"
HF_URL_BASE = "https://huggingface.co"


@dataclass(frozen=True)
class ModeConfig:
    active: Path
    example: Path


@dataclass(frozen=True)
class ResolvedAuth:
    hf_token: str | None
    api_key: str | None


@dataclass(frozen=True)
class Settings:
    root_dir: Path
    runtime_mode: str
    mode: str
    image_name: str
    container_name: str
    host_port: int
    container_port: int
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
    turboquant_llama_cpp_repo: str
    turboquant_llama_cpp_ref: str
    lucebox_hub_repo: str
    lucebox_hub_ref: str
    host_tz: str
    host_lang: str
    host_lc_all: str

    def with_mode(self, mode: str) -> Settings:
        return load_settings(mode_override=mode, runtime_mode_override=self.runtime_mode)


def project_root() -> Path:
    env_root = os.environ.get("EASYLLAMA_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parents[1]


def load_pyproject(root_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    pyproject_path = root_dir / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    tool_config = data.get("tool", {}).get("easyllama", {})
    defaults = dict(tool_config.get("defaults", {}))
    configs = dict(tool_config.get("configs", {}))
    return defaults, configs


def normalize_mode(value: str | None) -> str:
    selected = (value or MODE_BASIC).strip().lower()
    if selected not in VALID_MODES:
        allowed = ", ".join((MODE_BASIC, MODE_TURBOQUANT, MODE_LUCEBOX))
        raise SystemExit(f"unsupported mode: {selected}; allowed: {allowed}")
    return selected


def detect_runtime_mode(value: str | None = None) -> str:
    selected = value or os.environ.get("LLAMACPP_RUNTIME_MODE")
    if not selected:
        return RUNTIME_CONTAINER if Path("/.dockerenv").exists() else RUNTIME_HOST
    if selected not in {RUNTIME_HOST, RUNTIME_CONTAINER}:
        allowed = ", ".join((RUNTIME_HOST, RUNTIME_CONTAINER))
        raise SystemExit(f"unsupported LLAMACPP_RUNTIME_MODE={selected}; allowed: {allowed}")
    return selected


def absolute_path(root_dir: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (root_dir / candidate).resolve()


def detect_timezone() -> str:
    if os.environ.get("TZ"):
        return os.environ["TZ"]
    timezone_path = Path("/etc/timezone")
    if timezone_path.is_file():
        return timezone_path.read_text(encoding="utf-8").strip()
    timedatectl = shutil_which("timedatectl")
    if timedatectl:
        result = subprocess.run(
            [timedatectl, "show", "-p", "Timezone", "--value"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return "UTC"


def shutil_which(executable: str) -> str | None:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / executable
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def image_name_for_mode(image_name_base: str, image_tag_base: str, mode: str) -> str:
    explicit = os.environ.get("LLAMACPP_IMAGE_NAME")
    base = explicit or image_name_base
    if ":" in base:
        repository, tag = base.rsplit(":", 1)
        return f"{repository}:{tag}-{mode}"
    return f"{base}:{image_tag_base}-{mode}"


def load_settings(
    *,
    mode_override: str | None = None,
    runtime_mode_override: str | None = None,
) -> Settings:
    root_dir = project_root()
    defaults, config_defaults = load_pyproject(root_dir)
    runtime_mode = detect_runtime_mode(runtime_mode_override)
    mode = normalize_mode(mode_override or os.environ.get("LLAMACPP_MODE"))

    image_name_base = str(defaults["image_name_base"])
    image_tag_base = str(defaults["image_tag_base"])
    configs = {
        MODE_BASIC: ModeConfig(
            active=absolute_path(root_dir, str(config_defaults[MODE_BASIC])),
            example=absolute_path(root_dir, str(config_defaults["basic_example"])),
        ),
        MODE_TURBOQUANT: ModeConfig(
            active=absolute_path(root_dir, str(config_defaults[MODE_TURBOQUANT])),
            example=absolute_path(root_dir, str(config_defaults["turboquant_example"])),
        ),
        MODE_LUCEBOX: ModeConfig(
            active=absolute_path(root_dir, str(config_defaults[MODE_LUCEBOX])),
            example=absolute_path(root_dir, str(config_defaults["lucebox_example"])),
        ),
    }
    config_override = os.environ.get("LLAMACPP_LS_CONFIG_FILE")
    return Settings(
        root_dir=root_dir,
        runtime_mode=runtime_mode,
        mode=mode,
        image_name=image_name_for_mode(image_name_base, image_tag_base, mode),
        container_name=os.environ.get("LLAMACPP_CONTAINER_NAME", str(defaults["container_name"])),
        host_port=int(os.environ.get("LLAMACPP_HOST_PORT", defaults["host_port"])),
        container_port=int(os.environ.get("LLAMACPP_CONTAINER_PORT", defaults["container_port"])),
        models_dir=absolute_path(
            root_dir, os.environ.get("LLAMACPP_MODELS_DIR", str(defaults["models_dir"]))
        ),
        mmproj_dir=absolute_path(
            root_dir, os.environ.get("LLAMACPP_MMPROJ_DIR", str(defaults["mmproj_dir"]))
        ),
        chat_template_dir=absolute_path(
            root_dir,
            os.environ.get("LLAMACPP_CHAT_TEMPLATE_DIR", str(defaults["chat_template_dir"])),
        ),
        auth_file=absolute_path(
            root_dir, os.environ.get("LLAMACPP_AUTH_FILE", str(defaults["auth_file"]))
        ),
        auth_example_file=absolute_path(root_dir, str(defaults["auth_example_file"])),
        runtime_dir=(root_dir / ".runtime").resolve(),
        config_override=absolute_path(root_dir, config_override) if config_override else None,
        configs=configs,
        default_cuda_architectures=os.environ.get(
            "LLAMACPP_DEFAULT_CUDA_ARCHITECTURES",
            str(defaults["cuda_default_architectures"]),
        ),
        cmake_cuda_architectures=os.environ.get("LLAMACPP_CMAKE_CUDA_ARCHITECTURES", "auto"),
        llama_cpp_repo=os.environ.get("LLAMACPP_LLAMA_CPP_REPO", str(defaults["llama_cpp_repo"])),
        llama_cpp_ref=os.environ.get("LLAMACPP_LLAMA_CPP_REF", str(defaults["llama_cpp_ref"])),
        turboquant_llama_cpp_repo=os.environ.get(
            "LLAMACPP_TURBOQUANT_LLAMA_CPP_REPO",
            str(defaults["turboquant_llama_cpp_repo"]),
        ),
        turboquant_llama_cpp_ref=os.environ.get(
            "LLAMACPP_TURBOQUANT_LLAMA_CPP_REF",
            str(defaults["turboquant_llama_cpp_ref"]),
        ),
        lucebox_hub_repo=os.environ.get(
            "LLAMACPP_LUCEBOX_HUB_REPO", str(defaults["lucebox_hub_repo"])
        ),
        lucebox_hub_ref=os.environ.get(
            "LLAMACPP_LUCEBOX_HUB_REF", str(defaults["lucebox_hub_ref"])
        ),
        host_tz=os.environ.get("LLAMACPP_HOST_TZ", detect_timezone()),
        host_lang=os.environ.get("LLAMACPP_HOST_LANG", os.environ.get("LANG", "C.UTF-8")),
        host_lc_all=os.environ.get(
            "LLAMACPP_HOST_LC_ALL", os.environ.get("LC_ALL", os.environ.get("LANG", "C.UTF-8"))
        ),
    )


def load_auth(settings: Settings) -> ResolvedAuth:
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("LLAMACPP_HF_TOKEN")
    api_key = os.environ.get("LLAMACPP_API_KEY") or os.environ.get("API_KEY")
    if hf_token and api_key:
        return ResolvedAuth(hf_token=hf_token, api_key=api_key)

    source_path: Path | None = None
    if settings.auth_file.is_file():
        source_path = settings.auth_file
    elif settings.auth_example_file.is_file():
        source_path = settings.auth_example_file
        LOGGER.info(
            "Using %s; create %s for local credentials",
            settings.auth_example_file.name,
            settings.auth_file.name,
        )
    else:
        LOGGER.warning(
            "No auth file found at %s; private Hugging Face downloads may fail", settings.auth_file
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


def resolve_ls_config(settings: Settings) -> Path:
    if settings.config_override is not None:
        if not settings.config_override.is_file():
            raise SystemExit(
                "no llama-swap config found at "
                f"{settings.config_override}; set LLAMACPP_LS_CONFIG_FILE "
                "to a readable file"
            )
        return settings.config_override

    config_pair = settings.configs[settings.mode]
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
        f"no llama-swap config found for {settings.mode} mode; "
        "set LLAMACPP_LS_CONFIG_FILE or create "
        f"{config_pair.active} from {config_pair.example}"
    )


def effective_config_path(settings: Settings, auth: ResolvedAuth) -> tuple[Path, str]:
    config_path = resolve_ls_config(settings)
    if not auth.api_key:
        return config_path, f"/app/config.d/{config_path.name}"

    settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    effective_path = settings.runtime_dir / f"{config_path.name}.effective.yaml"
    escaped_api_key = auth.api_key.replace("\\", "\\\\").replace('"', '\\"')
    effective_path.write_text(
        f'apiKeys:\n  - "{escaped_api_key}"\n' + config_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with suppress(OSError):
        effective_path.chmod(0o600)
    return effective_path, f"/app/config.d/{effective_path.name}"


def container_config_path(settings: Settings) -> Path:
    if settings.config_override is not None:
        if not settings.config_override.is_file():
            raise SystemExit(f"container config not found at {settings.config_override}")
        return settings.config_override

    config_dir = Path("/app/config.d")
    if config_dir.is_dir():
        matches = sorted(list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml")))
        if matches:
            return matches[0]
    raise SystemExit(
        "no container config found under /app/config.d; "
        "mount one with run.sh start or set "
        "LLAMACPP_LS_CONFIG_FILE inside the container"
    )


def listen_url(settings: Settings) -> str:
    port = settings.host_port if settings.runtime_mode == RUNTIME_HOST else settings.container_port
    return f"http://127.0.0.1:{port}"


def resolved_api_key(settings: Settings, auth: ResolvedAuth) -> str | None:
    if auth.api_key:
        return auth.api_key
    config_path = (
        container_config_path(settings)
        if settings.runtime_mode == RUNTIME_CONTAINER
        else resolve_ls_config(settings)
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


def hf_mmproj_url(spec: str) -> str:
    parts = spec.split("/", 2)
    if len(parts) != 3 or not all(parts):
        raise SystemExit(f"LLAMACPP_HF_MMPROJ must be <owner>/<repo>/<file.gguf>; got: {spec}")
    owner, repo, filename = parts
    return f"{HF_URL_BASE}/{owner}/{repo}/blob/main/{filename}"


def _fetch_content_length(url: str, hf_token: str | None) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    if hf_token:
        request.add_header("Authorization", f"Bearer {hf_token}")
    try:
        with urllib.request.urlopen(request) as response:
            length = response.headers.get("Content-Length")
    except urllib.error.URLError:
        return None
    return int(length) if length and length.isdigit() else None


def _download_file(url: str, destination: Path, hf_token: str | None) -> None:
    request = urllib.request.Request(url)
    if hf_token:
        request.add_header("Authorization", f"Bearer {hf_token}")
    with urllib.request.urlopen(request) as response, destination.open("wb") as file_handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            file_handle.write(chunk)


def map_mmproj(settings: Settings, auth: ResolvedAuth, source: str) -> str:
    if not source:
        return ""
    if re.match(r"^https?://", source):
        url = source
        if re.match(r"^https?://huggingface\.co/.*/blob/", url):
            url = url.replace("/blob/", "/resolve/", 1)
        filename = Path(urllib.parse.urlsplit(url).path).name
        if not filename:
            raise SystemExit(f"could not infer mmproj filename from URL: {source}")
        settings.mmproj_dir.mkdir(parents=True, exist_ok=True)
        output_path = settings.mmproj_dir / filename
        expected_size = _fetch_content_length(url, auth.hf_token)
        if not output_path.is_file() or (
            expected_size is not None and output_path.stat().st_size != expected_size
        ):
            LOGGER.info("Downloading mmproj from %s", source)
            temp_path = output_path.with_suffix(output_path.suffix + ".part")
            _download_file(url, temp_path, auth.hf_token)
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
    if source.startswith(str(settings.mmproj_dir) + "/"):
        return f"{MMPROJ_DIR_CONTAINER}/{Path(source).relative_to(settings.mmproj_dir).as_posix()}"
    if source.startswith("mmproj/"):
        return f"{MMPROJ_DIR_CONTAINER}/{source.removeprefix('mmproj/')}"
    if "/" not in source:
        return f"{MMPROJ_DIR_CONTAINER}/{source}"
    if Path(source).is_absolute():
        raise SystemExit(
            "LLAMACPP_MMPROJ_FILE must be in "
            f"{settings.mmproj_dir}, use mmproj/<file>, or provide a URL"
        )
    return f"{MMPROJ_DIR_CONTAINER}/{source.removeprefix('./')}"


def mmproj_arg(settings: Settings, auth: ResolvedAuth) -> str:
    source = os.environ.get("LLAMACPP_MMPROJ_FILE")
    if not source and os.environ.get("LLAMACPP_HF_MMPROJ"):
        source = hf_mmproj_url(os.environ["LLAMACPP_HF_MMPROJ"])
    if not source and settings.mmproj_dir.is_dir():
        candidates = sorted(settings.mmproj_dir.glob("*.gguf"))
        if len(candidates) == 1:
            source = str(candidates[0])
            LOGGER.info("Using lone mmproj asset at %s", source)
    if not source:
        return ""
    return f"--mmproj {map_mmproj(settings, auth, source)}"
