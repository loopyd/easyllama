from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, cast

from easyllama.cli import _wipe_caches
from easyllama.config import (
    CONTAINERPATH,
    CPU_WEIGHT,
    IMAGE,
    MODE,
    RAM_WEIGHT,
    RUNTIME,
    SWAP_WEIGHT,
    Config,
    HardwareProfile,
    ProfilesConfig,
)
from easyllama.helpers.builder import DockerBuilder
from easyllama.helpers.cache import (
    HostCache,
    ModelCache,
    PackageCache,
    PythonCache,
    RootCache,
)
from easyllama.helpers.docker import DockerRuntime
import pytest

pytestmark = pytest.mark.unit


def test_config_string_enums() -> None:
    assert str(RUNTIME.HOST) == "host"
    assert str(MODE.QWEN) == "qwen"
    assert str(CONTAINERPATH.MODELS) == "/root/.cache/huggingface/hub"
    assert CONTAINERPATH.PYTHON_CACHE == "/root/.cache/pip"
    assert CPU_WEIGHT.XHIGH.allocate(32) == 24
    assert CPU_WEIGHT.HIGH.allocate(32) == 16
    assert CPU_WEIGHT.MEDIUM.allocate(32) == 8
    assert CPU_WEIGHT.LOW.allocate(32) == 2
    assert RAM_WEIGHT.XHIGH.allocate(128) == 96 * 1024**3
    assert RAM_WEIGHT.HIGH.allocate(128) == 64 * 1024**3
    assert RAM_WEIGHT.MEDIUM.allocate(128) == 32 * 1024**3
    assert RAM_WEIGHT.LOW.allocate(128) == 8 * 1024**3
    assert CPU_WEIGHT.LOW.allocate(1) == 2
    assert CPU_WEIGHT.MEDIUM.allocate(1) == 4
    assert CPU_WEIGHT.HIGH.allocate(1) == 8
    assert CPU_WEIGHT.XHIGH.allocate(1) == 16
    assert RAM_WEIGHT.LOW.allocate(16) == 8 * 1024**3
    assert RAM_WEIGHT.MEDIUM.allocate(16) == 16 * 1024**3
    assert RAM_WEIGHT.HIGH.allocate(16) == 32 * 1024**3
    assert RAM_WEIGHT.XHIGH.allocate(16) == 64 * 1024**3
    assert SWAP_WEIGHT.LOW.allocate(0) == 4 * 1024**3
    assert SWAP_WEIGHT.MEDIUM.allocate(0) == 8 * 1024**3
    assert SWAP_WEIGHT.HIGH.allocate(0) == 16 * 1024**3
    assert SWAP_WEIGHT.XHIGH.allocate(0) == 32 * 1024**3
    assert CPU_WEIGHT.from_string("xhigh") is CPU_WEIGHT.XHIGH
    assert RAM_WEIGHT.from_string("medium") is RAM_WEIGHT.MEDIUM
    assert SWAP_WEIGHT.from_string("LOW") is SWAP_WEIGHT.LOW
    profile = HardwareProfile(cpu="high", ram="medium", swap="low")  # type: ignore[arg-type]
    assert profile.model_dump(mode="json") == {"cpu": "high", "ram": "medium", "swap": "low"}


def test_mode_hardware_configuration() -> None:
    config = Config.load()
    assert config.resources.profile(IMAGE.VLLM).ram is RAM_WEIGHT.MEDIUM
    assert config.resources.profile(IMAGE.LMCACHE).swap is SWAP_WEIGHT.MEDIUM
    assert config.resources.profile(IMAGE.LLAMACPP).cpu is CPU_WEIGHT.XHIGH
    assert config.resources.profiles.model_dump() == {
        "cpu": {"low": 2, "medium": 4, "high": 8, "xhigh": 16},
        "ram": {"low": 8, "medium": 16, "high": 32, "xhigh": 64},
        "swap": {"low": 4, "medium": 8, "high": 16, "xhigh": 32},
    }


def test_performance_floors_are_validated() -> None:
    with pytest.raises(ValueError, match="low CPU floor must be at least 2"):
        ProfilesConfig(cpu={"low": 1})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="medium RAM floor must be at least 16 GiB"):
        ProfilesConfig(ram={"medium": 15})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="xhigh swap floor must be at least 32 GiB"):
        ProfilesConfig(swap={"xhigh": 31})  # type: ignore[arg-type]


def test_host_cache_lifecycle() -> None:
    with TemporaryDirectory() as temporary_dir:
        cache = HostCache("cache", Path(temporary_dir) / "cache", "/cache")
        assert not cache.exists()
        cache.ensure()
        (cache.host / "nested").mkdir()
        (cache.host / "nested/file").write_text("cached")
        (cache.host / "file").write_text("cached")
        cache.clean()
        assert cache.exists()
        assert list(cache.host.iterdir()) == [cache.host / ".gitkeep"]
        assert cache.volume() == {str(cache.host): {"bind": "/cache", "mode": "rw"}}


def test_host_cache_configuration() -> None:
    root = Path.cwd()
    settings = SimpleNamespace(
        dirs=SimpleNamespace(
            root_cache=root / "cache/root",
            pkg_cache=root / "cache/pkg",
            python_cache=root / "cache/python",
            models=root / "cache/models",
        )
    )
    runtime = object.__new__(DockerRuntime)
    runtime.settings = cast(Any, settings)
    caches = runtime.host_caches()
    assert tuple(type(cache) for cache in caches) == (
        RootCache,
        PackageCache,
        PythonCache,
        ModelCache,
    )
    assert tuple(cache.name for cache in caches) == ("root", "pkg", "python", "models")
    assert tuple(cache.container for cache in caches) == (
        CONTAINERPATH.ROOT_CACHE,
        CONTAINERPATH.PKG_CACHE,
        CONTAINERPATH.PYTHON_CACHE,
        CONTAINERPATH.MODELS,
    )
    ignored = Path(".gitignore").read_text()
    for cache in caches:
        assert (cache.host / ".gitkeep").is_file()
        assert f"!{cache.host.relative_to(root)}/.gitkeep" in ignored


def _clean_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DockerRuntime:
    """Build a DockerRuntime whose clean side effects are recorded, not executed."""
    settings = SimpleNamespace(
        dirs=SimpleNamespace(
            root_cache=tmp_path / "root",
            pkg_cache=tmp_path / "pkg",
            python_cache=tmp_path / "python",
            models=tmp_path / "models",
        )
    )
    runtime = object.__new__(DockerRuntime)
    runtime.settings = cast(Any, settings)
    runtime.client = None
    monkeypatch.setattr(DockerRuntime, "ensure_daemon", lambda self: None)
    monkeypatch.setattr(DockerRuntime, "remove_container", lambda self: None)
    monkeypatch.setattr(DockerRuntime, "remove_networks", lambda self: None)
    monkeypatch.setattr(DockerRuntime, "_remove_effective_configs", lambda self: None)
    monkeypatch.setattr(DockerRuntime, "builders", lambda self: [])
    monkeypatch.setattr(DockerBuilder, "managed_images", classmethod(lambda cls, _client: []))
    return runtime


def test_clean_keeps_caches_without_wipe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """clean without --wipe-cache must leave every host cache in place."""
    cleaned: list[str] = []
    monkeypatch.setattr(HostCache, "clean", lambda self: cleaned.append(self.name))
    runtime = _clean_runtime(tmp_path, monkeypatch)

    assert runtime.clean() == 0
    assert cleaned == []


def test_clean_wipes_selected_caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--wipe-cache removes only the listed caches, empty wipes all."""
    cleaned: list[str] = []
    monkeypatch.setattr(HostCache, "clean", lambda self: cleaned.append(self.name))
    runtime = _clean_runtime(tmp_path, monkeypatch)

    assert runtime.clean(wipe=frozenset({"models"})) == 0
    assert cleaned == ["models"]

    cleaned.clear()
    assert runtime.clean(wipe=frozenset({"root", "pkg", "python", "models"})) == 0
    assert cleaned == ["root", "pkg", "python", "models"]


def test_wipe_cache_parsing() -> None:
    valid = {"root", "pkg", "python", "models"}
    assert _wipe_caches(None, valid) is None
    assert _wipe_caches("", valid) == frozenset(valid)
    assert _wipe_caches("models", valid) == frozenset({"models"})
    assert _wipe_caches("root, pkg", valid) == frozenset({"root", "pkg"})
    with pytest.raises(SystemExit, match="unknown cache name bogus"):
        _wipe_caches("bogus", valid)


def test_qwen_lmcache_configuration() -> None:
    vllm_dockerfile = Path("docker/runtime-vllm.Dockerfile").read_text()
    lmcache_dockerfile = Path("docker/runtime-lmcache.Dockerfile").read_text()
    config = Path("config/config.qwen.yml.example").read_text()
    assert "lmcache==0.5.4" in vllm_dockerfile
    assert "lmcache==0.5.4" in lmcache_dockerfile
    assert "vllm" not in lmcache_dockerfile
    from easyllama.helpers.images import ModeImages

    lmcache = ModeImages.for_mode(MODE.QWEN).dependency(IMAGE.LMCACHE)
    assert lmcache.gpu is True
    assert lmcache.command[:2] == ("/opt/venv/bin/lmcache", "server")
    assert lmcache.command[lmcache.command.index("--instance-id") + 1] == "easyllama-{mode}"
    assert lmcache.command[lmcache.command.index("--chunk-size") + 1] == "{lmcache.chunk_size}"
    assert lmcache.command[lmcache.command.index("--l1-size-gb") + 1] == "{lmcache.l1_size_gb}"
    assert "/app/bin/qwen-lmcache-vllm" not in config
