"""Unit tests for warmup-time host model prefetching.

Covers the fresh-image-start cache case: every model spec must resolve into the
shared host model cache before the container warmup deadline starts, including
bare Hugging Face repo specs (for example GLM-5.3-Flash) that resolve to a
filtered snapshot rather than a single GGUF file.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast

from easyllama import runtime as runtime_module
from easyllama.helpers.hf import HuggingFace
import pytest
import yaml

pytestmark = pytest.mark.unit


class FakeHf:
    """Record prefetch calls instead of touching the network.

    Attributes:
        parse_spec: The parse spec.
        gets: The gets.
        snapshots: The snapshots."""

    parse_spec = staticmethod(HuggingFace.parse_spec)
    gets: ClassVar[list[tuple[str, str | None, dict[str, Any]]]] = []
    snapshots: ClassVar[list[tuple[str, str | None, dict[str, Any]]]] = []

    def __init__(self, name: str, repo: str | None = None) -> None:
        self.name = name
        self.repo = repo

    def file(self, choice: str | None, **kwargs: Any) -> str:
        """Return the selector unchanged for test specifications."""
        return choice or ""

    def get(self, file: str, **kwargs: Any) -> str:
        """Record a single-file prefetch."""
        FakeHf.gets.append((self.name, file, kwargs))
        return file

    def snapshot(self, **kwargs: Any) -> Path:
        """Record a filtered snapshot prefetch."""
        FakeHf.snapshots.append((self.name, self.repo, kwargs))
        return Path(kwargs.get("cache_dir") or Path(".")) / "snapshot"


def _settings(tmp_path: Path) -> Any:
    """Build a runtime settings stand-in around one temporary config file."""
    models_dir = tmp_path / "models"
    config_file = tmp_path / "config.yml"
    config_file.write_text(
        yaml.safe_dump(
            {
                "macros": {
                    "glm_repo": "RedHatAI/GLM-5.3-Flash-NVFP4",
                    "embedding_model": (
                        "/root/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B-GGUF"
                        "/snapshots/370f27d7550e0def9b39c1f16d3fbaa13aa67728"
                        "/Qwen3-Embedding-0.6B-f16.gguf"
                    ),
                },
                "models": {
                    "glm53-chat": {
                        "cmd": "easyllama server glm5.3-flash -hf ${glm_repo} --port 9000"
                    },
                    "qwen3-chat": {
                        "cmd": "easyllama server qwen -hf owner/repo:RVN-Q4_K_M.gguf --port 9001"
                    },
                    "qwen3-embeddings": {
                        "cmd": "easyllama server qwen --model ${embedding_model} --port 9003"
                    },
                    "local-model": {
                        "cmd": "easyllama server qwen --model /models/cached.gguf --port 9002"
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    settings = SimpleNamespace(
        resolve_ls_config=lambda: config_file,
        dirs=SimpleNamespace(models=models_dir),
    )
    return settings, models_dir


def test_prefetch_targets_shared_model_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both spec shapes must download into the mounted host cache, not a default."""
    settings, models_dir = _settings(tmp_path)
    FakeHf.gets.clear()
    FakeHf.snapshots.clear()
    monkeypatch.setattr(runtime_module, "HuggingFace", FakeHf)

    runtime_module._prefetch_models(
        cast(Any, settings), ["glm53-chat", "qwen3-chat", "local-model"]
    )

    assert len(FakeHf.snapshots) == 1
    name, repo, kwargs = FakeHf.snapshots[0]
    assert (name, repo) == ("glm53-chat", "RedHatAI/GLM-5.3-Flash-NVFP4")
    assert kwargs["cache_dir"] == models_dir
    assert "glm53-chat" in kwargs["warmup"]
    assert len(FakeHf.gets) == 1
    name, file, kwargs = FakeHf.gets[0]
    assert (name, file) == ("qwen3-chat", "RVN-Q4_K_M.gguf")
    assert kwargs["cache_dir"] == models_dir


def test_prefetch_skips_unresolvable_specs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Models without an -hf spec (local paths) must not trigger downloads."""
    settings, _models_dir = _settings(tmp_path)
    FakeHf.gets.clear()
    FakeHf.snapshots.clear()
    monkeypatch.setattr(runtime_module, "HuggingFace", FakeHf)

    runtime_module._prefetch_models(cast(Any, settings), ["local-model"])

    assert FakeHf.gets == []
    assert FakeHf.snapshots == []


def test_prefetch_downloads_hub_model_refs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--model hub-cache refs (embeddings/reranker) must download into the shared cache.

    The file is pinned to the exact snapshot commit the container command expects,
    so a wiped or fresh model cache resolves identically to a warm one."""
    settings, models_dir = _settings(tmp_path)
    FakeHf.gets.clear()
    FakeHf.snapshots.clear()
    monkeypatch.setattr(runtime_module, "HuggingFace", FakeHf)

    runtime_module._prefetch_models(cast(Any, settings), ["qwen3-embeddings"])

    assert FakeHf.snapshots == []
    assert len(FakeHf.gets) == 1
    name, file, kwargs = FakeHf.gets[0]
    assert name == "qwen3-embeddings"
    assert file == "Qwen3-Embedding-0.6B-f16.gguf"
    assert kwargs["cache_dir"] == models_dir
    assert kwargs["revision"] == "370f27d7550e0def9b39c1f16d3fbaa13aa67728"
    assert "qwen3-embeddings" in kwargs["warmup"]


def test_hf_token_prefers_env_then_configured_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Token resolution follows host env, then configured credentials, then unset."""
    from easyllama.config import Config

    hf = HuggingFace("probe", "owner/repo")
    monkeypatch.setenv("HF_TOKEN", "env-token")
    assert hf.token == "env-token"
    monkeypatch.delenv("HF_TOKEN")
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    loaded = SimpleNamespace(load_auth=lambda: SimpleNamespace(hf_token="cfg-token"))

    def fake_load(cls: Any, *args: Any, **kwargs: Any) -> Any:
        return loaded

    monkeypatch.setattr(Config, "load", classmethod(fake_load))
    assert hf.token == "cfg-token"

    def broken_load(cls: Any, *args: Any, **kwargs: Any) -> Any:
        raise SystemExit("no config")

    monkeypatch.setattr(Config, "load", classmethod(broken_load))
    assert hf.token is None


def test_hub_cache_ref_decoding() -> None:
    """models--<owner>--<repo>/snapshots/<commit>/<file> decodes to (repo, commit, file)."""
    ref = runtime_module._hub_cache_ref(
        "/root/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B-GGUF"
        "/snapshots/370f27d7550e0def9b39c1f16d3fbaa13aa67728/Qwen3-Embedding-0.6B-f16.gguf"
    )
    assert ref == (
        "Qwen/Qwen3-Embedding-0.6B-GGUF",
        "370f27d7550e0def9b39c1f16d3fbaa13aa67728",
        "Qwen3-Embedding-0.6B-f16.gguf",
    )
    assert runtime_module._hub_cache_ref("/models/cached.gguf") is None
    assert runtime_module._hub_cache_ref("/models--owner--repo/blob/abc.gguf") is None
    assert runtime_module._hub_cache_ref("/models--/snapshots/abc/noowner.gguf") is None
