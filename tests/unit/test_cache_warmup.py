"""Unit tests for cache-first Hugging Face model resolution."""

from pathlib import Path

from easyllama.helpers.hf import HuggingFace
import pytest

pytestmark = pytest.mark.unit


def _patch_download(
    monkeypatch, *, local: str | None, network: str | None, calls: list[dict]
) -> None:
    """Route hf_hub_download through a local-first fake.

    Args:
        monkeypatch: The monkeypatch.
        local: Cached path returned for local_files_only calls, or None to miss.
        network: Path returned for network calls, or None to assert unused.
        calls: Recorded call keyword arguments."""
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    def fake_download(**kwargs: object) -> str:
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            if local is None:
                raise LocalEntryNotFoundError("no local entry")
            return local
        if network is None:
            raise AssertionError("network download was attempted")
        return network

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)


def _patch_snapshot(
    monkeypatch, *, local: Path | None, network: Path | None, calls: list[dict]
) -> None:
    """Route snapshot_download through a local-first fake.

    Args:
        monkeypatch: The monkeypatch.
        local: Cached snapshot returned for local_files_only calls, or None to miss.
        network: Snapshot returned for network calls, or None to assert unused.
        calls: Recorded call keyword arguments."""
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    def fake_snapshot(**kwargs: object) -> str:
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            if local is None:
                raise LocalEntryNotFoundError("no local entry")
            return str(local)
        if network is None:
            raise AssertionError("network snapshot was attempted")
        return str(network)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)


def test_get_returns_cached_file_without_network(monkeypatch, tmp_path: Path, caplog) -> None:
    import logging

    cached = tmp_path / "RVN-Q4_K_M.gguf"
    cached.write_text("model")
    calls: list[dict] = []
    _patch_download(monkeypatch, local=str(cached), network=None, calls=calls)

    with caplog.at_level(logging.INFO, logger="easyllama.helpers.hf"):
        result = HuggingFace("model", "owner/repo").get("RVN-Q4_K_M.gguf", cache_dir=tmp_path)

    assert result == cached
    assert len(calls) == 1 and calls[0]["local_files_only"] is True
    assert "using cached model file RVN-Q4_K_M.gguf" in caplog.text


def test_get_downloads_only_when_local_cache_misses(monkeypatch, tmp_path: Path) -> None:
    downloaded = tmp_path / "fetched.gguf"
    downloaded.write_text("model")
    calls: list[dict] = []
    _patch_download(monkeypatch, local=None, network=str(downloaded), calls=calls)

    result = HuggingFace("model", "owner/repo").get("fetched.gguf", cache_dir=tmp_path)

    assert result == downloaded
    assert len(calls) == 2
    assert calls[0]["local_files_only"] is True and calls[1].get("local_files_only") is None


def test_snapshot_uses_cached_weights_without_network(monkeypatch, tmp_path: Path, caplog) -> None:
    import logging

    cached = tmp_path / "snapshot"
    cached.mkdir()
    (cached / "weights.safetensors").write_text("weights")
    calls: list[dict] = []
    _patch_snapshot(monkeypatch, local=cached, network=None, calls=calls)

    with caplog.at_level(logging.INFO, logger="easyllama.helpers.hf"):
        result = HuggingFace("model", "owner/repo").snapshot()

    assert result == cached
    assert len(calls) == 1 and calls[0]["local_files_only"] is True
    assert "using cached snapshot for model" in caplog.text


def test_snapshot_downloads_when_cache_is_cold(monkeypatch, tmp_path: Path) -> None:
    downloaded = tmp_path / "fresh"
    downloaded.mkdir()
    calls: list[dict] = []
    _patch_snapshot(monkeypatch, local=None, network=downloaded, calls=calls)

    result = HuggingFace("model", "owner/repo").snapshot()

    assert result == downloaded
    assert len(calls) == 2
    assert calls[0]["local_files_only"] is True and calls[1].get("local_files_only") is None


def test_snapshot_retries_network_when_local_lacks_weights(monkeypatch, tmp_path: Path) -> None:
    incomplete = tmp_path / "partial"
    incomplete.mkdir()
    (incomplete / "README.md").write_text("partial")
    downloaded = tmp_path / "complete"
    downloaded.mkdir()
    calls: list[dict] = []
    _patch_snapshot(monkeypatch, local=incomplete, network=downloaded, calls=calls)

    result = HuggingFace("model", "owner/repo").snapshot()

    assert result == downloaded
    assert len(calls) == 2 and calls[1].get("local_files_only") is None


def test_snapshot_targets_configured_cache_dir(monkeypatch, tmp_path: Path) -> None:
    """Host-side prefetches must be able to pin snapshots to the shared model cache."""
    hub = tmp_path / "hub"
    calls: list[dict] = []
    _patch_snapshot(monkeypatch, local=None, network=hub / "snapshot", calls=calls)

    result = HuggingFace("model", "owner/repo").snapshot(
        cache_dir=hub, warmup="Warming model 1/1: glm53-chat"
    )

    assert result == hub / "snapshot"
    assert len(calls) == 2
    assert all(call["cache_dir"] == hub for call in calls)
    assert all(call.get("tqdm_class") is not None for call in calls)


def test_snapshot_local_hit_ignores_cache_dir_network(monkeypatch, tmp_path: Path) -> None:
    """A cached snapshot with weights is served from disk without network access."""
    hub = tmp_path / "hub"
    cached = hub / "snapshot"
    cached.mkdir(parents=True)
    (cached / "weights.safetensors").write_text("weights")
    calls: list[dict] = []
    _patch_snapshot(monkeypatch, local=cached, network=None, calls=calls)

    result = HuggingFace("model", "owner/repo").snapshot(cache_dir=hub)

    assert result == cached
    assert len(calls) == 1 and calls[0]["local_files_only"] is True


def test_snapshot_weight_detection(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    gguf = tmp_path / "gguf"
    gguf.mkdir()
    (gguf / "model.gguf").write_text("model")
    tensors = tmp_path / "tensors"
    tensors.mkdir()
    (tensors / "model.safetensors").write_text("weights")

    assert not HuggingFace._snapshot_has_weights(empty)
    assert HuggingFace._snapshot_has_weights(gguf)
    assert HuggingFace._snapshot_has_weights(tensors)
