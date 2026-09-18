"""Unit tests for attaching to in-flight Hugging Face downloads.

Covers the warmup case where another process (for example a container
backend) is already downloading a model: the host-side warmup must latch
onto the existing download and mirror its progress instead of reporting
only "starting" or racing a second download.
"""

import logging
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from typing import Any, ClassVar

from easyllama.helpers import hf as hf_module
from easyllama.helpers.hf import HuggingFace
import pytest

pytestmark = pytest.mark.unit

ETAG = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
ETAG2 = "b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3"
REPO = "owner/repo"
FILE = "model.gguf"
REPO_FOLDER = f"models--{REPO.replace('/', '--')}"
HF_LOGGER = "easyllama.helpers.hf"


class FakeReporter:
    """Stand-in for HfProgress that records update() deltas."""

    instances: ClassVar[list] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.total = int(kwargs.get("total") or 0)
        self.n = int(kwargs.get("initial") or 0)
        self.desc = str(kwargs.get("desc") or "")
        self.updates: list[int] = []
        FakeReporter.instances.append(self)

    def update(self, n: int = 1) -> None:
        self.n += n
        self.updates.append(n)


class ExternalDownload:
    """Simulates another process downloading into the shared cache.

    Holds the huggingface_hub lock for the blob while growing the
    .incomplete partial, like a real (container-side) downloader."""

    def __init__(
        self,
        *,
        lock_path: Path,
        partial: Path,
        sizes: list[int],
        done: dict[str, bool],
        start: threading.Event | None = None,
        pace: float = 0.03,
        complete: bool = True,
    ) -> None:
        self.lock_path = lock_path
        self.partial = partial
        self.sizes = sizes
        self.done = done
        self.start_event = start
        self.pace = pace
        self.complete = complete
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        from filelock import FileLock

        lock = FileLock(str(self.lock_path))
        lock.acquire()
        self.ready.set()
        if self.start_event is not None:
            self.start_event.wait(timeout=10)
        try:
            for size in self.sizes:
                self.partial.write_bytes(b"x" * size)
                time.sleep(self.pace)
            if self.complete:
                self.done["value"] = True
        finally:
            lock.release()

    def start(self) -> None:
        """Start the simulated external downloader thread."""
        self.thread.start()

    def wait_ready(self) -> None:
        assert self.ready.wait(timeout=10), "external download never acquired the lock"


def _hub(tmp_path: Path) -> Path:
    hub = tmp_path / "hub"
    (hub / REPO_FOLDER / "blobs").mkdir(parents=True)
    return hub


def _patch_metadata(
    monkeypatch, *, etag: str = ETAG, size: int = 10240, counts: list[int] | None = None
) -> None:
    import huggingface_hub.file_download as file_download

    def fake_metadata(*args: Any, **kwargs: Any) -> SimpleNamespace:
        if counts is not None:
            counts.append(1)
        return SimpleNamespace(etag=etag, size=size)

    monkeypatch.setattr(file_download, "get_hf_file_metadata", fake_metadata)


def _patch_hf_download(
    monkeypatch, *, local: dict[str, bool], cached: Path
) -> list[dict[str, Any]]:
    """Patch hf_hub_download: local hit only once the external download completes."""
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    calls: list[dict[str, Any]] = []

    def fake_download(**kwargs: Any) -> str:
        calls.append(dict(kwargs))
        if kwargs.get("local_files_only"):
            if not local["value"]:
                raise LocalEntryNotFoundError("no local entry")
            return str(cached)
        raise AssertionError("network download was attempted")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    return calls


def _patch_hf_download_network(
    monkeypatch, *, network: Path, local: dict[str, bool] | None = None
) -> list[dict[str, Any]]:
    """Patch hf_hub_download: local miss until `local` flips; network returns path."""
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    if local is None:
        local = {"value": False}
    calls: list[dict[str, Any]] = []

    def fake_download(**kwargs: Any) -> str:
        calls.append(dict(kwargs))
        if kwargs.get("local_files_only"):
            if not local["value"]:
                raise LocalEntryNotFoundError("no local entry")
            return str(network)
        return str(network)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    return calls


class FakeApi:
    """HfApi stand-in reporting the repository tree for attribution."""

    counts: ClassVar[list[int]] = []
    entries: ClassVar[list[SimpleNamespace]] = []

    def __init__(self, **_: Any) -> None:
        pass

    def list_repo_tree(self, **kwargs: Any) -> list[SimpleNamespace]:
        FakeApi.counts.append(1)
        return FakeApi.entries


def _run_in_worker(fn: Any, timeout: float = 20.0) -> Any:
    """Run fn() on a worker thread and return its result (or raise)."""
    box: dict[str, Any] = {}

    def caller() -> None:
        try:
            box["result"] = fn()
        except BaseException as exc:
            box["error"] = exc

    worker = threading.Thread(target=caller, daemon=True)
    worker.start()
    worker.join(timeout)
    assert not worker.is_alive(), f"worker did not finish within {timeout}s"
    if "error" in box:
        raise box["error"]
    return box.get("result")


def _wait_for_attach(timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not FakeReporter.instances:
        if time.monotonic() > deadline:
            pytest.fail("attach never started")
        time.sleep(0.01)


@pytest.fixture()
def fast_attach(monkeypatch) -> None:
    monkeypatch.setattr(hf_module, "ATTACH_POLL_INTERVAL", 0.02)
    monkeypatch.setattr(hf_module, "ATTACH_QUIET_CYCLES", 4)


def test_get_attaches_to_external_download_and_returns_cache(
    tmp_path: Path, monkeypatch, fast_attach, caplog
) -> None:
    hub = _hub(tmp_path)
    partial = hub / REPO_FOLDER / "blobs" / f"{ETAG}.incomplete"
    partial.write_bytes(b"x" * 1024)
    lock_path = hub / ".locks" / REPO_FOLDER / f"{ETAG}.lock"
    cached = hub / REPO_FOLDER / "blobs" / ETAG
    cached.write_bytes(b"x" * 10240)
    done = {"value": False}
    start = threading.Event()
    FakeReporter.instances = []
    monkeypatch.setattr(hf_module, "HfProgress", FakeReporter)
    _patch_metadata(monkeypatch)
    _patch_hf_download(monkeypatch, local=done, cached=cached)
    external = ExternalDownload(
        lock_path=lock_path, partial=partial, sizes=[2048, 3072], done=done, start=start
    )
    external.start()
    external.wait_ready()

    with caplog.at_level(logging.INFO, logger=HF_LOGGER):
        box: dict[str, Any] = {}

        def caller() -> None:
            box["result"] = HuggingFace("test", REPO).get(
                FILE, cache_dir=hub, warmup="Warming 1/1: test"
            )

        worker = threading.Thread(target=caller, daemon=True)
        worker.start()
        _wait_for_attach()
        start.set()
        worker.join(timeout=20)
        assert not worker.is_alive(), "warmup get() did not return after external completion"
        result = box.get("result")

    assert result == cached
    assert any("attaching to in-flight download" in m for m in caplog.messages)
    reporter = FakeReporter.instances[0]
    assert reporter.n >= 3072, "mirrored progress should track the external download"
    external.thread.join(timeout=10)


def test_get_downloads_itself_when_nothing_in_flight(
    tmp_path: Path, monkeypatch, fast_attach
) -> None:
    hub = _hub(tmp_path)
    metadata_counts: list[int] = []
    _patch_metadata(monkeypatch, counts=metadata_counts)
    network = hub / "network-out.gguf"
    calls = _patch_hf_download_network(monkeypatch, network=network)

    result = HuggingFace("test", REPO).get(FILE, cache_dir=hub)

    assert result == network
    assert metadata_counts == []  # fast path: no partials, no metadata call
    assert any(not c.get("local_files_only") for c in calls)


def test_get_falls_back_to_own_download_when_external_dies(
    tmp_path: Path, monkeypatch, fast_attach, caplog
) -> None:
    hub = _hub(tmp_path)
    partial = hub / REPO_FOLDER / "blobs" / f"{ETAG}.incomplete"
    partial.write_bytes(b"x" * 512)
    lock_path = hub / ".locks" / REPO_FOLDER / f"{ETAG}.lock"
    done = {"value": False}
    FakeReporter.instances = []
    monkeypatch.setattr(hf_module, "HfProgress", FakeReporter)
    _patch_metadata(monkeypatch)
    network = hub / "own-out.gguf"
    _patch_hf_download_network(monkeypatch, network=network, local=done)
    external = ExternalDownload(
        lock_path=lock_path,
        partial=partial,
        sizes=[1024],
        done=done,
        complete=False,  # external download dies without completing
        pace=0.05,
    )
    external.start()
    external.wait_ready()

    with caplog.at_level(logging.INFO, logger=HF_LOGGER):
        result = _run_in_worker(lambda: HuggingFace("test", REPO).get(FILE, cache_dir=hub))

    assert result == network  # own resumable download after the quiet window
    assert any("attaching to in-flight download" in m for m in caplog.messages)
    external.thread.join(timeout=10)


def test_snapshot_attaches_to_external_download(
    tmp_path: Path, monkeypatch, fast_attach, caplog
) -> None:
    hub = _hub(tmp_path)
    partial = hub / REPO_FOLDER / "blobs" / f"{ETAG}.incomplete"
    partial.write_bytes(b"x" * 64)
    lock_path = hub / ".locks" / REPO_FOLDER / f"{ETAG}.lock"
    snapshot = hub / REPO_FOLDER / "snapshots" / "deadbeef"
    snapshot.mkdir(parents=True)
    (snapshot / "model-00001-of-00002.gguf").write_bytes(b"w")
    done = {"value": False}
    FakeReporter.instances = []
    monkeypatch.setattr(hf_module, "HfProgress", FakeReporter)

    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    FakeApi.counts = []
    FakeApi.entries = [
        SimpleNamespace(
            path="model-00001-of-00002.gguf",
            size=100,
            lfs=SimpleNamespace(sha256=ETAG),
            blob_id=None,
        ),
        SimpleNamespace(
            path="model-00002-of-00002.gguf",
            size=200,
            lfs=SimpleNamespace(sha256=ETAG2),
            blob_id=None,
        ),
    ]
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)

    def fake_snapshot(**kwargs: Any) -> str:
        if kwargs.get("local_files_only"):
            if not done["value"]:
                raise LocalEntryNotFoundError("no local snapshot")
            return str(snapshot)
        raise AssertionError("network snapshot was attempted")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    external = ExternalDownload(lock_path=lock_path, partial=partial, sizes=[128, 192], done=done)
    external.start()
    external.wait_ready()

    with caplog.at_level(logging.INFO, logger=HF_LOGGER):
        result = _run_in_worker(
            lambda: HuggingFace("test", REPO).snapshot(cache_dir=hub, warmup="Warming 1/1: test")
        )

    assert result == snapshot
    assert FakeApi.counts, "snapshot attach must resolve the file list for attribution"
    assert any("attaching to in-flight download" in m for m in caplog.messages)
    assert any("snapshot (2 files)" in m for m in caplog.messages)
    external.thread.join(timeout=10)


def test_snapshot_downloads_itself_when_nothing_in_flight(
    tmp_path: Path, monkeypatch, fast_attach
) -> None:
    hub = _hub(tmp_path)
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    FakeApi.counts = []
    FakeApi.entries = []
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    network = hub / "snapshot-out"

    def fake_snapshot(**kwargs: Any) -> str:
        if kwargs.get("local_files_only"):
            raise LocalEntryNotFoundError("no local snapshot")
        return str(network)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)

    result = HuggingFace("test", REPO).snapshot(cache_dir=hub)

    assert result == network
    assert FakeApi.counts == []  # fast path: no partials, no tree call


def test_lock_held_detection(tmp_path: Path) -> None:
    from filelock import FileLock

    lock_path = tmp_path / ".locks" / "x" / f"{ETAG}.lock"
    holder = FileLock(str(lock_path))
    assert HuggingFace._lock_held(lock_path) is False
    holder.acquire()
    try:
        assert HuggingFace._lock_held(lock_path) is True
    finally:
        holder.release()
    assert HuggingFace._lock_held(lock_path) is False


def test_incomplete_blobs_finds_repo_local_and_shared(tmp_path: Path) -> None:
    hub = _hub(tmp_path)
    repo_local = hub / REPO_FOLDER / "blobs" / f"{ETAG}.incomplete"
    shared_flat = hub / "blobs" / f"{ETAG2}.incomplete"
    shared_fanout = hub / "blobs" / "c3" / f"{ETAG2}.incomplete"
    for path in (repo_local, shared_flat, shared_fanout):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    found = HuggingFace._incomplete_blobs(hub)
    assert repo_local in found
    assert shared_flat in found
    assert shared_fanout in found
