"""Unit tests for DockerRuntime start idempotence against stale containers."""

from pathlib import Path
import types
from typing import Any

from easyllama.config import CONTAINERPATH, Config, CredentialsConfig
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.docker]


class FakeContainer:
    """Represent one Docker container for runtime lifecycle tests.

    Attributes:
        name: The name.
        status: The status.
        labels: The labels.
        attrs: The attrs.
        stopped: The stopped.
        removed: The removed."""

    def __init__(
        self,
        name: str,
        status: str = "exited",
        labels: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.status = status
        self.labels = dict(labels or {})
        self.attrs: dict[str, Any] = {
            "State": {"Status": "running", "Health": {"Status": "healthy"}}
        }
        self.stopped = False
        self.removed = False

    def reload(self) -> None:
        """No-op container reload."""

    def stop(self, timeout: int | None = None) -> None:
        """Record the stop and mark the container exited.

        Args:
            timeout: The timeout."""
        self.stopped = True
        self.status = "exited"

    def remove(self, force: bool = False) -> None:
        """Record the removal.

        Args:
            force: The force."""
        self.removed = True


class FakeContainers:
    """Emulate docker-py container collection semantics used by the runtime.

    Attributes:
        items: The items.
        run_calls: The run calls."""

    def __init__(self, items: list[FakeContainer]) -> None:
        self.items = list(items)
        self.run_calls: list[tuple[str, str | None]] = []
        self.run_records: list[tuple[str, str | None, dict[str, Any]]] = []

    def list(self, all: bool = False, filters: dict[str, Any] | None = None) -> list[FakeContainer]:
        """Filter the fake container collection.

        Args:
            all: The all.
            filters: The filters.

        Returns:
            list[FakeContainer]: The matching containers."""
        if filters:
            wanted = dict(item.split("=", 1) for item in filters.get("label", []))
            # The parameter name all shadows the builtin inside this comprehension,
            # so express the match with any() instead.
            return [
                c
                for c in self.items
                if not any(c.labels.get(key) != value for key, value in wanted.items())
            ]
        if all:
            return list(self.items)
        return [c for c in self.items if c.status == "running"]

    def run(self, image: str, **kwargs: Any) -> FakeContainer:
        """Record a container run and append the running container.

        Args:
            image: The image.
            **kwargs: The kwargs.

        Returns:
            FakeContainer: The started container."""
        name = str(kwargs.get("name") or "")
        container = FakeContainer(name, status="running")
        self.items.append(container)
        self.run_calls.append((image, name))
        self.run_records.append((image, name, kwargs))
        return container


class FakeImages:
    """Report every requested image as present."""

    def get(self, name: str) -> Any:
        """Return a minimal image record.

        Args:
            name: The name.

        Returns:
            Any: The image record."""
        return types.SimpleNamespace(name=name, tags=[name])


class FakeNetwork:
    """Emulate one docker network."""

    def __init__(self, name: str) -> None:
        self.name = name

    def remove(self) -> None:
        """Remove the network."""
        return


class FakeNetworks:
    """Emulate the docker networks collection."""

    def __init__(self) -> None:
        self.items: list[FakeNetwork] = []

    def list(self, **_kwargs: Any) -> list[FakeNetwork]:
        """List networks."""
        return list(self.items)

    def create(self, name: str, **_kwargs: Any) -> FakeNetwork:
        """Create and record a network."""
        network = FakeNetwork(name)
        self.items.append(network)
        return network


class FakeClient:
    """Emulate the docker-py client surface used by DockerRuntime.

    Attributes:
        containers: The containers.
        images: The images.
        networks: The networks."""

    def __init__(self, items: list[FakeContainer]) -> None:
        self.containers = FakeContainers(items)
        self.images = FakeImages()
        self.networks = FakeNetworks()
        self.api: Any = self

    def ping(self) -> bool:
        """Report the daemon as reachable."""
        return True

    def info(self) -> dict[str, Any]:
        """Return daemon metadata with an nvidia runtime."""
        return {"Runtimes": {"nvidia": {}}, "NCPU": 8, "MemTotal": 64 * 1024**3}


def _runtime(monkeypatch: pytest.MonkeyPatch, items: list[FakeContainer]) -> tuple[Any, FakeClient]:
    from easyllama.helpers import docker as docker_module

    client = FakeClient(items)
    monkeypatch.setattr(docker_module, "from_env", lambda: client)
    runtime = docker_module.DockerRuntime(Config.load())
    # A deterministic configured credential: the host environment must not be able
    # to blank it out through contract ${HF_TOKEN} expansion.
    runtime.settings = runtime.settings.model_copy(
        update={"credentials": CredentialsConfig(hf_token="hf_placeholder")}
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    return runtime, client


def test_start_sweeps_stale_dependency_containers(monkeypatch: pytest.MonkeyPatch) -> None:
    stale = FakeContainer(
        "easyllama-qwen-vllm-qwen3-chat",
        status="exited",
        labels={"easyllama.managed": "true", "easyllama.mode": "qwen"},
    )
    runtime, client = _runtime(monkeypatch, [stale])

    assert runtime.run_container() == 0
    assert stale.removed
    assert any(name == "easyllama-server-swap" for _image, name in client.containers.run_calls)


def test_start_is_unchanged_without_containers(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, client = _runtime(monkeypatch, [])

    assert runtime.run_container() == 0
    assert not any(container.removed for container in client.containers.items)
    assert any(name == "easyllama-server-swap" for _image, name in client.containers.run_calls)


def test_start_is_noop_when_orchestrator_is_running(monkeypatch: pytest.MonkeyPatch) -> None:
    running = FakeContainer(
        "easyllama-server-swap",
        status="running",
        labels={"easyllama.managed": "true", "easyllama.mode": "llamacpp"},
    )
    runtime, client = _runtime(monkeypatch, [running])

    assert runtime.run_container() == 0
    assert client.containers.run_calls == []
    assert not running.removed and not running.stopped


def test_fresh_start_mounts_persisted_hf_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh image start must bind the host model cache into every container.

    This is the fresh-image-start regression: without the bind, in-container
    Hugging Face downloads land in an ephemeral image layer and re-download on
    every start. The bind must coexist with the nested root-cache mount, so the
    parent mount is declared before the nested model-cache mount."""
    runtime, client = _runtime(monkeypatch, [])

    assert runtime.run_container() == 0
    assert client.containers.run_records
    for _image, _name, kwargs in client.containers.run_records:
        volumes = kwargs["volumes"]
        models_host = str(runtime.settings.dirs.models)
        assert models_host in volumes
        assert volumes[models_host] == {"bind": CONTAINERPATH.MODELS, "mode": "rw"}
        # Nested bind mounts only resolve when the parent is declared first.
        mounts = list(volumes)
        parent = str(runtime.settings.dirs.root_cache)
        if parent in mounts:
            assert mounts.index(parent) < mounts.index(models_host)


def test_fresh_start_forwards_configured_hf_token_and_cache_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured credentials must reach containers even without a host HF_TOKEN.

    The contract environment carries HF_TOKEN=${HF_TOKEN}; when the host does not
    export HF_TOKEN, the empty expansion must not blank out the configured token,
    and every container must pin its HF cache to the mounted host cache."""
    runtime, client = _runtime(monkeypatch, [])

    assert runtime.run_container() == 0
    assert client.containers.run_records
    for _image, _name, kwargs in client.containers.run_records:
        environment = kwargs["environment"]
        assert environment["HF_TOKEN"] == "hf_placeholder"
        assert environment["HF_HUB_CACHE"] == str(CONTAINERPATH.MODELS)
        assert environment["HF_HOME"] == str(Path(CONTAINERPATH.MODELS).parent)
