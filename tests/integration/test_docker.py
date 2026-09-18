"""Live Docker integration tests for the managed container stack.

Run with EASYLLAMA_INTEGRATION=1 against a started easyllama stack. The
fresh-image-start tests here guard the two field regressions: the configured
Hugging Face token reaching dependency containers, and the persistent model
cache bind mount surviving image rebuilds.
"""

import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.docker]

SKIP = pytest.mark.skipif(
    os.environ.get("EASYLLAMA_INTEGRATION") != "1",
    reason="set EASYLLAMA_INTEGRATION=1 to run Docker integration tests",
)


@SKIP
def test_docker_daemon() -> None:
    from docker import from_env

    assert from_env().ping()


@SKIP
def test_fresh_image_start_persists_hf_cache_and_token() -> None:
    """Running containers must expose the mounted cache and a live HF token.

    After a fresh image start, every managed container in the selected mode must
    bind the host model cache at the container HF hub path and carry a
    non-empty HF_TOKEN when credentials are configured. Either failure means
    in-container model downloads are unauthenticated and/or ephemeral.
    """
    from docker import from_env
    from easyllama.config import CONTAINERPATH, Config

    client = from_env()
    settings = Config.load()
    mode = settings.runtime.mode
    containers = client.containers.list(
        filters={
            "label": ["easyllama.managed=true", f"easyllama.mode={mode}"],
            "status": "running",
        }
    )
    assert containers, f"no running managed containers for mode {mode}; start it first"

    models_host = str(settings.dirs.models)
    for container in containers:
        container_id = container.id or str(container.short_id)
        attrs = client.api.inspect_container(container_id)
        binds = {
            mount["Destination"]: mount["Source"]
            for mount in attrs.get("Mounts", [])
            if mount.get("Type") == "bind"
        }
        env = dict(item.split("=", 1) for item in attrs["Config"].get("Env", []) if "=" in item)
        assert binds.get(CONTAINERPATH.MODELS) == models_host, (
            f"{container.name}: model cache not mounted at {CONTAINERPATH.MODELS}; "
            f"binds={sorted(binds)}"
        )
        assert env.get("HF_HUB_CACHE") == str(CONTAINERPATH.MODELS), (
            f"{container.name}: HF_HUB_CACHE not pinned to the mounted cache"
        )
        assert env.get("HF_HOME") == str(Path(CONTAINERPATH.MODELS).parent), (
            f"{container.name}: HF_HOME not pinned to the mounted cache"
        )
        if settings.credentials.hf_token:
            assert env.get("HF_TOKEN"), (
                f"{container.name}: HF_TOKEN is empty; configured credentials "
                "did not reach the container"
            )
