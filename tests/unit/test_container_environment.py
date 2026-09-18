"""Unit tests for dependency container environment resolution.

Covers the HF_TOKEN regression: the contract environment carries
``HF_TOKEN=${HF_TOKEN}``, and an empty host-side expansion must not blank out a
token resolved from configured credentials.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from easyllama.config import CONTAINERPATH, IMAGE, MODE
from easyllama.helpers.docker import DockerRuntime
from easyllama.helpers.orchestrator import ContainerContract
import pytest

pytestmark = pytest.mark.unit

CONTRACT_ENV = ("HF_TOKEN=${HF_TOKEN}",)


def _runtime() -> DockerRuntime:
    """Build a DockerRuntime bound to minimal in-memory settings."""
    settings = SimpleNamespace(
        runtime=SimpleNamespace(mode=MODE.LLAMACPP),
        locale=SimpleNamespace(timezone="UTC", lang="C.UTF-8", lc_all="C.UTF-8"),
        mmproj_arg=lambda auth: "",
    )
    runtime = object.__new__(DockerRuntime)
    runtime.settings = cast(Any, settings)
    return runtime


def _contract(environment: tuple[str, ...] = CONTRACT_ENV) -> ContainerContract:
    return ContainerContract(
        name="easyllama-test-llamacpp-model",
        image=IMAGE.LLAMACPP,
        command=(),
        environment=environment,
    )


def _auth(hf_token: str | None) -> Any:
    return SimpleNamespace(hf_token=hf_token, api_key=None)


def test_configured_token_survives_empty_host_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: ${HF_TOKEN} expanding to '' must keep the configured token."""
    monkeypatch.delenv("HF_TOKEN", raising=False)

    environment = _runtime()._container_environment(_auth("hf_config"), _contract())

    assert environment["HF_TOKEN"] == "hf_config"


def test_host_exported_token_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_host")

    environment = _runtime()._container_environment(_auth("hf_config"), _contract())

    assert environment["HF_TOKEN"] == "hf_host"


def test_missing_credential_and_host_export_expand_to_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)

    environment = _runtime()._container_environment(_auth(None), _contract())

    assert environment.get("HF_TOKEN") in (None, "")


def test_hf_cache_is_pinned_to_mounted_host_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every HF client in the container must target the persistent bind mount."""
    monkeypatch.delenv("HF_TOKEN", raising=False)

    environment = _runtime()._container_environment(_auth("hf_config"), _contract())

    assert environment["HF_HUB_CACHE"] == str(CONTAINERPATH.MODELS)
    assert environment["HF_HOME"] == str(Path(CONTAINERPATH.MODELS).parent)


def test_literal_contract_environment_is_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    contract = _contract(("HF_TOKEN=${HF_TOKEN}", "CUSTOM=value"))

    environment = _runtime()._container_environment(_auth("hf_config"), contract)

    assert environment["CUSTOM"] == "value"
    assert environment["HF_TOKEN"] == "hf_config"
