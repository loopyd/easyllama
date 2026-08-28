import os

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.docker]


@pytest.mark.skipif(
    os.environ.get("EASYLLAMA_INTEGRATION") != "1",
    reason="set EASYLLAMA_INTEGRATION=1 to run Docker integration tests",
)
def test_docker_daemon() -> None:
    from docker import from_env

    assert from_env().ping()
