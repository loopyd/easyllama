"""Unit tests for lifecycle backend-address derivation.

Regression guard for the container startup failure where the orchestrator
launches every backend through ``/bin/sh -lc 'exec ...'``: the server flags
live inside a single shell-word argument, so scanning the argv list for a
literal ``--port`` token found nothing and ``EASYLLAMA_SLEEP_MODE=slot``
aborted the controller (and therefore the container) with
``EASYLLAMA_SLEEP_MODE=slot requires EASYLLAMA_HTTP_BASE``.
"""

from easyllama.lifecycle import _api_base_from_command, _flag_value
import pytest

pytestmark = pytest.mark.unit


def test_api_base_from_plain_argv() -> None:
    """A directly-spawned command still derives its base from ``--port``."""
    command = ["/opt/venv/bin/llama-server", "--host", "127.0.0.1", "--port", "9000"]

    assert _api_base_from_command(command) == "http://127.0.0.1:9000"


def test_api_base_from_shell_wrapped_command() -> None:
    """The orchestrator's ``/bin/sh -lc`` wrapper must not hide ``--port``."""
    command = [
        "/bin/sh",
        "-lc",
        "exec /opt/venv/bin/easyllama server qwen --host 127.0.0.1 --port 9000 --no-mmproj",
    ]

    assert _api_base_from_command(command) == "http://127.0.0.1:9000"


def test_slot_save_path_from_shell_wrapped_command() -> None:
    """Slot cache derivation reads through the same shell wrapper."""
    command = [
        "/bin/sh",
        "-lc",
        "exec /opt/venv/bin/easyllama server qwen --port 9000 "
        "--slot-save-path /root/.cache/slot-cache",
    ]

    assert _flag_value(command, "--slot-save-path") == "/root/.cache/slot-cache"


def test_flag_value_supports_equals_form() -> None:
    """``--port=9000`` is equivalent to ``--port 9000``."""
    command = ["/opt/venv/bin/llama-server", "--port=9000"]

    assert _flag_value(command, "--port") == "9000"
    assert _api_base_from_command(command) == "http://127.0.0.1:9000"


def test_quoted_value_containing_spaces_is_one_token() -> None:
    """Shell quoting is honoured so multi-word values survive tokenisation."""
    command = [
        "/bin/sh",
        "-lc",
        'exec /opt/venv/bin/vllm serve model --served-model-name "qwen 3" --port 9002',
    ]

    assert _flag_value(command, "--served-model-name") == "qwen 3"
    assert _api_base_from_command(command) == "http://127.0.0.1:9002"


def test_missing_port_yields_no_base() -> None:
    assert _api_base_from_command(["/opt/venv/bin/llama-server", "--host", "127.0.0.1"]) is None


def test_non_numeric_port_yields_no_base() -> None:
    assert _api_base_from_command(["--port", "${PORT}"]) is None
