from __future__ import annotations

import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from easyllama.config import RUNTIME_CONTAINER, Settings
from easyllama.runtime import _prefetch_models, _trigger_warmup, _warmup_state, warmup_models
from easyllama.servers.common import _HFProgress


class Messages(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def test_warmup_reports_loading_until_ready() -> None:
    settings = cast(Settings, type("Settings", (), {"runtime_mode": RUNTIME_CONTAINER})())
    with (
        patch("easyllama.runtime.load_auth", return_value={}),
        patch("easyllama.runtime.resolved_api_key", return_value=None),
        patch("easyllama.runtime.listen_url", return_value="http://localhost:8080"),
        patch("easyllama.runtime._http_status", return_value=200),
        patch(
            "easyllama.runtime.model_status",
            side_effect=[None, {"state": "starting"}, {"state": "ready"}],
        ),
        patch("easyllama.runtime._prefetch_models"),
        patch("easyllama.runtime.time.sleep"),
        patch.dict(os.environ, {"LLAMACPP_WARMUP_POLL_INTERVAL": "0.01"}),
    ):
        assert warmup_models(settings, ["model"]) == 0

    assert _warmup_state(None) == "loading"


def test_trigger_does_not_block_warmup() -> None:
    with patch("easyllama.runtime.threading.Thread") as thread:
        assert _trigger_warmup("http://swap", "model", headers={}, timeout=1800) == [None]
    thread.assert_called_once()
    thread.return_value.start.assert_called_once()


def test_prefetch_finds_hf_model_after_macro_expansion() -> None:
    with TemporaryDirectory() as temp_dir:
        config = Path(temp_dir) / "config.yml"
        config.write_text(
            "\n".join(
                (
                    "macros:",
                    "  model: owner/repo:Q4_K_M",
                    "models:",
                    "  test:",
                    "    cmd: |",
                    "      server -hf ${model}",
                )
            ),
            encoding="utf-8",
        )
        settings = cast(Settings, SimpleNamespace(models_dir=Path(temp_dir)))
        with (
            patch("easyllama.runtime.resolve_ls_config", return_value=config),
            patch("easyllama.runtime.hf_file", return_value="model.gguf"),
            patch("easyllama.runtime.hf_get") as get,
        ):
            _prefetch_models(settings, ["test"])

    get.assert_called_once_with(
        "owner/repo",
        "model.gguf",
        "test",
        cache_dir=Path(temp_dir),
        warmup="Warming model 1/1: test",
    )


def test_hf_progress_is_canonical_and_close_is_silent() -> None:
    messages = Messages()
    logger = logging.getLogger("easyllama.servers.common")
    logger.addHandler(messages)
    logger.setLevel(logging.INFO)
    try:
        with patch("easyllama.servers.common.time.monotonic", side_effect=[0, 6]):
            progress = _HFProgress(
                total=200,
                desc="model.gguf",
                warmup="Warming model 2/3: qwen3-chat",
            )
            progress.update(100)
            progress.close()
    finally:
        logger.removeHandler(messages)

    assert messages.messages == [
        "Warming model 2/3: qwen3-chat — model.gguf: "
        "100 B/200 B (50.0%, 17 B/s, ETA 6s)"
    ]


if __name__ == "__main__":
    test_warmup_reports_loading_until_ready()
    test_trigger_does_not_block_warmup()
    test_prefetch_finds_hf_model_after_macro_expansion()
    test_hf_progress_is_canonical_and_close_is_silent()
