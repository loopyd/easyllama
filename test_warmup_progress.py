from __future__ import annotations

import os
from typing import cast
from unittest.mock import patch

from easyllama.config import RUNTIME_CONTAINER, Settings
from easyllama.runtime import warmup_models

settings = cast(Settings, type("Settings", (), {"runtime_mode": RUNTIME_CONTAINER})())
with (
    patch("easyllama.runtime.load_auth", return_value={}),
    patch("easyllama.runtime.resolved_api_key", return_value=None),
    patch("easyllama.runtime.listen_url", return_value="http://localhost:8080"),
    patch("easyllama.runtime._http_status", return_value=200),
    patch("easyllama.runtime._http_response", return_value=(202, "")) as response,
    patch("easyllama.runtime.model_status", return_value={"state": "ready"}),
    patch("easyllama.runtime._prefetch_models"),
    patch.dict(os.environ, {"LLAMACPP_WARMUP_POLL_INTERVAL": "2"}),
):
    assert warmup_models(settings, ["model"]) == 0

assert response.call_count == 1
assert response.call_args.kwargs["timeout"] == 2.0
