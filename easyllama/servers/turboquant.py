"""Implement the turboquant llama.cpp server launcher."""

from __future__ import annotations

from pathlib import Path

from .base import BuildSource, RuntimeModeMetadata, server_metadata
from .llamacpp import LlamaCppServer


@server_metadata(
    name="turboquant",
    help="Run the Turboquant llama.cpp server launcher",
    runtime_modes=(
        RuntimeModeMetadata(
            mode="turboquant",
            docker_target="runtime-turboquant",
            backend="llamacpp",
            build_sources=(
                BuildSource(
                    label="turboquant",
                    repo_attr="turboquant",
                    repo_build_arg="LLAMA_CPP_REPO",
                    ref_build_arg="LLAMA_CPP_REF",
                ),
            ),
        ),
    ),
)
class TurboquantServer(LlamaCppServer):
    """Run the turboquant-specific llama.cpp binary."""

    def parser(self):
        parser = super().parser()
        parser.set_defaults(bin=Path("/app/bin/llama-server-turboquant"))
        return parser
