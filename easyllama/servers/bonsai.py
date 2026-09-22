"""Implement the bonsai llama.cpp server launcher."""

from __future__ import annotations

from pathlib import Path

from .base import BuildSource, RuntimeModeMetadata, server_metadata
from .llamacpp import LlamaCppServer


@server_metadata(
    name="bonsai",
    help="Run the PrismML Bonsai llama.cpp server launcher",
    runtime_modes=(
        RuntimeModeMetadata(
            mode="bonsai",
            docker_target="runtime-bonsai",
            backend="llamacpp",
            build_sources=(
                BuildSource(
                    label="bonsai",
                    repo_attr="bonsai",
                    repo_build_arg="LLAMA_CPP_REPO",
                    ref_build_arg="LLAMA_CPP_REF",
                ),
            ),
        ),
    ),
)
class BonsaiServer(LlamaCppServer):
    """Run the PrismML fork llama.cpp binary that loads ternary Bonsai GGUFs."""

    def parser(self):
        parser = super().parser()
        parser.set_defaults(bin=Path("/app/bin/llama-server-bonsai"))
        return parser
