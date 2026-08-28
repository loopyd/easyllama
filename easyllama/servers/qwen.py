"""Implement the qwen llama.cpp server launcher."""

from __future__ import annotations

from pathlib import Path

from .base import BuildSource, RuntimeModeMetadata, server_metadata
from .llamacpp import LlamaCppServer


@server_metadata(
    name="qwen",
    help="Run the Qwen auxiliary llama.cpp server launcher",
    runtime_modes=(
        RuntimeModeMetadata(
            mode="qwen",
            docker_target="runtime-qwen",
            backend="vllm",
            build_sources=(
                BuildSource(
                    label="llamacpp-auxiliary",
                    repo_attr="qwen",
                    repo_build_arg="LLAMA_CPP_REPO",
                    ref_build_arg="LLAMA_CPP_REF",
                ),
            ),
        ),
    ),
)
class QwenServer(LlamaCppServer):
    """Run the qwen-specific llama.cpp binary."""

    def parser(self):
        parser = super().parser()
        parser.set_defaults(bin=Path("/app/bin/llama-server-qwen"))
        return parser
