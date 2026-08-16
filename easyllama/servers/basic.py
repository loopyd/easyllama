"""Implement the basic llama.cpp server mode."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..helpers.model import Model
from .base import BuildSource, RuntimeModeMetadata, ServerBase, Spec, server_metadata

DEFAULT_BIN = Path("/app/bin/llama-server-basic")


@server_metadata(
    name="basic",
    help="Run the plain llama-server launcher",
    runtime_modes=(
        RuntimeModeMetadata(
            mode="basic",
            docker_target="runtime-basic",
            build_sources=(
                BuildSource(
                    label="basic",
                    repo_attr="llama_cpp_repo",
                    ref_attr="llama_cpp_ref",
                    repo_build_arg="LLAMA_CPP_REPO",
                    ref_build_arg="LLAMA_CPP_REF",
                    default_repo="https://github.com/Luce-Org/llama.cpp.git",
                    default_ref="luce-dflash",
                ),
            ),
        ),
        RuntimeModeMetadata(
            mode="turboquant",
            docker_target="runtime-turboquant",
            build_sources=(
                BuildSource(
                    label="turboquant",
                    repo_attr="llama_cpp_repo",
                    ref_attr="llama_cpp_ref",
                    repo_build_arg="LLAMA_CPP_REPO",
                    ref_build_arg="LLAMA_CPP_REF",
                    default_repo="https://github.com/TheTom/llama-cpp-turboquant.git",
                    default_ref="feature/turboquant-kv-cache",
                ),
            ),
        ),
        RuntimeModeMetadata(
            mode="mtp",
            docker_target="runtime-mtp",
            backend="vllm",
            build_sources=(
                BuildSource(
                    label="llamacpp-auxiliary",
                    repo_attr="llama_cpp_repo",
                    ref_attr="llama_cpp_ref",
                    repo_build_arg="LLAMA_CPP_REPO",
                    ref_build_arg="LLAMA_CPP_REF",
                    default_repo="https://github.com/ggml-org/llama.cpp.git",
                    default_ref="master",
                ),
            ),
        ),
    ),
)
class BasicServer(ServerBase):
    """Represent BasicServer state and behavior."""

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        """Add mode-specific command-line arguments.

        Args:
            parser: The parser."""
        parser.add_argument(
            "--bin", type=Path, default=DEFAULT_BIN, help="llama-server binary to exec"
        )
        parser.add_argument(
            "-m", "--model", type=Path, default=None, help="Local GGUF or safetensors model source"
        )
        parser.add_argument(
            "-hf",
            "--hf",
            "--model-hf",
            dest="hf",
            default=None,
            help="HF model spec as repo:quant or repo:file",
        )
        parser.add_argument(
            "--hf-repo",
            "--model-hf-repo",
            dest="hf_repo",
            default=None,
            help="HF repo for the model when using split repo/file flags",
        )
        parser.add_argument(
            "--hf-file",
            "--model-hf-file",
            dest="hf_file",
            default=None,
            help="HF file or selector for the model when using split repo/file flags",
        )
        parser.add_argument(
            "--gguf-outtype",
            choices=["f16", "bf16"],
            default="bf16",
            help="Outtype to use when converting a safetensors model to cached GGUF",
        )

    def model_path(self, args: argparse.Namespace) -> Path:
        """Perform the model path operation.

        Args:
            args: The args.

        Returns:
            Path: The model path result.

        Raises:
            SystemExit: If the model path operation cannot be completed."""
        return Model.from_args(
            "model",
            local=args.model,
            spec=args.hf,
            repo=args.hf_repo,
            file=args.hf_file,
            outtype=args.gguf_outtype,
        ).resolve()

    def build(self, args: argparse.Namespace, extra: list[str]) -> Spec:
        """Build the process specification for parsed server arguments.

        Args:
            args: The args.
            extra: The extra.

        Returns:
            Spec: The build result.

        Raises:
            SystemExit: If the build operation cannot be completed."""
        if not args.bin.is_file():
            raise SystemExit(f"binary not found at {args.bin}")
        model = self.model_path(args)
        return Spec(
            cmd=[str(args.bin), "-m", str(model), *extra],
            env=self.proc_env(args.bin),
            data={"bin": args.bin, "model": model},
        )

    def warmup(self, spec: Spec) -> None:
        """Log resolved server assets before startup.

        Args:
            spec: The spec."""
        self.log.info("Launching llama-server via %s", spec.data["bin"])
        self.log.info("Model resolved to %s", spec.data["model"])

    def run(self, spec: Spec) -> int:
        """Run a built server process specification.

        Args:
            spec: The spec.

        Returns:
            int: The run result."""
        return self.run_proc(spec.cmd, env=spec.env)
