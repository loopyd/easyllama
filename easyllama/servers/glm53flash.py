"""Implement the glm5.3-flash FreeToken server launcher."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..helpers.hf import HuggingFace
from ..helpers.model import Model
from .base import BuildSource, RuntimeModeMetadata, ServerBase, Spec, server_metadata

DEFAULT_BIN = Path("/opt/ft-venv/bin/ft")
# FreeToken known-good checkpoint for GLM-5.3-Flash (320B MoE, 18B active). The FTW layout
# ships pre-packed expert banks, so the engine skips the slow in-load HF bank build.
DEFAULT_HF = "oakmindai/GLM-5.3-Flash-NVFP4-FTW"


def checkpoint_layout(checkpoint: Path) -> str:
    """Classify a FreeToken checkpoint as ``ftw`` or ``hf`` layout.

    Args:
        checkpoint: The checkpoint directory.

    Returns:
        str: The layout result.

    Raises:
        SystemExit: If the layout check cannot be completed."""
    if (checkpoint / "freetoken_weight.json").is_file() and any(checkpoint.glob("*.ftw")):
        return "ftw"
    if any(checkpoint.glob("*.safetensors")):
        return "hf"
    raise SystemExit(
        f"model checkpoint at {checkpoint} has no FTW weights "
        "(freetoken_weight.json + *.ftw) and no HF safetensors"
    )


@server_metadata(
    name="glm5.3-flash",
    help="Run the GLM-5.3 Flash FreeToken server launcher",
    runtime_modes=(
        RuntimeModeMetadata(
            mode="glm5.3-flash",
            docker_target="runtime-freetoken",
            backend="freetoken",
            build_sources=(
                BuildSource(
                    label="freetoken",
                    repo_attr="glm5.3-flash",
                    repo_build_arg="FREETOKEN_REPO",
                    ref_build_arg="FREETOKEN_REF",
                ),
            ),
        ),
    ),
)
class Glm53FlashServer(ServerBase):
    """Run the FreeToken API server for GLM-5.3 Flash."""

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        """Add mode-specific command-line arguments.

        Args:
            parser: The parser."""
        parser.add_argument("--bin", type=Path, default=DEFAULT_BIN, help="ft binary to exec")
        parser.add_argument(
            "-m",
            "--model",
            type=Path,
            default=None,
            help="Local FreeToken checkpoint directory (HF or FTW layout)",
        )
        parser.add_argument(
            "-hf",
            "--hf",
            "--model-hf",
            dest="hf",
            default=None,
            help=f"HF checkpoint repo (default {DEFAULT_HF})",
        )

    def model_path(self, args: argparse.Namespace) -> Path:
        """Resolve the checkpoint directory for FreeToken.

        Args:
            args: The args.

        Returns:
            Path: The model path result.

        Raises:
            SystemExit: If the model path operation cannot be completed."""
        if args.model is not None:
            if not args.model.is_dir():
                raise SystemExit(f"model checkpoint directory not found at {args.model}")
            return args.model
        hf, _file = HuggingFace.from_args("model", spec=args.hf or DEFAULT_HF, repo=None, file=None)
        checkpoint = Model("model", None, hf=hf).snapshot()
        if not checkpoint.is_dir():
            raise SystemExit(f"model checkpoint snapshot missing at {checkpoint}")
        return checkpoint

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
            cmd=[str(args.bin), "serve", "--model", str(model), *extra],
            env=self.proc_env(args.bin),
            data={"bin": args.bin, "model": model, "layout": checkpoint_layout(model)},
        )

    def warmup(self, spec: Spec) -> None:
        """Log resolved server assets before startup.

        Args:
            spec: The spec."""
        self.log.info("Launching ft serve via %s", spec.data["bin"])
        self.log.info(
            "Model checkpoint resolved to %s (%s layout)",
            spec.data["model"],
            spec.data["layout"],
        )

    def run(self, spec: Spec) -> int:
        """Run a built server process specification.

        Args:
            spec: The spec.

        Returns:
            int: The run result."""
        return self.run_proc(spec.cmd, env=spec.env)
