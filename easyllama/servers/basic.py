from __future__ import annotations

import argparse
from pathlib import Path

from .common import ensure_gguf, hf_args, hf_file, hf_get, hf_snap
from .server_base import ServerBase, Spec

DEFAULT_BIN = Path("/app/bin/llama-server-basic")


class BasicServer(ServerBase):
    name = "basic"
    help = "Run the plain llama-server launcher"

    def add_args(self, parser: argparse.ArgumentParser) -> None:
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
        repo, file = hf_args(label="model", spec=args.hf, repo=args.hf_repo, file=args.hf_file)
        if repo:
            file = hf_file(
                repo, file, "model", suffixes=(".gguf", ".safetensors", ".safetensors.index.json")
            )
            if file.endswith(".gguf"):
                return hf_get(repo, file, "model")
            snap = hf_snap(repo, "model")
            src = snap / file
            if not src.exists():
                raise SystemExit(f"model source {file} not found in {repo}")
            return ensure_gguf(src, label="model", repo=repo, outtype=args.gguf_outtype)
        if args.model is None:
            raise SystemExit("model path or HF selector is required")
        return ensure_gguf(args.model, label="model", outtype=args.gguf_outtype)

    def build(self, args: argparse.Namespace, extra: list[str]) -> Spec:
        if not args.bin.is_file():
            raise SystemExit(f"binary not found at {args.bin}")
        model = self.model_path(args)
        return Spec(
            cmd=[str(args.bin), "-m", str(model), *extra],
            env=self.proc_env(args.bin),
            data={"bin": args.bin, "model": model},
        )

    def warmup(self, spec: Spec) -> None:
        self.log.info("Launching llama-server via %s", spec.data["bin"])
        self.log.info("Model resolved to %s", spec.data["model"])

    def run(self, spec: Spec) -> int:
        return self.run_proc(spec.cmd, env=spec.env)
