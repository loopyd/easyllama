from __future__ import annotations

import argparse
from pathlib import Path

from .common import ensure_gguf, hf_args, hf_file, hf_get, hf_snap
from .server_base import BuildSource, RuntimeModeMetadata, ServerBase, Spec, server_metadata

DEFAULT_BIN = Path("/app/bin/llama-server-spiritbuun")


@server_metadata(
    name="spiritbuun",
    help="Run the Spiritbuun dflash llama-server launcher",
    runtime_modes=(
        RuntimeModeMetadata(
            mode="spiritbuun",
            docker_target="runtime-spiritbuun",
            build_sources=(
                BuildSource(
                    label="spiritbuun",
                    repo_attr="spiritbuun_llama_cpp_repo",
                    ref_attr="spiritbuun_llama_cpp_ref",
                    repo_build_arg="SPIRITBUUN_LLAMA_CPP_REPO",
                    ref_build_arg="SPIRITBUUN_LLAMA_CPP_REF",
                ),
            ),
        ),
    ),
)
class SpiritbuunServer(ServerBase):
    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--bin", type=Path, default=DEFAULT_BIN, help="Spiritbuun llama-server binary to exec"
        )
        parser.add_argument(
            "--target",
            type=Path,
            default=None,
            help="Local GGUF or safetensors target model source",
        )
        parser.add_argument(
            "--target-hf",
            dest="target_hf",
            default=None,
            help="HF target spec as repo:quant or repo:file",
        )
        parser.add_argument(
            "--target-hf-repo",
            default=None,
            help="HF target repo for split repo/file syntax",
        )
        parser.add_argument(
            "--target-hf-file",
            default=None,
            help="HF target file or selector for split repo/file syntax",
        )
        parser.add_argument(
            "--target-gguf-outtype",
            choices=["f16", "bf16"],
            default="bf16",
            help="Outtype to use when converting a safetensors target to cached GGUF",
        )
        parser.add_argument(
            "--draft",
            type=Path,
            default=None,
            help="Local GGUF or safetensors Spiritbuun dflash draft source",
        )
        parser.add_argument(
            "--draft-hf",
            dest="draft_hf",
            default=None,
            help="HF draft spec as repo:quant or repo:file",
        )
        parser.add_argument(
            "--draft-hf-repo",
            default=None,
            help="HF draft repo for split repo/file syntax",
        )
        parser.add_argument(
            "--draft-hf-file",
            default=None,
            help="HF draft file or selector for split repo/file syntax",
        )
        parser.add_argument(
            "--draft-gguf-outtype",
            choices=["f16", "bf16"],
            default="bf16",
            help="Outtype to use when converting a safetensors draft to cached GGUF",
        )

    def _resolve_model(
        self,
        *,
        label: str,
        local: Path | None,
        spec: str | None,
        repo: str | None,
        file: str | None,
        outtype: str,
    ) -> Path:
        repo, file = hf_args(label=label, spec=spec, repo=repo, file=file)
        if repo:
            file = hf_file(
                repo,
                file,
                label,
                suffixes=(".gguf", ".safetensors", ".safetensors.index.json"),
            )
            if file.endswith(".gguf"):
                return hf_get(repo, file, label)
            snap = hf_snap(repo, label)
            src = snap / file
            if not src.exists():
                raise SystemExit(f"{label} source {file} not found in {repo}")
            return ensure_gguf(src, label=label, repo=repo, outtype=outtype)

        if local is None:
            raise SystemExit(f"{label} model path or HF selector is required")
        return ensure_gguf(local, label=label, outtype=outtype)

    def build(self, args: argparse.Namespace, extra: list[str]) -> Spec:
        if not args.bin.is_file():
            raise SystemExit(f"binary not found at {args.bin}")

        target = self._resolve_model(
            label="target",
            local=args.target,
            spec=args.target_hf,
            repo=args.target_hf_repo,
            file=args.target_hf_file,
            outtype=args.target_gguf_outtype,
        )
        draft = self._resolve_model(
            label="draft",
            local=args.draft,
            spec=args.draft_hf,
            repo=args.draft_hf_repo,
            file=args.draft_hf_file,
            outtype=args.draft_gguf_outtype,
        )
        cmd = [
            str(args.bin),
            "-m",
            str(target),
            "-md",
            str(draft),
            "--spec-type",
            "dflash",
            *extra,
        ]
        return Spec(
            cmd=cmd,
            env=self.proc_env(args.bin),
            data={
                "bin": args.bin,
                "target": target,
                "draft": draft,
            },
        )

    def warmup(self, spec: Spec) -> None:
        self.log.info("Launching Spiritbuun llama-server via %s", spec.data["bin"])
        self.log.info("Target model resolved to %s", spec.data["target"])
        self.log.info("Draft model resolved to %s", spec.data["draft"])

    def run(self, spec: Spec) -> int:
        return self.run_proc(spec.cmd, env=spec.env)