from __future__ import annotations

import argparse
from functools import lru_cache
import inspect
import json
import os
from pathlib import Path
import sys
from typing import Any, cast

from .common import ensure_gguf, hf_args, hf_file, hf_get, hf_snap, pick_asset
from .server_base import ServerBase, Spec

LUCE_SCRIPTS = Path("/opt/lucebox/dflash/scripts")
FALLBACK_MODEL = "qwen3-chat"
FALLBACK_TARGET = Path("/models/target.gguf")
FALLBACK_DRAFT = Path("/models/draft")
FALLBACK_BIN = Path("/opt/lucebox/dflash/build/test_dflash")
FALLBACK_BUDGET = 22
FALLBACK_CHAT_MAX_TOKENS = 512
FALLBACK_THINKING_BUDGET_EXTRA = 0


def patch_luce_finish_reason(luce: Any) -> None:
    if getattr(luce, "_easyllama_finish_reason_patched", False):
        return

    source = inspect.getsource(luce.build_app)
    non_stream_old = '''        msg: dict = {"role": "assistant"}
        finish_reason = "stop"
        if reasoning:
            msg["reasoning_content"] = reasoning
        if tool_calls:
            msg["content"] = cleaned if cleaned else None
            msg["tool_calls"] = tool_calls
            finish_reason = "tool_calls"
        else:
            msg["content"] = cleaned
'''
    non_stream_new = '''        msg: dict = {"role": "assistant"}
        finish_reason = "length" if len(tokens) >= gen_len else "stop"
        if reasoning:
            msg["reasoning_content"] = reasoning
        if tool_calls:
            msg["content"] = cleaned if cleaned else None
            msg["tool_calls"] = tool_calls
            finish_reason = "tool_calls"
        else:
            msg["content"] = cleaned
'''
    stream_old = '''                    finish_reason = "stop"
                    if mode == "tool_buffer":
                        cleaned_after, tool_calls = parse_tool_calls(tool_buffer, tools=req.tools)
'''
    stream_new = '''                    finish_reason = (
                        "length" if completion_tokens >= gen_len else "stop"
                    )
                    if mode == "tool_buffer":
                        cleaned_after, tool_calls = parse_tool_calls(tool_buffer, tools=req.tools)
'''
    if non_stream_old not in source or stream_old not in source:
        raise RuntimeError("unsupported Luce build_app source shape for finish_reason patch")

    patched_source = source.replace(non_stream_old, non_stream_new, 1).replace(
        stream_old,
        stream_new,
        1,
    )
    namespace = dict(luce.__dict__)
    exec(patched_source, namespace)
    luce.build_app = namespace["build_app"]
    luce._easyllama_finish_reason_patched = True


class ChatRequestBudgetMiddleware:
    def __init__(self, app: Any, extra_tokens: int) -> None:
        self.app = app
        self.extra_tokens = extra_tokens

    @staticmethod
    def _wrap_receive(body: bytes, original_receive: Any) -> Any:
        sent = False

        async def receive() -> dict[str, Any]:
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await original_receive()

        return receive

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if (
            scope.get("type") != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != "/v1/chat/completions"
        ):
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_type = headers.get(b"content-type", b"").decode("latin-1")
        if "application/json" not in content_type:
            await self.app(scope, receive, send)
            return

        chunks: list[bytes] = []
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] != "http.request":
                await self.app(scope, self._wrap_receive(b"".join(chunks), receive), send)
                return
            chunks.append(message.get("body", b""))
            more_body = message.get("more_body", False)

        body = b"".join(chunks)
        try:
            payload = json.loads(body)
        except (TypeError, ValueError):
            await self.app(scope, self._wrap_receive(body, receive), send)
            return

        if not isinstance(payload, dict):
            await self.app(scope, self._wrap_receive(body, receive), send)
            return

        payload_changed = False
        kwargs = payload.get("chat_template_kwargs")
        normalized_kwargs = dict(kwargs) if isinstance(kwargs, dict) else {}
        if normalized_kwargs.get("enable_thinking") is not False:
            normalized_kwargs["enable_thinking"] = False
            payload_changed = True
        if normalized_kwargs.get("preserve_thinking") is not False:
            normalized_kwargs["preserve_thinking"] = False
            payload_changed = True
        payload["chat_template_kwargs"] = normalized_kwargs

        if self.extra_tokens > 0:
            max_tokens = payload.get("max_tokens")
            if not isinstance(max_tokens, int) or max_tokens <= 0:
                alt_max_tokens = payload.get("max_completion_tokens")
                if isinstance(alt_max_tokens, int) and alt_max_tokens > 0:
                    max_tokens = alt_max_tokens
                else:
                    max_tokens = FALLBACK_CHAT_MAX_TOKENS

            payload["max_tokens"] = max_tokens + self.extra_tokens
            payload_changed = True

        if not payload_changed:
            await self.app(scope, self._wrap_receive(body, receive), send)
            return

        new_body = json.dumps(payload).encode("utf-8")
        new_scope = dict(scope)
        new_headers = [
            (key, value)
            for key, value in scope.get("headers", [])
            if key.lower() != b"content-length"
        ]
        new_headers.append((b"content-length", str(len(new_body)).encode("ascii")))
        new_scope["headers"] = new_headers
        await self.app(new_scope, self._wrap_receive(new_body, receive), send)


def install_chat_request_budget_middleware(app: Any, extra_tokens: int) -> None:
    if getattr(app, "_easyllama_chat_request_budget_middleware", False):
        return
    app.add_middleware(ChatRequestBudgetMiddleware, extra_tokens=extra_tokens)
    app._easyllama_chat_request_budget_middleware = True


@lru_cache(maxsize=1)
def load_luce() -> tuple[Any, Any, Any]:
    if str(LUCE_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(LUCE_SCRIPTS))

    from fastapi.responses import JSONResponse
    import server_tools as luce  # pyright: ignore[reportMissingImports]
    from transformers import AutoTokenizer

    patch_luce_finish_reason(luce)
    return luce, JSONResponse, AutoTokenizer


class LuceboxServer(ServerBase):
    name = "lucebox"
    help = "Run the Luce dflash server"

    def add_prefill_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--prefill-compression",
            choices=["off", "auto"],
            default="auto",
            help="Enable or disable speculative prefill compression.",
        )
        parser.add_argument(
            "--prefill-threshold",
            type=int,
            default=32000,
            help="Prompt length threshold for enabling speculative prefill.",
        )
        parser.add_argument(
            "--prefill-keep-ratio",
            type=float,
            default=0.05,
            help="Prompt keep ratio when speculative prefill is enabled.",
        )
        parser.add_argument(
            "--prefill-drafter",
            type=Path,
            default=None,
            help="Local prefill drafter GGUF or safetensors source.",
        )
        parser.add_argument(
            "--prefill-drafter-tokenizer",
            default="Qwen/Qwen3-0.6B",
            help="Tokenizer id for the prefill drafter model.",
        )

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        try:
            luce, _, _ = load_luce()
        except ModuleNotFoundError:
            luce = None
        parser.add_argument("--host", default="0.0.0.0")
        parser.add_argument("--port", type=int, default=8000)
        parser.add_argument("--model-name", default=luce.MODEL_NAME if luce else FALLBACK_MODEL)
        parser.add_argument(
            "--target", type=Path, default=luce.DEFAULT_TARGET if luce else FALLBACK_TARGET
        )
        parser.add_argument(
            "--target-hf", default=None, help="HF target spec as repo:quant or repo:file"
        )
        parser.add_argument(
            "--target-hf-repo", default=None, help="HF target repo for split repo/file syntax"
        )
        parser.add_argument(
            "--target-hf-file",
            default=None,
            help="HF target file or selector for split repo/file syntax",
        )
        parser.add_argument(
            "--draft", type=Path, default=luce.DEFAULT_DRAFT_ROOT if luce else FALLBACK_DRAFT
        )
        parser.add_argument(
            "--draft-hf", default=None, help="HF draft spec as repo[:file] or repo:selector"
        )
        parser.add_argument(
            "--draft-hf-repo", default=None, help="HF draft repo for split repo/file syntax"
        )
        parser.add_argument(
            "--draft-hf-file",
            default=None,
            help="HF draft file or selector for split repo/file syntax",
        )
        parser.add_argument("--bin", type=Path, default=luce.DEFAULT_BIN if luce else FALLBACK_BIN)
        parser.add_argument(
            "--budget", type=int, default=luce.DEFAULT_BUDGET if luce else FALLBACK_BUDGET
        )
        parser.add_argument(
            "--thinking-budget-extra",
            type=int,
            default=FALLBACK_THINKING_BUDGET_EXTRA,
            help=(
                "Extra generation tokens to reserve when thinking is enabled "
                "so final content is not cut off."
            ),
        )
        parser.add_argument(
            "--max-ctx",
            type=int,
            default=16384,
            help="Maximum context length; oversizing this can slow attention significantly.",
        )
        parser.add_argument(
            "--kv-f16",
            action="store_true",
            help="Force F16 KV cache instead of the default long-context fallback.",
        )
        parser.add_argument(
            "--cache-type-k",
            "--ctk",
            dest="cache_type_k",
            default=None,
            choices=["f16", "bf16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "tq3_0"],
            help="K cache element type.",
        )
        parser.add_argument(
            "--cache-type-v",
            "--ctv",
            dest="cache_type_v",
            default=None,
            choices=["f16", "bf16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "tq3_0"],
            help="V cache element type.",
        )
        parser.add_argument(
            "--fa-window",
            type=int,
            default=None,
            help="Sliding attention window; 0 means full attention.",
        )
        parser.add_argument(
            "--tokenizer",
            default="Qwen/Qwen3.5-27B",
            help="HF tokenizer id; Qwen3.6 shares this tokenizer.",
        )
        parser.add_argument(
            "--pflash-use-bsa",
            type=int,
            choices=[0, 1],
            default=1,
            help="Set DFLASH_FP_USE_BSA for speculative prefill.",
        )
        parser.add_argument(
            "--pflash-alpha",
            type=float,
            default=0.85,
            help="Set DFLASH_FP_ALPHA for speculative prefill block selection.",
        )
        parser.add_argument(
            "--pflash-profile",
            action="store_true",
            help="Set DFLASH_FP_PROFILE=1 to log pflash stage timings.",
        )
        if luce:
            luce.add_cli_flags(parser)
        else:
            self.add_prefill_args(parser)
        parser.add_argument(
            "--prefill-drafter-hf", default=None, help="HF drafter spec as repo:quant or repo:file"
        )
        parser.add_argument(
            "--prefill-drafter-hf-repo",
            default=None,
            help="HF prefill drafter repo for split repo/file syntax",
        )
        parser.add_argument(
            "--prefill-drafter-hf-file",
            default=None,
            help="HF prefill drafter file or selector for split repo/file syntax",
        )

    def drafter_path(self, args: argparse.Namespace) -> Path | None:
        repo, file = hf_args(
            label="prefill drafter",
            spec=args.prefill_drafter_hf,
            repo=args.prefill_drafter_hf_repo,
            file=args.prefill_drafter_hf_file,
        )
        if repo:
            file = hf_file(
                repo,
                file,
                "prefill drafter",
                suffixes=(".gguf", ".safetensors", ".safetensors.index.json"),
                default="model.safetensors",
            )
            if file.endswith(".gguf"):
                return hf_get(repo, file, "prefill drafter GGUF")
            snap = hf_snap(repo, "prefill drafter")
            src = snap / file
            if not src.exists():
                raise SystemExit(f"prefill drafter source {file} not found in {repo}")
            return ensure_gguf(src, label="prefill drafter", repo=repo, outtype="bf16")
        if args.prefill_drafter is None:
            return None
        return ensure_gguf(
            cast(Path, args.prefill_drafter), label="prefill drafter", outtype="bf16"
        )

    def set_env(self, args: argparse.Namespace) -> None:
        if not 0.0 < args.pflash_alpha < 1.0:
            raise SystemExit("--pflash-alpha must be in (0, 1)")
        if args.cache_type_k:
            os.environ["DFLASH27B_KV_K"] = args.cache_type_k
        if args.cache_type_v:
            os.environ["DFLASH27B_KV_V"] = args.cache_type_v
        if (
            args.max_ctx > 6144
            and not args.kv_f16
            and not args.cache_type_k
            and not args.cache_type_v
        ):
            os.environ.setdefault("DFLASH27B_KV_TQ3", "1")
        if args.fa_window is not None:
            os.environ["DFLASH27B_FA_WINDOW"] = str(args.fa_window)
        if args.prefill_compression != "off":
            os.environ.setdefault("DFLASH_FP_USE_BSA", str(args.pflash_use_bsa))
            os.environ.setdefault("DFLASH_FP_ALPHA", str(args.pflash_alpha))
            if args.pflash_profile:
                os.environ.setdefault("DFLASH_FP_PROFILE", "1")
            os.environ.setdefault("DFLASH27B_LM_HEAD_FIX", "0")
            os.environ.setdefault("DFLASH27B_FA_WINDOW", "0")

    def build(self, args: argparse.Namespace, extra: list[str]) -> Spec:
        if extra:
            raise SystemExit(f"unexpected extra args for lucebox server: {' '.join(extra)}")

        self.set_env(args)
        luce, JSONResponse, AutoTokenizer = load_luce()
        luce.MODEL_NAME = args.model_name

        target_repo, target_file = hf_args(
            label="target", spec=args.target_hf, repo=args.target_hf_repo, file=args.target_hf_file
        )
        draft_repo, draft_file = hf_args(
            label="draft",
            spec=args.draft_hf,
            repo=args.draft_hf_repo,
            file=args.draft_hf_file,
            default="model.safetensors",
        )
        target = pick_asset(
            label="target GGUF",
            local=None if target_repo else cast(Path, args.target),
            repo=target_repo,
            file=target_file,
            suffixes=(".gguf",),
        )
        draft_src = pick_asset(
            label="draft weights",
            local=None if draft_repo else cast(Path, args.draft),
            repo=draft_repo,
            file=draft_file,
            suffixes=(".safetensors", ".safetensors.index.json"),
            default="model.safetensors",
        )
        if args.prefill_compression != "off":
            args.prefill_drafter = self.drafter_path(args)

        if not args.bin.is_file():
            raise SystemExit(f"binary not found at {args.bin}")
        if not target.is_file():
            raise SystemExit(f"target GGUF not found at {target}")

        draft = luce.resolve_draft(draft_src) if draft_src.is_dir() else draft_src
        if not draft.is_file():
            raise SystemExit(f"draft safetensors not found at {draft_src}")

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        stop_ids: set[int] = set()
        for special in ("<|im_end|>", "<|endoftext|>"):
            ids = tokenizer.encode(special, add_special_tokens=False)
            if ids:
                stop_ids.add(ids[0])

        prefill = luce.config_from_args(args)
        drafter_tok = None
        if prefill.enabled:
            drafter_tok = AutoTokenizer.from_pretrained(
                prefill.drafter_tokenizer_id, trust_remote_code=True
            )

        app = luce.build_app(
            target,
            draft,
            args.bin,
            args.budget,
            args.max_ctx,
            tokenizer,
            stop_ids,
            prefill_cfg=prefill if prefill.enabled else None,
            drafter_tokenizer=drafter_tok,
        )
        install_chat_request_budget_middleware(app, args.thinking_budget_extra)

        @app.get("/health")
        def health() -> Any:
            return JSONResponse({"status": "ok", "model": args.model_name})

        return Spec(
            app=app,
            host=args.host,
            port=args.port,
            data={
                "model": args.model_name,
                "target": target,
                "draft": draft,
                "bin": args.bin,
                "budget": args.budget,
                "thinking_budget_extra": args.thinking_budget_extra,
                "ctx": args.max_ctx,
                "tokenizer": args.tokenizer,
                "prefill": prefill,
            },
        )

    def warmup(self, spec: Spec) -> None:
        prefill = spec.data["prefill"]
        self.log.info("Luce DFlash OpenAI server on http://%s:%s", spec.host, spec.port)
        self.log.info("Model name: %s", spec.data["model"])
        self.log.info("Target GGUF: %s", spec.data["target"])
        self.log.info("Draft weights: %s", spec.data["draft"])
        self.log.info("Binary: %s", spec.data["bin"])
        self.log.info("Budget: %s", spec.data["budget"])
        self.log.info("Thinking budget extra: %s", spec.data["thinking_budget_extra"])
        self.log.info("Max ctx: %s", spec.data["ctx"])
        self.log.info("Tokenizer: %s", spec.data["tokenizer"])
        if prefill.enabled:
            self.log.info(
                "Pflash mode=%s threshold=%s keep=%s drafter=%s",
                prefill.mode,
                prefill.threshold,
                prefill.keep_ratio,
                prefill.drafter_gguf,
            )
        else:
            self.log.info("Pflash disabled")

    def run(self, spec: Spec) -> int:
        import uvicorn

        uvicorn.run(spec.app, host=spec.host, port=spec.port, log_level="info")
        return 0
