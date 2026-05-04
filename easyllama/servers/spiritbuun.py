from __future__ import annotations

import argparse
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
from starlette.background import BackgroundTask

from .common import ensure_gguf, hf_args, hf_file, hf_get, hf_snap
from .server_base import BuildSource, RuntimeModeMetadata, ServerBase, Spec, server_metadata

DEFAULT_BIN = Path("/app/bin/llama-server-spiritbuun")
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_UPSTREAM_HOST = "127.0.0.1"
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)
CHAT_PROXY_PATHS = {
    "/chat/completions",
    "/responses",
    "/v1/chat/completions",
    "/v1/messages",
    "/v1/responses",
}
DROP_CONTENT_BLOCK_TYPES = {"reasoning", "redacted_thinking", "thinking"}
DROP_MESSAGE_KEYS = {"reasoning_content", "thinking"}
HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def _default_upstream_port(public_port: int) -> int:
    return public_port + 1000 if public_port <= 64535 else public_port - 1000


def _strip_think_tags(text: str) -> tuple[str, bool]:
    sanitized = THINK_BLOCK_RE.sub("", text)
    return sanitized, sanitized != text


def _sanitize_content(content: Any) -> tuple[Any, bool]:
    if isinstance(content, str):
        return _strip_think_tags(content)

    if not isinstance(content, list):
        return content, False

    sanitized_content: list[Any] = []
    changed = False
    for block in content:
        if isinstance(block, dict):
            block_type = str(block.get("type", "")).lower()
            if block_type in DROP_CONTENT_BLOCK_TYPES:
                changed = True
                continue
            sanitized_block, block_changed = _sanitize_message_node(block)
            sanitized_content.append(sanitized_block)
            changed = changed or block_changed
            continue
        if isinstance(block, str):
            sanitized_block, block_changed = _strip_think_tags(block)
            sanitized_content.append(sanitized_block)
            changed = changed or block_changed
            continue
        sanitized_content.append(block)

    return sanitized_content, changed


def _sanitize_message_node(node: Any) -> tuple[Any, bool]:
    if isinstance(node, list):
        sanitized_items: list[Any] = []
        changed = False
        for item in node:
            sanitized_item, item_changed = _sanitize_message_node(item)
            sanitized_items.append(sanitized_item)
            changed = changed or item_changed
        return sanitized_items, changed

    if not isinstance(node, dict):
        return node, False

    sanitized_node: dict[str, Any] = {}
    changed = False
    for key, value in node.items():
        if key in DROP_MESSAGE_KEYS:
            changed = True
            continue
        if key == "content":
            sanitized_value, value_changed = _sanitize_content(value)
        elif key == "text" and isinstance(value, str):
            sanitized_value, value_changed = _strip_think_tags(value)
        else:
            sanitized_value, value_changed = _sanitize_message_node(value)
        sanitized_node[key] = sanitized_value
        changed = changed or value_changed

    return sanitized_node, changed


def sanitize_chat_request_body(body: bytes) -> tuple[bytes, bool]:
    try:
        payload = json.loads(body)
    except (TypeError, ValueError):
        return body, False

    if not isinstance(payload, dict):
        return body, False

    changed = False
    for key in ("thinking", "thinking_budget_tokens"):
        if key in payload:
            payload.pop(key)
            changed = True

    if payload.get("reasoning") != "off":
        payload["reasoning"] = "off"
        changed = True

    max_tokens = payload.get("max_tokens")
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        alt_max_tokens = payload.get("max_completion_tokens")
        if isinstance(alt_max_tokens, int) and alt_max_tokens > 0:
            payload["max_tokens"] = alt_max_tokens
            changed = True

    kwargs = payload.get("chat_template_kwargs")
    normalized_kwargs = dict(kwargs) if isinstance(kwargs, dict) else {}
    if normalized_kwargs.get("enable_thinking") is not False:
        normalized_kwargs["enable_thinking"] = False
        changed = True
    if normalized_kwargs.get("preserve_thinking") is not False:
        normalized_kwargs["preserve_thinking"] = False
        changed = True
    if payload.get("chat_template_kwargs") != normalized_kwargs:
        payload["chat_template_kwargs"] = normalized_kwargs

    for key in ("input", "messages"):
        if key not in payload:
            continue
        sanitized_value, value_changed = _sanitize_message_node(payload[key])
        payload[key] = sanitized_value
        changed = changed or value_changed

    if not changed:
        return body, False

    return json.dumps(payload, separators=(",", ":")).encode("utf-8"), True


def _filtered_headers(headers: Iterable[tuple[str, str]]) -> dict[str, str]:
    return {
        key: value
        for key, value in headers
        if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "host"
    }


def build_spiritbuun_proxy_app(
    server: ServerBase,
    *,
    upstream_cmd: list[str],
    upstream_env: dict[str, str],
    upstream_url: str,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.client = None
        server.proc = subprocess.Popen(upstream_cmd, env=upstream_env, start_new_session=True)
        app.state.client = httpx.AsyncClient(timeout=None)
        try:
            yield
        finally:
            client = app.state.client
            if client is not None:
                await client.aclose()
                app.state.client = None
            server.stop()

    app = FastAPI(lifespan=lifespan)

    @app.api_route(
        "/{path:path}",
        methods=["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"],
        response_model=None,
    )
    async def proxy(path: str, request: Request) -> Any:
        del path
        body = await request.body()
        path_value = request.url.path
        content_type = request.headers.get("content-type", "")
        if (
            request.method == "POST"
            and path_value in CHAT_PROXY_PATHS
            and "application/json" in content_type
        ):
            body, changed = sanitize_chat_request_body(body)
            if changed:
                server.log.debug("Sanitized Spiritbuun chat request for %s", path_value)

        target_url = f"{upstream_url}{path_value}"
        if request.url.query:
            target_url = f"{target_url}?{request.url.query}"

        headers = _filtered_headers(request.headers.items())
        client = app.state.client
        if client is None:
            return JSONResponse({"error": "Spiritbuun upstream is not ready"}, status_code=503)

        try:
            upstream_request = client.build_request(
                request.method,
                target_url,
                headers=headers,
                content=body,
            )
            upstream_response = await client.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            return JSONResponse(
                {"error": f"failed to reach Spiritbuun upstream: {exc}"},
                status_code=502,
            )

        response_headers = _filtered_headers(upstream_response.headers.items())
        return StreamingResponse(
            upstream_response.aiter_raw(),
            status_code=upstream_response.status_code,
            headers=response_headers,
            background=BackgroundTask(upstream_response.aclose),
        )

    return app


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
        parser.add_argument("--host", default=DEFAULT_HOST)
        parser.add_argument("--port", type=int, default=DEFAULT_PORT)
        parser.add_argument(
            "--bin", type=Path, default=DEFAULT_BIN, help="Spiritbuun llama-server binary to exec"
        )
        parser.add_argument(
            "--upstream-host",
            default=DEFAULT_UPSTREAM_HOST,
            help="Internal host for the raw Spiritbuun llama-server backend.",
        )
        parser.add_argument(
            "--upstream-port",
            type=int,
            default=None,
            help="Internal port for the raw Spiritbuun llama-server backend.",
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
        upstream_port = args.upstream_port or _default_upstream_port(args.port)
        upstream_cmd = [
            str(args.bin),
            "-m",
            str(target),
            "-md",
            str(draft),
            "--spec-type",
            "dflash",
            "--host",
            args.upstream_host,
            "--port",
            str(upstream_port),
            *extra,
        ]
        upstream_env = self.proc_env(args.bin)
        upstream_url = f"http://{args.upstream_host}:{upstream_port}"
        return Spec(
            app=build_spiritbuun_proxy_app(
                self,
                upstream_cmd=upstream_cmd,
                upstream_env=upstream_env,
                upstream_url=upstream_url,
            ),
            host=args.host,
            port=args.port,
            env=upstream_env,
            data={
                "bin": args.bin,
                "target": target,
                "draft": draft,
                "upstream_cmd": upstream_cmd,
                "upstream_url": upstream_url,
            },
        )

    def warmup(self, spec: Spec) -> None:
        self.log.info("Spiritbuun proxy listening on http://%s:%s", spec.host, spec.port)
        self.log.info("Launching Spiritbuun llama-server via %s", spec.data["bin"])
        self.log.info("Target model resolved to %s", spec.data["target"])
        self.log.info("Draft model resolved to %s", spec.data["draft"])
        self.log.info("Upstream Spiritbuun backend: %s", spec.data["upstream_url"])

    def run(self, spec: Spec) -> int:
        import uvicorn

        uvicorn.run(spec.app, host=spec.host, port=spec.port, log_level="info")
        return 0