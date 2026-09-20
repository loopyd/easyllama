"""Keep a backend container alive while loading at most one model on demand.

Sleep modes (per backend, via environment in the model container):

``EASYLLAMA_SLEEP_MODE=process`` (default)
    Classic behaviour: SIGTERM the backend on sleep, cold-start on wake.

``EASYLLAMA_SLEEP_MODE=slot`` (llama.cpp servers)
    Before killing the backend, every non-empty slot's KV cache is saved via
    ``POST /slots/{id}?action=save`` into the ``--slot-save-path`` directory
    (a tracked host cache mount). After a cold start, the newest save is
    restored into slot 0 via ``POST /slots/0?action=restore``, so the session
    resumes without re-prefilling its whole conversation. The backend API base
    and the save directory are derived from the server command's ``--port``
    and ``--slot-save-path`` arguments (``EASYLLAMA_HTTP_BASE`` /
    ``EASYLLAMA_SLOT_CACHE_DIR`` override them).

``EASYLLAMA_SLEEP_MODE=http`` (vLLM with ``--enable-sleep-mode``)
    The process stays alive across swaps. Sleep calls
    ``POST /sleep?level=1`` (weights offloaded to host RAM, KV discarded to
    LMCache's disk backend) freeing the GPU; wake calls ``POST /wake_up``,
    which reloads weights over PCIe in seconds instead of re-downloading and
    re-initialising. The backend API base is derived from the command's
    ``--port`` (``EASYLLAMA_HTTP_BASE`` overrides it) and the server must be
    started with ``VLLM_SERVER_DEV_MODE=1`` so the sleep endpoints are
    registered.
"""

from __future__ import annotations

import argparse
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
from threading import Condition, Lock, Thread
import time
from typing import Any, cast
import urllib.error
import urllib.request

from .helpers.logger import LOG as APP_LOG

LOGGER = APP_LOG.get(__name__)

# Slot-save guards: skip trivially small contexts (the save is a disk
# round-trip only worth taking for real conversations) and never write when
# the target filesystem is nearly full.
MIN_SLOT_SAVE_TOKENS = 256
MIN_SLOT_SAVE_FREE_BYTES = 2 * (1 << 30)


def _http_json(
    method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 10.0
) -> Any:
    """Issue one HTTP request and decode a JSON body (empty for raw endpoints)."""
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    if not body:
        return {}
    try:
        return json.loads(body)
    except ValueError:
        return {}


def _command_tokens(command: list[str]) -> list[str]:
    """Flatten a backend command into comparable argv tokens.

    The orchestrator launches backends through ``/bin/sh -lc 'exec ...'``, so
    the server flags usually live inside a single shell-word argument rather
    than as discrete argv entries. Split those shell words the way the shell
    would before scanning for flags, so derivation keeps working for both the
    wrapped and the already-tokenised form.
    """
    tokens: list[str] = []
    for arg in command:
        if any(character.isspace() for character in arg):
            try:
                tokens.extend(shlex.split(arg))
                continue
            except ValueError:
                pass  # unbalanced quotes: treat it as an opaque token
        tokens.append(arg)
    return tokens


def _flag_value(command: list[str], flag: str) -> str | None:
    """Return the value following ``flag`` in a backend command line, if any."""
    tokens = _command_tokens(command)
    for index, arg in enumerate(tokens):
        if arg == flag and index + 1 < len(tokens):
            return tokens[index + 1]
        if arg.startswith(f"{flag}="):
            return arg[len(flag) + 1 :]
    return None


def _api_base_from_command(command: list[str]) -> str | None:
    """Derive the backend API base URL from the command's ``--port`` argument."""
    port = _flag_value(command, "--port")
    if port is not None and port.isdigit():
        return f"http://127.0.0.1:{int(port)}"
    return None


class Backend:
    """Start, stop, and observe one model-server process."""

    def __init__(
        self,
        command: list[str],
        *,
        sleep_mode: str = "process",
        http_base: str | None = None,
        slot_cache_dir: str | None = None,
    ) -> None:
        self.command = command
        self.proc: subprocess.Popen[bytes] | None = None
        self.lock = Lock()
        self.changed = Condition(self.lock)
        self.sleep_mode = sleep_mode
        self.http_base = http_base.rstrip("/") if http_base else None
        self.slot_cache_dir = slot_cache_dir

    def wake(self) -> int:
        """Bring the backend up when it is not already serving.

        A concurrent /sleep may still be tearing down the previous incarnation
        (SIGTERM sent, process still releasing its CUDA allocations). Wait it
        out first so we never hand back a dying pid as the running backend and
        never start a second model over a live one: both would make the
        orchestrator believe the model is up when it is not, leaving the swap
        group with no restorable model.

        In http sleep mode the process survives sleep, so wake is either a
        no-op (already awake), a ``/wake_up`` round-trip (seconds: weights come
        back from host RAM over PCIe), or a full cold start (first use, or the
        process died).
        """
        with self.changed:
            proc = self.proc
            if proc is not None and proc.poll() is None and self.sleep_mode == "http":
                if not self._backend_sleeping():
                    LOGGER.debug("wake: backend pid=%s already awake", proc.pid)
                    return proc.pid
                LOGGER.info("waking model backend via http (pid=%s)", proc.pid)
                self._http_post(f"{self.http_base}/wake_up", timeout=30)
                # /wake_up can return before the restore completes; hold until
                # the engine reports itself awake (readiness gating for
                # queued requests is still the orchestrator's job).
                self._wait_while_locked(
                    lambda: self._backend_sleeping(),
                    timeout=180,
                    message=f"wake_up of pid={proc.pid} did not finish in time",
                )
                return proc.pid
            if proc is not None and proc.poll() is None:
                LOGGER.warning(
                    "wake: previous backend pid=%s still terminating; waiting for exit",
                    proc.pid,
                )
                settled = self.changed.wait_for(
                    lambda: self.proc is None or self.proc.poll() is not None,
                    timeout=45,
                )
                if not settled:
                    current = self.proc
                    if current is not None and current is proc and current.poll() is None:
                        LOGGER.error("wake: pid=%s stuck; escalating to SIGKILL", proc.pid)
                        with suppress(ProcessLookupError):
                            os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
            current = self.proc
            if current is None or current.poll() is not None:
                # No backend, or the previous incarnation died (crash, OOM,
                # or a completed sleep): start a fresh one. A live Popen here
                # can only be a still-terminating process from a racing sleep.
                LOGGER.info("waking model backend")
                current = subprocess.Popen(self.command, start_new_session=True)
                self.proc = current
                self.changed.notify_all()
                if self.sleep_mode == "slot":
                    Thread(target=self._restore_slots, daemon=True).start()
            return current.pid

    def sleep(self) -> None:
        """Free the backend's GPU memory, keeping whatever state is valuable.

        process: SIGTERM (default, legacy).
        slot:    persist non-empty slots' KV caches first, then SIGTERM.
        http:    keep the process alive; offload weights to host RAM.
        """
        with self.changed:
            proc = self.proc
            if proc is None or proc.poll() is not None:
                self.proc = None
                return
            if self.sleep_mode == "http":
                LOGGER.info("sleeping model backend via http level 1 (pid=%s)", proc.pid)
                self._http_post(f"{self.http_base}/sleep?level=1&mode=abort", timeout=30)
                # Offloading ~tens of GB of weights over PCIe is fast but not
                # instantaneous; hold until the engine reports sleep so the
                # orchestrator is never told "free" while VRAM is still held.
                self._wait_while_locked(
                    lambda: not self._backend_sleeping(),
                    timeout=180,
                    message=f"sleep of pid={proc.pid} did not complete in time",
                )
                return
            if self.sleep_mode == "slot":
                self._save_slots()
            LOGGER.info("sleeping model backend pid=%s", proc.pid)
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        with self.changed:
            if self.proc is proc:
                self.proc = None
            self.changed.notify_all()

    def wait(self, pid: int) -> int:
        """Wait for the selected backend incarnation to exit."""
        with self.changed:
            proc = self.proc
        if proc is None or proc.pid != pid:
            return 0
        return proc.wait()

    def _http_post(self, url: str, timeout: float = 10.0) -> Any:
        """POST an empty body and return the decoded response (best effort)."""
        return _http_json("POST", url, timeout=timeout)

    def _backend_health(self) -> bool:
        """Return whether the backend API answers on its health endpoint."""
        if self.http_base is None:
            return True
        try:
            _http_json("GET", f"{self.http_base}/health", timeout=3)
            return True
        except urllib.error.HTTPError:
            return True  # the server answered (404 path differences are fine)
        except Exception:
            return False

    def _backend_sleeping(self) -> bool:
        """Ask a vLLM backend whether its engine is in sleep mode."""
        if self.http_base is None:
            return False
        try:
            payload = _http_json("GET", f"{self.http_base}/is_sleeping", timeout=5)
            return bool(payload.get("is_sleeping", False))
        except Exception:
            return False

    def _wait_while_locked(self, predicate: Any, timeout: float, message: str) -> bool:
        """Poll ``predicate()`` until it is false or the deadline passes.

        Called with the backend lock held; predicates only touch the network,
        so blocking here merely serialises lifecycle operations, which is the
        point.
        """
        deadline = time.monotonic() + timeout
        while predicate():
            if time.monotonic() >= deadline:
                LOGGER.error(message)
                return False
            time.sleep(0.5)
        return True

    def _save_slots(self) -> None:
        """Persist every non-empty slot's KV cache before the process exits.

        Files land in the backend's ``--slot-save-path`` directory (a tracked
        host cache mount), so they survive the kill and are restorable after a
        cold start. Save failures are logged and swallowed: the cache is an
        optimisation, and sleep must not be blocked by it.
        """
        if self.http_base is None:
            return
        try:
            slots = _http_json("GET", f"{self.http_base}/slots", timeout=10)
        except Exception as exc:
            LOGGER.warning("slot save: cannot list slots: %s", exc)
            return
        if not isinstance(slots, list):
            return
        if self.slot_cache_dir:
            try:
                usage = shutil.disk_usage(self.slot_cache_dir)
                if usage.free < MIN_SLOT_SAVE_FREE_BYTES:
                    LOGGER.warning(
                        "slot save: only %d GiB free on %s; skipping to avoid filling the disk",
                        usage.free // (1 << 30),
                        self.slot_cache_dir,
                    )
                    return
            except OSError:
                pass
        for slot in slots:
            if not isinstance(slot, dict):
                continue
            slot_id = slot.get("id")
            prompt = slot.get("prompt") or {}
            tokens = prompt.get("tokens") if isinstance(prompt, dict) else None
            if slot_id is None or not tokens:
                continue
            if len(tokens) < MIN_SLOT_SAVE_TOKENS:
                continue  # trivially small context: not worth a disk round-trip
            filename = f"slot_{slot_id}.bin"
            try:
                _http_json(
                    "POST",
                    f"{self.http_base}/slots/{slot_id}?action=save&filename={filename}",
                    timeout=300,
                )
                LOGGER.info(
                    "slot save: saved slot %s (%d tokens) to %s", slot_id, len(tokens), filename
                )
            except Exception as exc:
                LOGGER.warning("slot save: slot %s failed: %s", slot_id, exc)

    def _restore_slots(self) -> None:
        """After a cold start, restore the newest saved slot into slot 0.

        Runs on a daemon thread: the server may still be loading its model
        when the process is spawned, so poll its health first. The restored
        KV gives the session's conversation back without a full re-prefill.
        """
        if self.http_base is None or not self.slot_cache_dir:
            return
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            if self._backend_health():
                break
            time.sleep(2)
        else:
            LOGGER.warning("slot restore: server never became healthy; skipping")
            return
        try:
            candidates = []
            for name in os.listdir(self.slot_cache_dir):
                if name.startswith("slot_") and name.endswith(".bin"):
                    path = os.path.join(self.slot_cache_dir, name)
                    candidates.append((os.path.getmtime(path), name))
        except OSError as exc:
            LOGGER.warning("slot restore: cannot scan %s: %s", self.slot_cache_dir, exc)
            return
        if not candidates:
            LOGGER.debug("slot restore: no saved slots found")
            return
        candidates.sort(reverse=True)
        filename = candidates[0][1]
        try:
            _http_json(
                "POST", f"{self.http_base}/slots/0?action=restore&filename={filename}", timeout=300
            )
            LOGGER.info("slot restore: restored %s into slot 0", filename)
        except Exception as exc:
            LOGGER.warning("slot restore: %s failed: %s", filename, exc)

    def status(self) -> dict[str, object]:
        """Return bounded lifecycle state."""
        with self.changed:
            proc = self.proc
            if proc is None or proc.poll() is not None:
                return {"state": "sleeping", "pid": None, "sleep_mode": self.sleep_mode}
            return {"state": "awake", "pid": proc.pid, "sleep_mode": self.sleep_mode}


class LifecycleServer(ThreadingHTTPServer):
    """Private HTTP control plane for one backend."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], backend: Backend) -> None:
        super().__init__(address, LifecycleHandler)
        self.backend = backend

    def handle_error(self, request: Any, client_address: Any) -> None:
        """Report handler failures with the exception, not a bare traceback."""
        exc = sys.exception()
        if isinstance(exc, (ConnectionError, TimeoutError)):
            LOGGER.debug("dropped lifecycle request from %s: %s", client_address, exc)
            return
        LOGGER.error(
            "lifecycle request from %s failed: %s: %s",
            client_address,
            type(exc).__name__ if exc else "unknown error",
            exc,
            exc_info=exc,
        )


class LifecycleHandler(BaseHTTPRequestHandler):
    """Expose health, wake, held-run, sleep, and status operations."""

    @property
    def lifecycle(self) -> LifecycleServer:
        """Return the narrowed server type."""
        return cast(LifecycleServer, self.server)

    def _reply(self, status: int, payload: dict[str, object] | None = None) -> None:
        body = json.dumps(payload or {}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._reply(200, {"status": "ok"})
        elif self.path == "/status":
            self._reply(200, self.lifecycle.backend.status())
        else:
            self._reply(404)

    def do_POST(self) -> None:
        if self.path == "/wake":
            self._reply(200, {"pid": self.lifecycle.backend.wake()})
        elif self.path == "/run":
            pid = self.lifecycle.backend.wake()
            returncode = self.lifecycle.backend.wait(pid)
            self._reply(200 if returncode == 0 else 502, {"returncode": returncode})
        elif self.path == "/sleep":
            self.lifecycle.backend.sleep()
            self._reply(200, {"state": "sleeping"})
        else:
            self._reply(404)

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.debug(format, *args)


def serve_lifecycle(host: str, port: int, command: list[str]) -> int:
    """Serve lifecycle events until the container receives a stop signal."""
    if not command:
        raise SystemExit("lifecycle requires a backend command after --")
    sleep_mode = os.environ.get("EASYLLAMA_SLEEP_MODE", "process").strip().lower()
    if sleep_mode not in ("process", "slot", "http"):
        raise SystemExit(f"invalid EASYLLAMA_SLEEP_MODE: {sleep_mode!r}")
    # Explicit environment wins; otherwise derive from the backend command so
    # the model config only has to state the server flags themselves.
    http_base = (
        os.environ.get("EASYLLAMA_HTTP_BASE", "").strip() or _api_base_from_command(command) or None
    )
    slot_cache_dir = (
        os.environ.get("EASYLLAMA_SLOT_CACHE_DIR", "").strip()
        or _flag_value(command, "--slot-save-path")
    )
    if sleep_mode in ("slot", "http") and not http_base:
        raise SystemExit(f"EASYLLAMA_SLEEP_MODE={sleep_mode} requires EASYLLAMA_HTTP_BASE")
    backend = Backend(
        command,
        sleep_mode=sleep_mode,
        http_base=http_base,
        slot_cache_dir=slot_cache_dir,
    )
    LOGGER.info("lifecycle configured: sleep_mode=%s http_base=%s", sleep_mode, http_base)
    server = LifecycleServer((host, port), backend)

    def stop(_signum: int, _frame: Any) -> None:
        Thread(target=server.shutdown, daemon=True).start()

    saved = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    LOGGER.info("model lifecycle controller listening on %s:%s (sleeping)", host, port)
    try:
        server.serve_forever()
    finally:
        backend.sleep()
        server.server_close()
        for sig, handler in saved.items():
            signal.signal(sig, handler)
    return 0


def lifecycle_main(argv: list[str]) -> int:
    """Parse lifecycle controller arguments."""
    parser = argparse.ArgumentParser(prog="easyllama lifecycle")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    return serve_lifecycle(args.host, args.port, command)
