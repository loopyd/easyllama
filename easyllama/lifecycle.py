"""Keep a backend container alive while loading at most one model on demand."""

from __future__ import annotations

import argparse
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import signal
import subprocess
from threading import Condition, Lock, Thread
from typing import Any, cast

from .helpers.logger import LOG as APP_LOG

LOGGER = APP_LOG.get(__name__)


class Backend:
    """Start, stop, and observe one model-server process."""

    def __init__(self, command: list[str]) -> None:
        self.command = command
        self.proc: subprocess.Popen[bytes] | None = None
        self.lock = Lock()
        self.changed = Condition(self.lock)

    def wake(self) -> int:
        """Start the backend when it is not already running."""
        with self.changed:
            if self.proc is None or self.proc.poll() is not None:
                LOGGER.info("waking model backend")
                self.proc = subprocess.Popen(self.command, start_new_session=True)
                self.changed.notify_all()
            return self.proc.pid

    def sleep(self) -> None:
        """Stop the backend and wait until its GPU allocations are released."""
        with self.changed:
            proc = self.proc
            if proc is None or proc.poll() is not None:
                self.proc = None
                return
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

    def status(self) -> dict[str, object]:
        """Return bounded lifecycle state."""
        with self.changed:
            proc = self.proc
            running = proc is not None and proc.poll() is None
            return {
                "state": "awake" if running else "sleeping",
                "pid": proc.pid if proc is not None and running else None,
            }


class LifecycleServer(ThreadingHTTPServer):
    """Private HTTP control plane for one backend."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], backend: Backend) -> None:
        super().__init__(address, LifecycleHandler)
        self.backend = backend


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
    backend = Backend(command)
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
