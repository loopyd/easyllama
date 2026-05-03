from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import os
from pathlib import Path
import signal
import subprocess
from typing import Any

from ..logger import get_logger


@dataclass(slots=True)
class Spec:
    cmd: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    app: Any = None
    host: str = "0.0.0.0"
    port: int = 0
    data: dict[str, Any] = field(default_factory=dict)


class ServerBase:
    name = ""
    help = ""

    def __init__(self) -> None:
        key = self.name or self.__class__.__name__.lower()
        self.log = get_logger(f"easyllama.server.{key}")
        self.proc: subprocess.Popen[str] | None = None
        self._signals: dict[signal.Signals, Any] = {}

    def parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(prog=f"easyllama server {self.name}")
        self.add_args(parser)
        return parser

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        del parser

    def parse(self, argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
        return self.parser().parse_known_args(argv)

    def build(self, args: argparse.Namespace, extra: list[str]) -> Spec:
        raise NotImplementedError

    def warmup(self, spec: Spec) -> None:
        del spec

    def run(self, spec: Spec) -> int:
        raise NotImplementedError

    def stop(self) -> int:
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return 0
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return 0
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        return 0

    def main(self, argv: list[str] | None = None) -> int:
        args, extra = self.parse(argv)
        spec = self.build(args, extra)
        try:
            self.warmup(spec)
            return self.run(spec)
        finally:
            self.stop()

    def proc_env(self, bin_path: Path) -> dict[str, str]:
        env = os.environ.copy()
        lib_dir = str(bin_path.resolve().parent)
        current = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = f"{lib_dir}:{current}" if current else lib_dir
        return env

    def run_proc(self, cmd: list[str], *, env: dict[str, str] | None = None) -> int:
        merged = os.environ.copy()
        if env:
            merged.update(env)
        self.proc = subprocess.Popen(cmd, env=merged, start_new_session=True)
        self._set_signals()
        try:
            return self.proc.wait()
        finally:
            self._reset_signals()
            self.proc = None

    def _set_signals(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._signals[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handle_signal)

    def _reset_signals(self) -> None:
        for sig, handler in self._signals.items():
            signal.signal(sig, handler)
        self._signals.clear()

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        self.log.info("received signal %s", signum)
        self.stop()
