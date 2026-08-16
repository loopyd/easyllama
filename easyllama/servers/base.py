"""Define server metadata and process lifecycle primitives."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
import os
from pathlib import Path
import signal
import subprocess
from typing import TYPE_CHECKING, Any, TypeVar

from ..helpers.logger import LOG as APP_LOG

if TYPE_CHECKING:
    from ..config import Config


@dataclass(frozen=True, slots=True)
class BuildSource:
    """Describe a source repository and its Docker build arguments.

    Attributes:
        label: The label (str).
        repo_attr: The repo attr (str).
        ref_attr: The ref attr (str).
        repo_build_arg: The repo build arg (str).
        ref_build_arg: The ref build arg (str).
        default_repo: The default repo (str | None).
        default_ref: The default ref (str | None)."""

    label: str
    repo_attr: str
    ref_attr: str
    repo_build_arg: str
    ref_build_arg: str
    default_repo: str | None = None
    default_ref: str | None = None

    def values(self, settings: Config) -> tuple[str, str]:
        """Perform the values operation.

        Args:
            settings: The settings.

        Returns:
            tuple[str, str]: The values result."""
        repo = str(getattr(settings, self.repo_attr))
        ref = str(getattr(settings, self.ref_attr))
        repo_overridden = "EASYLLAMA_LLAMA_CPP_REPO" in os.environ
        ref_overridden = "EASYLLAMA_LLAMA_CPP_REF" in os.environ
        if self.repo_attr == "llama_cpp_repo" and not repo_overridden:
            repo = self.default_repo or repo
        if self.ref_attr == "llama_cpp_ref" and not ref_overridden:
            ref = self.default_ref or ref
        return repo, ref

    def summary(self, settings: Config) -> str:
        """Perform the summary operation.

        Args:
            settings: The settings.

        Returns:
            str: The summary result."""
        repo, ref = self.values(settings)
        return f"{self.label}={repo}@{ref}"

    def build_args(self, settings: Config) -> dict[str, str]:
        """Perform the build args operation.

        Args:
            settings: The settings.

        Returns:
            dict[str, str]: The build args result."""
        repo, ref = self.values(settings)
        return {self.repo_build_arg: repo, self.ref_build_arg: ref}


@dataclass(frozen=True, slots=True)
class RuntimeModeMetadata:
    """Describe build and runtime properties for a server mode.

    Attributes:
        mode: The mode (str).
        docker_target: The docker target (str).
        backend: The backend (str).
        build_sources: The build sources (tuple[BuildSource, ...])."""

    mode: str
    docker_target: str
    backend: str = "llamacpp"
    build_sources: tuple[BuildSource, ...] = ()

    def build_summary(self, settings: Config, *, image_name: str, target: str) -> str:
        """Perform the build summary operation.

        Args:
            settings: The settings.
            image_name: The image name.
            target: The target.

        Returns:
            str: The build summary result."""
        details = " ".join(source.summary(settings) for source in self.build_sources)
        suffix = f" {details}" if details else ""
        return (
            f"building {image_name} "
            f"(mode={self.mode} backend={self.backend} target={target}{suffix})"
        )

    def build_args(self, settings: Config) -> dict[str, str]:
        """Perform the build args operation.

        Args:
            settings: The settings.

        Returns:
            dict[str, str]: The build args result."""
        build_args: dict[str, str] = {}
        for source in self.build_sources:
            build_args.update(source.build_args(settings))
        return build_args


@dataclass(slots=True)
class Spec:
    """Describe a server child process invocation.

    Attributes:
        cmd: The cmd (list[str]).
        env: The env (dict[str, str]).
        app: The app (Any).
        host: The host (str).
        port: The port (int).
        data: The data (dict[str, Any])."""

    cmd: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    app: Any = None
    host: str = "0.0.0.0"
    port: int = 0
    data: dict[str, Any] = field(default_factory=dict)


class ServerBase:
    """Define common parsing and child-process lifecycle behavior.

    Attributes:
        runtime_modes: The runtime modes (tuple[RuntimeModeMetadata, ...]).
    log: The log.
    proc: The proc.
    name: The name.
    help: The help.
    _signals: The signals."""

    name = ""
    help = ""
    runtime_modes: tuple[RuntimeModeMetadata, ...] = ()

    def __init__(self) -> None:
        """Initialize the instance."""
        key = self.name or self.__class__.__name__.lower()
        self.log = APP_LOG.get(f"easyllama.server.{key}")
        self.proc: subprocess.Popen[str] | None = None
        self._signals: dict[signal.Signals, Any] = {}

    def parser(self) -> argparse.ArgumentParser:
        """Build the parser for this server mode.

        Returns:
            argparse.ArgumentParser: The parser result."""
        parser = argparse.ArgumentParser(prog=f"easyllama server {self.name}")
        self.add_args(parser)
        return parser

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        """Add mode-specific command-line arguments.

        Args:
            parser: The parser."""
        del parser

    def parse(self, argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
        """Perform the parse operation.

        Args:
            argv: The argv.

        Returns:
            tuple[argparse.Namespace, list[str]]: The parse result."""
        return self.parser().parse_known_args(argv)

    def build(self, args: argparse.Namespace, extra: list[str]) -> Spec:
        """Build the process specification for parsed server arguments.

        Args:
            args: The args.
            extra: The extra.

        Returns:
            Spec: The build result.

        Raises:
            NotImplementedError: If the build operation cannot be completed."""
        raise NotImplementedError

    def warmup(self, spec: Spec) -> None:
        """Log resolved server assets before startup.

        Args:
            spec: The spec."""
        del spec

    def run(self, spec: Spec) -> int:
        """Run a built server process specification.

        Args:
            spec: The spec.

        Returns:
            int: The run result.

        Raises:
            NotImplementedError: If the run operation cannot be completed."""
        raise NotImplementedError

    def stop(self) -> int:
        """Perform the stop operation.

        Returns:
            int: The stop result."""
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
        """Parse arguments and run the selected command.

        Args:
            argv: The argv.

        Returns:
            int: The main result."""
        args, extra = self.parse(argv)
        spec = self.build(args, extra)
        try:
            self.warmup(spec)
            return self.run(spec)
        finally:
            self.stop()

    def proc_env(self, bin_path: Path) -> dict[str, str]:
        """Perform the proc env operation.

        Args:
            bin_path: The bin path.

        Returns:
            dict[str, str]: The proc env result."""
        env = os.environ.copy()
        lib_dir = str(bin_path.resolve().parent)
        current = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = f"{lib_dir}:{current}" if current else lib_dir
        return env

    def run_proc(self, cmd: list[str], *, env: dict[str, str] | None = None) -> int:
        """Run and supervise the configured child process.

        Args:
            cmd: The cmd.
            env: The env.

        Returns:
            int: The run proc result."""
        merged = os.environ.copy()
        if env:
            merged.update(env)
        self.proc = subprocess.Popen[str](cmd, env=merged, start_new_session=True)
        self._set_signals()
        try:
            assert self.proc is not None
            return self.proc.wait()
        finally:
            self._reset_signals()
            self.proc = None

    def _set_signals(self) -> None:
        """Perform the internal set signals operation."""
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._signals[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handle_signal)

    def _reset_signals(self) -> None:
        """Perform the internal reset signals operation."""
        for sig, handler in self._signals.items():
            signal.signal(sig, handler)
        self._signals.clear()

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        """Perform the internal handle signal operation.

        Args:
            signum: The signum.
            _frame: The frame."""
        self.log.info("received signal %s", signum)
        proc = self.proc
        if proc is not None and proc.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)


ServerClass = TypeVar("ServerClass", bound=type[ServerBase])


def server_metadata(
    *,
    name: str,
    help: str,
    runtime_modes: tuple[RuntimeModeMetadata, ...] = (),
) -> Callable[[ServerClass], ServerClass]:
    """Perform the server metadata operation.

    Args:
        name: The name.
        help: The help.
        runtime_modes: The runtime modes.

    Returns:
        Callable[[ServerClass], ServerClass]: The server metadata result."""

    def decorate(server_cls: ServerClass) -> ServerClass:
        """Perform the decorate operation.

        Args:
            server_cls: The server cls.

        Returns:
            ServerClass: The decorate result."""
        server_cls.name = name
        server_cls.help = help
        server_cls.runtime_modes = runtime_modes
        return server_cls

    return decorate
