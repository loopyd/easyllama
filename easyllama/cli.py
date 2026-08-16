"""Define the command-line interface and dispatch commands."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field

from .config import RUNTIME_CONTAINER, Config
from .helpers.docker import DockerRuntime
from .helpers.logger import LOG as APP_LOG
from .runtime import serve, warmup_models
from .servers import defs as server_defs, mode_names, run as run_server

Handler = Callable[[argparse.Namespace, list[str]], int]
Configurer = Callable[[argparse.ArgumentParser], None]


@dataclass(frozen=True)
class CommandNode:
    """Describe one node in the command hierarchy.

    Attributes:
        name: The name (str).
        help: The help (str).
        handler: The handler (Handler | None).
        configure_parser: The configure parser (Configurer | None).
        add_help: The add help (bool).
        children: The children (tuple[CommandNode, ...])."""

    name: str
    help: str
    handler: Handler | None = None
    configure_parser: Configurer | None = None
    add_help: bool = True
    children: tuple[CommandNode, ...] = field(default_factory=tuple)


def _server_handler(name: str) -> Handler:
    """Handle the server command.

    Args:
        name: The name.

    Returns:
        Handler: The server handler result."""

    def handler(args: argparse.Namespace, extra_args: list[str]) -> int:
        """Perform the handler operation.

        Args:
            args: The args.
            extra_args: The extra args.

        Returns:
            int: The handler result."""
        return run_server(name, extra_args + list(getattr(args, "server_args", [])))

    return handler


def _server_passthrough_config(parser: argparse.ArgumentParser) -> None:
    """Configure arguments for the server passthrough command.

    Args:
        parser: The parser."""
    parser.add_argument("server_args", nargs=argparse.REMAINDER)


def _server_nodes() -> tuple[CommandNode, ...]:
    """Perform the internal server nodes operation.

    Returns:
        tuple[CommandNode, ...]: The server nodes result."""
    return tuple(
        CommandNode(
            name=item.name,
            help=item.help,
            handler=_server_handler(item.name),
            configure_parser=_server_passthrough_config,
            add_help=False,
        )
        for item in server_defs()
    )


def _build_handler(args: argparse.Namespace, extra_args: list[str]) -> int:
    """Handle the build command.

    Args:
        args: The args.
        extra_args: The extra args.

    Returns:
        int: The build handler result.

    Raises:
        SystemExit: If the build handler operation cannot be completed."""
    if extra_args:
        raise SystemExit(f"unexpected args for build: {' '.join(extra_args)}")
    settings = Config(mode_override=args.mode)
    return DockerRuntime(settings).build_image()


def _start_handler(args: argparse.Namespace, extra_args: list[str]) -> int:
    """Handle the start command.

    Args:
        args: The args.
        extra_args: The extra args.

    Returns:
        int: The start handler result.

    Raises:
        SystemExit: If the start handler operation cannot be completed."""
    if extra_args:
        raise SystemExit(f"unexpected args for start: {' '.join(extra_args)}")
    settings = Config(mode_override=args.mode)
    return DockerRuntime(settings).run_container()


def _warmup_handler(args: argparse.Namespace, extra_args: list[str]) -> int:
    """Handle the warmup command.

    Args:
        args: The args.
        extra_args: The extra args.

    Returns:
        int: The warmup handler result."""
    settings = Config(mode_override=args.mode)
    return warmup_models(settings, list(args.models) + extra_args)


def _stop_handler(args: argparse.Namespace, extra_args: list[str]) -> int:
    """Handle the stop command.

    Args:
        args: The args.
        extra_args: The extra args.

    Returns:
        int: The stop handler result.

    Raises:
        SystemExit: If the stop handler operation cannot be completed."""
    if extra_args:
        raise SystemExit(f"unexpected args for stop: {' '.join(extra_args)}")
    settings = Config(mode_override=args.mode)
    return DockerRuntime(settings).stop_container()


def _restart_handler(args: argparse.Namespace, extra_args: list[str]) -> int:
    """Handle the restart command.

    Args:
        args: The args.
        extra_args: The extra args.

    Returns:
        int: The restart handler result.

    Raises:
        SystemExit: If the restart handler operation cannot be completed."""
    if extra_args:
        raise SystemExit(f"unexpected args for restart: {' '.join(extra_args)}")
    settings = Config(mode_override=args.mode)
    return DockerRuntime(settings).restart_container()


def _logs_handler(args: argparse.Namespace, extra_args: list[str]) -> int:
    """Handle the logs command.

    Args:
        args: The args.
        extra_args: The extra args.

    Returns:
        int: The logs handler result.

    Raises:
        SystemExit: If the logs handler operation cannot be completed."""
    if extra_args:
        raise SystemExit(f"unexpected args for logs: {' '.join(extra_args)}")
    settings = Config(mode_override=args.mode)
    return DockerRuntime(settings).print_logs(tail=args.tail)


def _status_handler(args: argparse.Namespace, extra_args: list[str]) -> int:
    """Handle the status command.

    Args:
        args: The args.
        extra_args: The extra args.

    Returns:
        int: The status handler result.

    Raises:
        SystemExit: If the status handler operation cannot be completed."""
    if extra_args:
        raise SystemExit(f"unexpected args for status: {' '.join(extra_args)}")
    settings = Config(mode_override=args.mode)
    return DockerRuntime(settings).status()


def _clean_handler(args: argparse.Namespace, extra_args: list[str]) -> int:
    """Handle the clean command.

    Args:
        args: The args.
        extra_args: The extra args.

    Returns:
        int: The clean handler result.

    Raises:
        SystemExit: If the clean handler operation cannot be completed."""
    if extra_args:
        raise SystemExit(f"unexpected args for clean: {' '.join(extra_args)}")
    settings = Config(mode_override=args.mode)
    return DockerRuntime(settings).clean(all_images=args.all_images)


def _serve_handler(args: argparse.Namespace, extra_args: list[str]) -> int:
    """Handle the serve command.

    Args:
        args: The args.
        extra_args: The extra args.

    Returns:
        int: The serve handler result.

    Raises:
        SystemExit: If the serve handler operation cannot be completed."""
    if extra_args:
        raise SystemExit(f"unexpected args for serve: {' '.join(extra_args)}")
    settings = Config(mode_override=args.mode, runtime_mode_override=RUNTIME_CONTAINER)
    return serve(settings)


def _help_handler(args: argparse.Namespace, extra_args: list[str]) -> int:
    """Handle the help command.

    Args:
        args: The args.
        extra_args: The extra args.

    Returns:
        int: The help handler result."""
    parser = build_parser()
    parser.print_help()
    return 0


def _warmup_config(parser: argparse.ArgumentParser) -> None:
    """Configure arguments for the warmup command.

    Args:
        parser: The parser."""
    parser.add_argument("models", nargs="*")


def _logs_config(parser: argparse.ArgumentParser) -> None:
    """Configure arguments for the logs command.

    Args:
        parser: The parser."""
    parser.add_argument("--tail", type=lambda value: max(1, int(value)), metavar="N")


def _clean_config(parser: argparse.ArgumentParser) -> None:
    """Configure arguments for the clean command.

    Args:
        parser: The parser."""
    parser.add_argument("--all-images", action="store_true")


def command_tree() -> tuple[CommandNode, ...]:
    """Return the command hierarchy exposed by the CLI.

    Returns:
        tuple[CommandNode, ...]: The command tree result."""
    return (
        CommandNode(
            name="build", help="Build the mode-specific Docker image", handler=_build_handler
        ),
        CommandNode(name="start", help="Start the selected mode container", handler=_start_handler),
        CommandNode(
            name="warmup",
            help="Warm one or more configured models through llama-swap",
            handler=_warmup_handler,
            configure_parser=_warmup_config,
        ),
        CommandNode(
            name="stop", help="Stop and remove the runtime container", handler=_stop_handler
        ),
        CommandNode(
            name="restart", help="Restart the selected mode container", handler=_restart_handler
        ),
        CommandNode(
            name="logs",
            help="Follow runtime logs or show the last N lines",
            handler=_logs_handler,
            configure_parser=_logs_config,
        ),
        CommandNode(name="status", help="Show runtime container status", handler=_status_handler),
        CommandNode(
            name="clean",
            help="Remove the runtime container and image",
            handler=_clean_handler,
            configure_parser=_clean_config,
        ),
        CommandNode(
            name="serve",
            help="Run llama-swap directly inside the container",
            handler=_serve_handler,
        ),
        CommandNode(
            name="server",
            help="Run a mode-specific upstream server directly",
            children=_server_nodes(),
        ),
        CommandNode(name="help", help="Show help output", handler=_help_handler),
    )


def add_command_nodes(parser: argparse.ArgumentParser, nodes: tuple[CommandNode, ...]) -> None:
    """Add command nodes recursively to an argument parser.

    Args:
        parser: The parser.
        nodes: The nodes."""
    if not nodes:
        return
    subparsers = parser.add_subparsers(dest=f"subcommand_{id(parser)}")
    for node in nodes:
        child_parser = subparsers.add_parser(
            node.name,
            help=node.help,
            description=node.help,
            add_help=node.add_help,
        )
        if node.configure_parser is not None:
            node.configure_parser(child_parser)
        if node.handler is not None:
            child_parser.set_defaults(_handler=node.handler)
        add_command_nodes(child_parser, node.children)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level command-line parser.

    Returns:
        argparse.ArgumentParser: The build parser result."""
    parser = argparse.ArgumentParser(prog="easyllama")
    parser.add_argument("--mode", choices=mode_names(), default=None)
    parser.add_argument("--verbosity", choices=["debug", "info", "warning", "error"], default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    add_command_nodes(parser, command_tree())
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the selected command.

    Args:
        argv: The argv.

    Returns:
        int: The main result."""
    parser = build_parser()
    args, extra_args = parser.parse_known_args(argv)
    APP_LOG.configure(verbosity=args.verbosity, quiet=args.quiet, no_color=args.no_color)
    handler = getattr(args, "_handler", None)
    if handler is None:
        parser.print_help()
        return 0
    try:
        return handler(args, extra_args)
    except KeyboardInterrupt:
        return 130
