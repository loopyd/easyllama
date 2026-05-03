from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field

from .config import MODE_BASIC, MODE_LUCEBOX, MODE_TURBOQUANT, RUNTIME_CONTAINER, load_settings
from .logger import configure_logging
from .runtime import DockerRuntime, serve, warmup_models
from .servers import defs as server_defs, run as run_server

Handler = Callable[[argparse.Namespace, list[str]], int]
Configurer = Callable[[argparse.ArgumentParser], None]


@dataclass(frozen=True)
class CommandNode:
    name: str
    help: str
    handler: Handler | None = None
    configure_parser: Configurer | None = None
    add_help: bool = True
    children: tuple[CommandNode, ...] = field(default_factory=tuple)


def _server_handler(name: str) -> Handler:
    def handler(args: argparse.Namespace, extra_args: list[str]) -> int:
        return run_server(name, extra_args + list(getattr(args, "server_args", [])))

    return handler


def _server_passthrough_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("server_args", nargs=argparse.REMAINDER)


def _server_nodes() -> tuple[CommandNode, ...]:
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
    if extra_args:
        raise SystemExit(f"unexpected args for build: {' '.join(extra_args)}")
    settings = load_settings(mode_override=args.mode)
    return DockerRuntime(settings).build_image()


def _start_handler(args: argparse.Namespace, extra_args: list[str]) -> int:
    if extra_args:
        raise SystemExit(f"unexpected args for start: {' '.join(extra_args)}")
    settings = load_settings(mode_override=args.mode)
    return DockerRuntime(settings).run_container()


def _warmup_handler(args: argparse.Namespace, extra_args: list[str]) -> int:
    settings = load_settings(mode_override=args.mode)
    return warmup_models(settings, list(args.models) + extra_args)


def _stop_handler(args: argparse.Namespace, extra_args: list[str]) -> int:
    if extra_args:
        raise SystemExit(f"unexpected args for stop: {' '.join(extra_args)}")
    settings = load_settings(mode_override=args.mode)
    return DockerRuntime(settings).stop_container()


def _restart_handler(args: argparse.Namespace, extra_args: list[str]) -> int:
    if extra_args:
        raise SystemExit(f"unexpected args for restart: {' '.join(extra_args)}")
    settings = load_settings(mode_override=args.mode)
    return DockerRuntime(settings).restart_container()


def _logs_handler(args: argparse.Namespace, extra_args: list[str]) -> int:
    if extra_args:
        raise SystemExit(f"unexpected args for logs: {' '.join(extra_args)}")
    settings = load_settings(mode_override=args.mode)
    return DockerRuntime(settings).print_logs()


def _status_handler(args: argparse.Namespace, extra_args: list[str]) -> int:
    if extra_args:
        raise SystemExit(f"unexpected args for status: {' '.join(extra_args)}")
    settings = load_settings(mode_override=args.mode)
    return DockerRuntime(settings).status()


def _clean_handler(args: argparse.Namespace, extra_args: list[str]) -> int:
    if extra_args:
        raise SystemExit(f"unexpected args for clean: {' '.join(extra_args)}")
    settings = load_settings(mode_override=args.mode)
    return DockerRuntime(settings).clean(all_images=args.all_images)


def _serve_handler(args: argparse.Namespace, extra_args: list[str]) -> int:
    if extra_args:
        raise SystemExit(f"unexpected args for serve: {' '.join(extra_args)}")
    settings = load_settings(mode_override=args.mode, runtime_mode_override=RUNTIME_CONTAINER)
    return serve(settings)


def _help_handler(args: argparse.Namespace, extra_args: list[str]) -> int:
    parser = build_parser()
    parser.print_help()
    return 0


def _warmup_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("models", nargs="*")


def _clean_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--all-images", action="store_true")


def command_tree() -> tuple[CommandNode, ...]:
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
        CommandNode(name="logs", help="Follow runtime logs", handler=_logs_handler),
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
    parser = argparse.ArgumentParser(prog="easyllama")
    parser.add_argument("--mode", choices=[MODE_BASIC, MODE_TURBOQUANT, MODE_LUCEBOX], default=None)
    parser.add_argument("--verbosity", choices=["debug", "info", "warning", "error"], default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    add_command_nodes(parser, command_tree())
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, extra_args = parser.parse_known_args(argv)
    configure_logging(verbosity=args.verbosity, quiet=args.quiet, no_color=args.no_color)
    handler = getattr(args, "_handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return handler(args, extra_args)
