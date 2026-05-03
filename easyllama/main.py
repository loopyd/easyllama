from __future__ import annotations

from .cli import main as cli_main


def main(argv: list[str] | None = None) -> int:
    return cli_main(argv)
