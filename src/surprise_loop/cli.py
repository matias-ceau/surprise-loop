"""Command line.

`lens-page` writes the lens page of a config's two bodies as one self-contained HTML file. Anything
informational goes to stderr. A bad config, or a file that cannot be written, is reported as one
`error: ...` line on stderr with exit code 2.
"""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from surprise_loop.config import ConfigError, load_config
from surprise_loop.viewer.lens_page import lens_page_html


class CommandError(Exception):
    """A command that cannot do its work for a reason outside the config (a path, say)."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="surprise-loop", description="Run and inspect surprise-loop experiments."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    page_cmd = commands.add_parser(
        "lens-page", help="write the lens page of a config's two bodies as one HTML file"
    )
    page_cmd.add_argument("config", type=Path, help="experiment YAML file with two agents")
    page_cmd.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("lens_page.html"),
        help="the HTML file to write (default: lens_page.html)",
    )
    return parser


def _lens_page(args: argparse.Namespace) -> None:
    html = lens_page_html(load_config(args.config))
    try:
        args.output.write_text(html, encoding="utf-8")
    except OSError as e:
        raise CommandError(f"{args.output}: cannot write: {e.strerror}") from None
    print(f"wrote {args.output}", file=sys.stderr)


_COMMANDS: dict[str, Callable[[argparse.Namespace], None]] = {"lens-page": _lens_page}


def _error(e: Exception) -> None:
    """Print `e` as a single `error: ...` line, joining a multi-line message with '; '."""
    message = "; ".join(line.strip() for line in str(e).splitlines() if line.strip())
    print(f"error: {message}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _COMMANDS[args.command](args)
    except (ConfigError, CommandError) as e:
        _error(e)
        return 2
    return 0
