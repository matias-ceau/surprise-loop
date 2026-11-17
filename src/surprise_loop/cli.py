"""Command line.

`lens-page` writes the lens page of a config's two bodies as one self-contained HTML file.
`ladder-check` runs the ladder check (see `ladder.py`) on the configured model, prints one table
per scene and exits 0 only if both scenes pass, 1 when the check ran to completion and a scene
failed. Anything informational goes to stderr. A bad config, or a file that cannot be read or
written, is reported as one `error: ...` line on stderr with exit code 2; so is a ladder check
left with a cell short of answers (no verdict then).
"""

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

import pydantic_ai

from surprise_loop.agents import load_model
from surprise_loop.config import MODEL_ENV_VAR, ConfigError, Params, load_config
from surprise_loop.ladder import (
    ANSWERS_FILE,
    SCENES,
    SUMMARY_FILE,
    current_answers,
    ladder_table,
    read_answers,
    run_ladder,
    short_cells,
    summarize,
)
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

    ladder_cmd = commands.add_parser(
        "ladder-check",
        help="check that the lens's heat wording moves a model's choice the right way",
    )
    ladder_cmd.add_argument(
        "--config",
        type=Path,
        help=f"experiment YAML file: its model (else ${MODEL_ENV_VAR}) and its params",
    )
    ladder_cmd.add_argument(
        "--out",
        type=Path,
        default=Path("runs/ladder"),
        help="directory for answers.jsonl and summary.json; a rerun asks only for what is "
        "missing (default: runs/ladder)",
    )
    ladder_cmd.add_argument(
        "--samples", type=_positive, help="answers per scene and level (default: from params)"
    )
    ladder_cmd.add_argument(
        "--jobs", type=_positive, help="model calls at once (default: from params)"
    )
    return parser


def _positive(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be a whole number of at least 1: {text!r}")
    return value


def _lens_page(args: argparse.Namespace) -> int:
    html = lens_page_html(load_config(args.config))
    try:
        args.output.write_text(html, encoding="utf-8")
    except OSError as e:
        raise CommandError(f"{args.output}: cannot write: {e.strerror}") from None
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


def _ladder_check(args: argparse.Namespace) -> int:
    params, name = Params(), None
    if args.config is not None:
        config = load_config(args.config)
        params, name = config.params, config.model
    name = name or os.environ.get(MODEL_ENV_VAR)
    if not name:
        raise ConfigError(f"ladder-check needs a model: set config `model` or {MODEL_ENV_VAR}")
    lp = params.ladder
    samples = args.samples or lp.samples
    jobs = args.jobs or lp.jobs
    try:
        done = read_answers(args.out / ANSWERS_FILE)
    except (OSError, ValueError) as e:
        raise CommandError(str(e)) from None
    model = load_model(name)
    assert model is not None  # name is set
    current, stale = current_answers(done, name, samples, params)
    print(f"ladder: {len(current)} current answers in {args.out}; model {name}", file=sys.stderr)
    if stale:
        print(
            f"ladder: ignored {stale} stale answers (another model, or a prompt that has "
            "changed); their slots are asked again",
            file=sys.stderr,
        )
    pydantic_ai.BANNER_ENABLED = False
    try:
        new = asyncio.run(run_ladder(model, samples, jobs, args.out, params, done, name))
    except OSError as e:
        raise CommandError(f"{args.out}: cannot write: {e.strerror}") from None
    print(f"ladder: {len(new)} new answers", file=sys.stderr)

    answers = current_answers(done + new, name, samples, params)[0]
    short = short_cells(answers, samples, lp)
    if short:
        raise CommandError(
            "no verdict, some cells are short of answers (rerun to ask for the rest): "
            + "; ".join(short)
        )

    curves = summarize(answers, lp)
    summary = {
        "model": name,
        "samples": samples,
        "params": params.model_dump(mode="json"),
        "scenes": {scene: curve.model_dump() for scene, curve in curves.items()},
    }
    try:
        (args.out / SUMMARY_FILE).write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    except OSError as e:
        raise CommandError(f"{args.out / SUMMARY_FILE}: cannot write: {e.strerror}") from None
    print("\n\n".join(ladder_table(s, answers, curves[s], lp) for s in SCENES))
    return 0 if all(curve.passed for curve in curves.values()) else 1


_COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "lens-page": _lens_page,
    "ladder-check": _ladder_check,
}


def _error(e: Exception) -> None:
    """Print `e` as a single `error: ...` line, joining a multi-line message with '; '."""
    message = "; ".join(line.strip() for line in str(e).splitlines() if line.strip())
    print(f"error: {message}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _COMMANDS[args.command](args)
    except (ConfigError, CommandError) as e:
        _error(e)
        return 2
