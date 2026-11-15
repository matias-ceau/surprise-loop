"""Command line: run one agent of an experiment config, print a run's metrics, replay a run.

`run` and `metrics` print the run's metrics as JSON on stdout and nothing else, so the output can
be piped; anything informational goes to stderr. A bad config or run directory is reported as one
`error: ...` line on stderr with exit code 2; a replay that differs from its dumps exits 1.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from surprise_loop.config import ConfigError, RunConfig, load_config
from surprise_loop.loop import ReplayMismatch, RunError, RunResult, load_result, replay, run
from surprise_loop.metrics import summarize


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="surprise-loop", description="Run and inspect surprise-loop experiments."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run_cmd = commands.add_parser("run", help="run one agent and seed of a config, print metrics")
    run_cmd.add_argument("config", type=Path, help="experiment YAML file")
    run_cmd.add_argument("--agent", help="agent name (default: the first agent in the file)")
    run_cmd.add_argument(
        "--seed", type=int, help="one of the file's seeds (default: the first seed)"
    )
    run_cmd.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="parent of the run directory (default: runs)",
    )
    run_cmd.add_argument("--resume", action="store_true", help="continue an existing run directory")

    metrics_cmd = commands.add_parser("metrics", help="print the metrics of a stored run")
    metrics_cmd.add_argument("run_dir", type=Path)

    replay_cmd = commands.add_parser("replay", help="verify a stored run by replaying its ticks")
    replay_cmd.add_argument("run_dir", type=Path)
    return parser


def _select(path: Path, agent: str | None, seed: int | None) -> RunConfig:
    """The run of `agent` and `seed` in the config at `path`; their defaults are the first ones."""
    experiment = load_config(path)
    agent = experiment.agents[0].name if agent is None else agent
    seed = experiment.seeds[0] if seed is None else seed
    names = [a.name for a in experiment.agents]
    if agent not in names:
        raise ConfigError(f"{path}: no agent named {agent!r}; agents: {', '.join(names)}")
    if seed not in experiment.seeds:
        seeds = ", ".join(map(str, experiment.seeds))
        raise ConfigError(f"{path}: seed {seed} is not in the config; seeds: {seeds}")
    return next(r for r in experiment.runs() if r.agent.name == agent and r.seed == seed)


def _print_metrics(result: RunResult) -> None:
    print(summarize(result).model_dump_json(indent=2))


async def _run(args: argparse.Namespace) -> None:
    config = _select(args.config, args.agent, args.seed)
    run_dir = args.runs_dir / config.run_id
    result = await run(config, run_dir, resume=args.resume)
    print(f"run directory: {run_dir}", file=sys.stderr)
    _print_metrics(result)


async def _metrics(args: argparse.Namespace) -> None:
    _print_metrics(load_result(args.run_dir))


async def _replay(args: argparse.Namespace) -> None:
    ticks = await replay(args.run_dir)
    print(f"replay of {args.run_dir}: {ticks} ticks verified")


_COMMANDS = {"run": _run, "metrics": _metrics, "replay": _replay}


def _error(e: Exception) -> None:
    """Print `e` as a single `error: ...` line, joining a multi-line message with '; '."""
    message = "; ".join(line.strip() for line in str(e).splitlines() if line.strip())
    print(f"error: {message}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        asyncio.run(_COMMANDS[args.command](args))
    except ReplayMismatch as e:
        _error(e)
        return 1
    except (ConfigError, RunError) as e:
        _error(e)
        return 2
    return 0
