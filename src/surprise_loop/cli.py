"""Command line: run one agent or a whole experiment, print a run's metrics, replay a run.

`run` and `metrics` print the run's metrics as JSON on stdout and nothing else, so the output can
be piped; anything informational goes to stderr. A bad config or run directory is reported as one
`error: ...` line on stderr with exit code 2; a replay that differs from its dumps exits 1.

`experiment` runs every (agent, seed) of a config into `DIR/<run_id>/` and writes results.csv,
phases.csv, pe_trace.csv and stability.csv into `DIR/<experiment name>/`. Runs share one event
loop, at most `--jobs` at a time; each run resolves its own model and keeps its HTTP client open
for its own duration only. Before any run starts, every existing run directory must hold the same
config as the experiment (with the model resolved), otherwise nothing runs (exit 2). Then each run
is one of:

- complete: its dumps reach the end of the schedule or the tick cap (`loop.is_complete`, the stop
  rule `run` itself uses). It is loaded, not rerun, and its files are not touched;
- partial: some dumps, not complete. It is resumed from its last dump;
- fresh: no directory, or an empty one.

A run that fails (an LLM error, say) does not stop the others. When all have ended, each failed run
is reported as one `error: <run_id> failed: ...` line, no CSV is written (a table missing some runs
could pass for a whole one) and the exit code is 1. Rerunning the same command resumes the failed
runs from their last dump and skips the complete ones. One progress line per run goes to stderr.
"""

import argparse
import asyncio
import sys
from pathlib import Path

import pydantic_ai

from surprise_loop.config import ConfigError, RunConfig, load_config, resolve_model
from surprise_loop.loop import (
    ReplayMismatch,
    RunError,
    RunResult,
    is_complete,
    load_result,
    read_config,
    replay,
    run,
)
from surprise_loop.metrics import experiment_tables, summarize


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


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

    exp_cmd = commands.add_parser(
        "experiment", help="run every agent and seed of a config, write the CSV tables"
    )
    exp_cmd.add_argument("config", type=Path, help="experiment YAML file")
    exp_cmd.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="parent of the run directories and the experiment's CSV directory (default: runs)",
    )
    exp_cmd.add_argument(
        "--jobs", type=_positive, default=1, help="runs in flight at once (default: 1)"
    )

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


class ExperimentFailed(RuntimeError):
    """Some runs of an experiment failed; `.lines` holds one message per failed run."""

    def __init__(self, lines: list[str], total: int) -> None:
        super().__init__(f"{len(lines)} of {total} runs failed")
        self.lines = lines


def _one_line(e: BaseException) -> str:
    """`e` and its notes on one line: its type, then each non-empty line joined with '; '."""
    text = "\n".join([str(e), *getattr(e, "__notes__", [])])
    message = "; ".join(line.strip() for line in text.splitlines() if line.strip())
    return f"{type(e).__name__}: {message}" if message else type(e).__name__


def _plan(configs: list[RunConfig], runs_dir: Path) -> list[RunConfig]:
    """The configs with their model resolved, after checking every existing run directory.

    Raises ConfigError or RunError, before any run starts, for an LLM backend without a model or
    a run directory whose stored config differs.
    """
    resolved = [c.model_copy(update={"model": resolve_model(c)}) for c in configs]
    for config in resolved:
        run_dir = runs_dir / config.run_id
        if run_dir.is_dir() and any(run_dir.iterdir()) and read_config(run_dir) != config:
            raise RunError(f"{run_dir}: config.json differs from the experiment's config")
    return resolved


async def _one(config: RunConfig, runs_dir: Path) -> tuple[RunResult, str]:
    """Run, resume or load one run; return its result and what was done."""
    run_dir = runs_dir / config.run_id
    if run_dir.is_dir() and any(run_dir.iterdir()):
        stored = load_result(run_dir)
        if is_complete(stored):
            return stored, "complete, not rerun"
        result = await run(config, run_dir, resume=True)
        return result, f"resumed at tick {len(stored.records)}"
    return await run(config, run_dir), "done"


async def _experiment(args: argparse.Namespace) -> None:
    experiment = load_config(args.config)
    configs = _plan(experiment.runs(), args.runs_dir)
    gate = asyncio.Semaphore(args.jobs)
    finished = 0

    async def one(config: RunConfig) -> RunResult | BaseException:
        nonlocal finished
        async with gate:
            try:
                result, what = await _one(config, args.runs_dir)
            except Exception as e:
                outcome: RunResult | BaseException = e
                what = "failed"
            else:
                outcome = result
                what += f", {len(result.records)} ticks"
        finished += 1
        print(f"[{finished}/{len(configs)}] {config.run_id}: {what}", file=sys.stderr)
        return outcome

    outcomes = await asyncio.gather(*(one(c) for c in configs))
    failed = [
        f"{c.run_id} failed: {_one_line(o)}"
        for c, o in zip(configs, outcomes, strict=True)
        if isinstance(o, BaseException)
    ]
    if failed:
        raise ExperimentFailed(failed, len(configs))

    results = [o for o in outcomes if isinstance(o, RunResult)]
    out_dir = args.runs_dir / experiment.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in experiment_tables(results).items():
        (out_dir / name).write_text(text, encoding="utf-8", newline="")
    print(f"tables: {out_dir}", file=sys.stderr)


async def _metrics(args: argparse.Namespace) -> None:
    _print_metrics(load_result(args.run_dir))


async def _replay(args: argparse.Namespace) -> None:
    ticks = await replay(args.run_dir)
    print(f"replay of {args.run_dir}: {ticks} ticks verified")


_COMMANDS = {"run": _run, "experiment": _experiment, "metrics": _metrics, "replay": _replay}


def _error(e: Exception) -> None:
    """Print `e` as a single `error: ...` line, joining a multi-line message with '; '."""
    message = "; ".join(line.strip() for line in str(e).splitlines() if line.strip())
    print(f"error: {message}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # The CLI owns its output: no pydantic-ai first-run banner on stderr (pydantic_ai.BANNER_ENABLED
    # is its documented switch for programs; PYDANTIC_AI_NO_BANNER is the environment equivalent).
    pydantic_ai.BANNER_ENABLED = False
    try:
        asyncio.run(_COMMANDS[args.command](args))
    except ReplayMismatch as e:
        _error(e)
        return 1
    except ExperimentFailed as e:
        for line in e.lines:
            print(f"error: {line}", file=sys.stderr)
        print(f"{e}; no tables written; rerun the same command to resume them", file=sys.stderr)
        return 1
    except (ConfigError, RunError) as e:
        _error(e)
        return 2
    return 0
