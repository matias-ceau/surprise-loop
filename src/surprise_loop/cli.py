"""Command line: run one body or a whole experiment, print a run's metrics, replay a run, and the
lens page and the ladder check.

`run` and `metrics` print the run's metrics as JSON on stdout and nothing else, so the output can
be piped; anything informational goes to stderr. `replay` re-runs a stored run from its files and
exits 1 when it differs from them.

`experiment` runs every (agent, seed) of a config into `DIR/<run_id>/` and writes results.csv,
trace.csv and (with at least two seeds) stability.csv into `DIR/<experiment name>/`. Runs share
one event loop, at most `--jobs` at a time; each run resolves its own model and keeps its HTTP
client open for its own duration only. Before any run starts, every existing run directory must
hold the same config as the experiment (with the model resolved), and every model a run still
needs must load, otherwise nothing runs (exit 2). A complete run needs no model: when none is set,
the one in its config.json is used. Then each run is one of:

- complete: its dumps reach the end of the day or the tick cap (`loop.is_complete`, the stop rule
  `run` itself uses). It is loaded, not rerun, and its files are not touched;
- partial: some dumps, not complete. It is resumed from its last dump;
- fresh: no directory, or an empty one.

A run that fails does not stop the others. When all have ended, each failed run is reported as
one `error: <run_id> failed: ...` line, no CSV is written and any CSVs an earlier invocation left
in `DIR/<experiment name>/` are removed (a table missing some runs could pass for a whole one),
and the exit code is 1. The CSVs are written atomically. Rerunning the same command resumes the
failed runs from their last dump and skips the complete ones. One progress line per run goes to
stderr.

`lens-page` writes the lens page of a config's two bodies as one self-contained HTML file.
`scene` writes the film of two stored runs of the same day (same seed, world config and number of
world steps, else an error) as one self-contained HTML file.
`ladder-check` runs the ladder check (see `ladder.py`) on the configured model, prints one table
per scene and exits 0 only if both scenes pass, 1 when the check ran to completion and a scene
failed. Anything informational goes to stderr. A bad config, or a file that cannot be read or
written, or a run directory that cannot be started, resumed or read, is reported as one
`error: ...` line on stderr with exit code 2; so is a ladder check left with a cell short of
answers (no verdict then).
"""

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pydantic_ai

from surprise_loop.agents import load_model
from surprise_loop.config import (
    MODEL_ENV_VAR,
    ConfigError,
    Params,
    RunConfig,
    load_config,
    resolve_model,
)
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
from surprise_loop.loop import (
    ReplayMismatch,
    RunError,
    RunResult,
    has_run_files,
    is_complete,
    load_result,
    replay,
    run,
    write_atomic,
)
from surprise_loop.metrics import TABLE_FILES, experiment_tables
from surprise_loop.metrics import summarize as summarize_run
from surprise_loop.viewer.lens_page import lens_page_html
from surprise_loop.viewer.scene import scene_html


class CommandError(Exception):
    """A command that cannot do its work for a reason outside the config (a path, say)."""


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

    scene_cmd = commands.add_parser(
        "scene", help="write the film of two stored runs of the same day as one HTML file"
    )
    scene_cmd.add_argument("run_a", type=Path, help="run directory shown on the left")
    scene_cmd.add_argument("run_b", type=Path, help="run directory shown on the right")
    scene_cmd.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("scene.html"),
        help="the HTML file to write (default: scene.html)",
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


def _scene(args: argparse.Namespace) -> int:
    html = scene_html(load_result(args.run_a), load_result(args.run_b))
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


# --- run, experiment, metrics, replay -----------------------------------------------------------


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
    print(summarize_run(result).model_dump_json(indent=2))


def _check_runs_dir(runs_dir: Path) -> None:
    if runs_dir.exists() and not runs_dir.is_dir():
        raise RunError(f"{runs_dir}: --runs-dir is not a directory")


def _run(args: argparse.Namespace) -> int:
    config = _select(args.config, args.agent, args.seed)
    _check_runs_dir(args.runs_dir)
    run_dir = args.runs_dir / config.run_id
    result = asyncio.run(run(config, run_dir, resume=args.resume))
    print(f"run directory: {run_dir}", file=sys.stderr)
    _print_metrics(result)
    return 0


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


@dataclass(frozen=True)
class _Planned:
    """One run of an experiment: its config (model resolved) and what is already on disk."""

    config: RunConfig
    stored: RunResult | None  # None: no run directory, or an empty one

    @property
    def complete(self) -> bool:
        return self.stored is not None and is_complete(self.stored)


async def _plan(configs: list[RunConfig], runs_dir: Path) -> list[_Planned]:
    """Every run with its model resolved and its existing directory read, checked before any run.

    Raises ConfigError or RunError, before any run starts, for an LLM backend without a model, a
    model that cannot be loaded, or a run directory whose stored config differs. A complete run
    is only loaded, so it needs no model: when none is set, the model stored in its config.json
    stands in.
    """
    planned = []
    for config in configs:
        run_dir = runs_dir / config.run_id
        stored = load_result(run_dir) if has_run_files(run_dir) else None
        if stored is not None and is_complete(stored):
            try:
                model = resolve_model(config)
            except ConfigError:
                model = stored.config.model
        else:
            model = resolve_model(config)
        config = config.model_copy(update={"model": model})
        if stored is not None and stored.config != config:
            raise RunError(f"{run_dir}: config.json differs from the experiment's config")
        planned.append(_Planned(config=config, stored=stored))
    # Load each model the runs will use once, so a bad one fails here rather than in every run;
    # close it again at once (each run loads and holds its own).
    for name in sorted({p.config.model for p in planned if not p.complete} - {None}):
        model = load_model(name)
        if model is not None:
            async with model:
                pass
    return planned


async def _one(planned: _Planned, runs_dir: Path) -> tuple[RunResult, str]:
    """Run, resume or load one run; return its result and what was done."""
    config, stored = planned.config, planned.stored
    run_dir = runs_dir / config.run_id
    if stored is not None:
        if planned.complete:
            return stored, "complete, not rerun"
        result = await run(config, run_dir, resume=True)
        return result, f"resumed at tick {len(stored.records)}"
    return await run(config, run_dir), "done"


async def _run_experiment(
    configs: list[RunConfig], runs_dir: Path, jobs: int
) -> tuple[list[_Planned], list[RunResult | BaseException]]:
    """Plan every run, then run them in this event loop, at most `jobs` at a time. Each outcome is
    a RunResult or the exception that ended the run."""
    plan = await _plan(configs, runs_dir)
    gate = asyncio.Semaphore(jobs)
    finished = 0

    async def one(planned: _Planned) -> RunResult | BaseException:
        nonlocal finished
        async with gate:
            try:
                result, what = await _one(planned, runs_dir)
            except Exception as e:
                outcome: RunResult | BaseException = e
                what = "failed"
            else:
                outcome = result
                what += f", {len(result.records)} ticks"
        finished += 1
        print(f"[{finished}/{len(plan)}] {planned.config.run_id}: {what}", file=sys.stderr)
        return outcome

    return plan, list(await asyncio.gather(*(one(p) for p in plan)))


def _experiment(args: argparse.Namespace) -> int:
    experiment = load_config(args.config)
    _check_runs_dir(args.runs_dir)
    plan, outcomes = asyncio.run(_run_experiment(experiment.runs(), args.runs_dir, args.jobs))
    failed = [
        f"{p.config.run_id} failed: {_one_line(o)}"
        for p, o in zip(plan, outcomes, strict=True)
        if isinstance(o, BaseException)
    ]
    out_dir = args.runs_dir / experiment.name
    if failed:
        # tables from an earlier invocation would no longer match the run directories
        if out_dir.is_dir():
            _write_tables(out_dir, {})
        raise ExperimentFailed(failed, len(plan))

    results = [o for o in outcomes if isinstance(o, RunResult)]
    _write_tables(out_dir, experiment_tables(results), create=True)
    print(f"tables: {out_dir}", file=sys.stderr)
    return 0


def _write_tables(out_dir: Path, tables: dict[str, str], *, create: bool = False) -> None:
    """Write `tables` into `out_dir` atomically and remove every other table an earlier
    invocation may have left there (e.g. a stability.csv from a run with more seeds)."""
    try:
        if create:
            out_dir.mkdir(parents=True, exist_ok=True)
        for name in TABLE_FILES:
            if name in tables:
                write_atomic(out_dir / name, tables[name])
            else:
                (out_dir / name).unlink(missing_ok=True)
    except OSError as e:
        raise CommandError(f"{out_dir}: cannot write: {e.strerror or e}") from None


def _metrics(args: argparse.Namespace) -> int:
    _print_metrics(load_result(args.run_dir))
    return 0


def _replay(args: argparse.Namespace) -> int:
    ticks = asyncio.run(replay(args.run_dir))
    print(f"replay of {args.run_dir}: {ticks} ticks verified")
    return 0


_COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "run": _run,
    "experiment": _experiment,
    "metrics": _metrics,
    "replay": _replay,
    "lens-page": _lens_page,
    "scene": _scene,
    "ladder-check": _ladder_check,
}


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
        return _COMMANDS[args.command](args)
    except ReplayMismatch as e:
        _error(e)
        return 1
    except ExperimentFailed as e:
        for line in e.lines:
            print(f"error: {line}", file=sys.stderr)
        print(f"{e}; no tables written; rerun the same command to resume them", file=sys.stderr)
        return 1
    except (ConfigError, CommandError, RunError) as e:
        _error(e)
        return 2
