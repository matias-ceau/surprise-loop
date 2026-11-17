"""The tick loop, tick dumps, resume and replay.

One tick is Perceive, Appraise, then pass 0 (Deliberate, Act, Monitor, Evaluate, Reconcile) and,
when Reconcile raises an override and the world is not done, exactly one more pass, offered only
the actions that serve the urgent need. `run` loops ticks and, given a run directory, writes
`config.json` once and one `tick_<n>.json` per tick, so a run can be resumed and replayed from its
files. Every written byte is deterministic.
"""

import asyncio
import dataclasses
import os
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from surprise_loop import phases as real
from surprise_loop.agents import load_model
from surprise_loop.config import RunConfig, Runtime, initial_agent_state, resolve_model
from surprise_loop.phases.heuristics import require
from surprise_loop.phases.perceive import build_percept
from surprise_loop.state import AgentState, PassRecord, TickDump, TickRecord, Workspace
from surprise_loop.world import ClearingWorld, Frozen, WorldState

Phase = Callable[[Workspace, Runtime], Awaitable[Workspace]]

CONFIG_FILE = "config.json"
DUMP_GLOB = "tick_*.json"


class RunError(RuntimeError):
    """A run directory that cannot be started, resumed or read."""


class ReplayMismatch(RunError):
    """A replayed tick that differs from its stored dump; `.tick` is the first differing tick."""

    def __init__(self, run_dir: Path, tick: int, problem: str) -> None:
        super().__init__(f"{run_dir}: replay differs at tick {tick}: {problem}")
        self.run_dir, self.tick, self.problem = run_dir, tick, problem

    def __reduce__(self) -> tuple[type["ReplayMismatch"], tuple[Path, int, str]]:
        # picklable (e.g. across a process pool) despite the custom __init__
        return type(self), (self.run_dir, self.tick, self.problem)


class RunResult(Frozen):
    config: RunConfig
    records: list[TickRecord]
    agent: AgentState
    world: WorldState


@dataclass(frozen=True)
class _Phases:
    """The seven phase functions a tick calls; the real ones by default."""

    perceive: Phase = real.perceive
    appraise: Phase = real.appraise
    deliberate: Phase = real.deliberate
    act: Phase = real.act
    monitor: Phase = real.monitor
    evaluate: Phase = real.evaluate
    reconcile: Phase = real.reconcile


_REAL_PHASES = _Phases()


# --- tick ---------------------------------------------------------------------------------------


async def tick(agent: AgentState, rt: Runtime) -> tuple[AgentState, TickRecord]:
    """Run one tick (`agent.tick + 1`) and return the updated agent and the tick's record."""
    return await _tick(agent, rt, _REAL_PHASES)


async def _pass(ws: Workspace, rt: Runtime, ph: _Phases) -> Workspace:
    for phase in (ph.deliberate, ph.act, ph.monitor, ph.evaluate, ph.reconcile):
        ws = await phase(ws, rt)
    return ws


async def _tick(agent: AgentState, rt: Runtime, ph: _Phases) -> tuple[AgentState, TickRecord]:
    # During the tick `agent.tick` stays at the completed count, so the phases read the number of
    # the tick in progress as `agent.tick + 1`; it is advanced only after the last pass.
    number = agent.tick + 1
    observation = rt.world.observe()
    ws = Workspace(agent=agent, observation=observation)
    ws = await ph.perceive(ws, rt)
    ws = await ph.appraise(ws, rt)
    appraisal = require(ws.appraisal, "appraisal", "tick")

    ws = await _pass(ws, rt, ph)
    if ws.override is not None and not rt.world.done:
        outcome = require(ws.outcome, "outcome", "tick")
        params = rt.config.params
        ws = Workspace(
            agent=ws.agent,
            observation=outcome.observation,
            pass_index=1,
            percept=build_percept(
                outcome.observation, ws.agent, params, offers=ws.override.actions
            ),
            appraisal=appraisal,
            override=ws.override,
            passes=ws.passes,
        )
        ws = await _pass(ws, rt, ph)

    record = TickRecord(
        tick=number, observation=observation, appraisal=appraisal, passes=list(ws.passes)
    )
    return ws.agent.model_copy(update={"tick": number}), record


# --- files --------------------------------------------------------------------------------------


def write_atomic(path: Path, text: str) -> None:
    """Write `text` (UTF-8, newlines as given) to a temp file beside `path`, then rename it into
    place, so `path` holds either its old content or all of `text`, never part of it."""
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text, encoding="utf-8", newline="")
    # No fsync: after a power loss a renamed file may still be empty or partial. That is loud,
    # not silent: load_dumps or read_config then refuses the file, and the run is restarted.
    os.replace(temp, path)


def _write_run_file(path: Path, text: str) -> None:
    """`write_atomic` for a file of a run (a tick dump, config.json): an `OSError` becomes a
    `RunError` naming the path, so the command reports it in one line."""
    try:
        write_atomic(path, text)
    except OSError as e:
        raise RunError(f"{path}: cannot write: {e.strerror or e}") from None


def _dump_name(tick: int) -> str:
    return f"tick_{tick:06d}.json"


def _dump(config: RunConfig, agent: AgentState, world: ClearingWorld, rec: TickRecord) -> TickDump:
    """The dump of the tick `rec` records, after it: the agent and the world as they now are."""
    return TickDump(
        run_id=config.run_id, tick=rec.tick, agent=agent, world=world.snapshot(), record=rec
    )


def write_dump(run_dir: Path, dump: TickDump) -> Path:
    """Write `dump` to `run_dir/tick_<tick>.json` atomically and return its path."""
    path = Path(run_dir) / _dump_name(dump.tick)
    _write_run_file(path, dump.model_dump_json())
    return path


def load_dumps(run_dir: Path) -> list[TickDump]:
    """Every `tick_*.json` in `run_dir`, sorted by tick; their ticks must be exactly 1..n."""
    run_dir = Path(run_dir)
    dumps = []
    for path in run_dir.glob(DUMP_GLOB):
        try:
            dump = TickDump.model_validate_json(path.read_bytes())
        except (OSError, ValidationError) as e:
            raise RunError(f"{run_dir}: cannot read {path.name}: {_reason(e)}") from None
        if path.name != _dump_name(dump.tick):
            raise RunError(f"{run_dir}: {path.name} holds tick {dump.tick}")
        dumps.append(dump)
    dumps.sort(key=lambda d: d.tick)
    ticks = [d.tick for d in dumps]
    if ticks != list(range(1, len(dumps) + 1)):
        missing = sorted(set(range(1, max(ticks) + 1)) - set(ticks))
        raise RunError(f"{run_dir}: tick dumps are not 1..{max(ticks)}: missing tick {missing[0]}")
    return dumps


def _reason(e: Exception) -> str:
    if isinstance(e, ValidationError):
        return f"invalid tick dump ({e.error_count()} errors)"
    return getattr(e, "strerror", None) or type(e).__name__


def read_config(run_dir: Path) -> RunConfig:
    """The `RunConfig` stored in `run_dir/config.json`."""
    run_dir = Path(run_dir)
    path = run_dir / CONFIG_FILE
    if not run_dir.is_dir():
        raise RunError(f"{run_dir}: no such run directory")
    if not path.is_file():
        raise RunError(f"{run_dir}: {CONFIG_FILE} is missing")
    try:
        return RunConfig.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as e:
        problem = "invalid config" if isinstance(e, ValidationError) else _reason(e)
        raise RunError(f"{run_dir}: cannot read {CONFIG_FILE}: {problem}") from None


def _initial(config: RunConfig) -> tuple[AgentState, ClearingWorld]:
    world = ClearingWorld(config.world, config.seed, config.agent.body, config.start_place)
    return initial_agent_state(config), world


def load_result(run_dir: Path) -> RunResult:
    """The `RunResult` of the run stored in `run_dir`, as far as its dumps go."""
    run_dir = Path(run_dir)
    config = read_config(run_dir)
    dumps = load_dumps(run_dir)
    if dumps:
        agent, world = dumps[-1].agent, dumps[-1].world
    else:
        agent, initial_world = _initial(config)
        world = initial_world.snapshot()
    return RunResult(config=config, records=[d.record for d in dumps], agent=agent, world=world)


def has_run_files(run_dir: Path) -> bool:
    """Whether `run_dir` holds anything but a leftover config.json.tmp (a run killed before its
    config.json was in place, which left no tick either): such a directory starts afresh."""
    if not run_dir.is_dir():
        return False
    return any(p.name != CONFIG_FILE + ".tmp" for p in run_dir.iterdir())


def _finished(config: RunConfig, ticks_done: int, world_done: bool) -> bool:
    """Whether a run has nothing left to do: the day has ended or the tick cap is reached."""
    return world_done or (config.ticks is not None and ticks_done >= config.ticks)


def is_complete(result: RunResult) -> bool:
    """Whether `run` would add no tick to `result`: the same stop rule `run` uses."""
    world_done = ClearingWorld.restore(result.config.world, result.world).done
    return _finished(result.config, len(result.records), world_done)


# --- run ----------------------------------------------------------------------------------------


async def run(
    config: RunConfig,
    run_dir: Path | None = None,
    *,
    resume: bool = False,
    stop_after_tick: int | None = None,
) -> RunResult:
    """Run ticks until the world is done, `config.ticks` are complete or `stop_after_tick`."""
    return await _run(
        config, run_dir, resume=resume, stop_after_tick=stop_after_tick, phases=_REAL_PHASES
    )


async def _run(
    config: RunConfig,
    run_dir: Path | None,
    *,
    resume: bool = False,
    stop_after_tick: int | None = None,
    phases: _Phases,
) -> RunResult:
    config = config.model_copy(update={"model": resolve_model(config)})
    model = load_model(config.model)
    agent, world = _initial(config)
    records: list[TickRecord] = []

    if run_dir is not None:
        run_dir = Path(run_dir)
        if run_dir.exists() and not run_dir.is_dir():
            raise RunError(f"{run_dir}: not a directory")
        if has_run_files(run_dir):
            if not resume:
                raise RunError(
                    f"{run_dir}: directory is not empty; resume the run to continue it (--resume)"
                )
            stored = read_config(run_dir)
            if stored != config:
                raise RunError(f"{run_dir}: {CONFIG_FILE} differs from the given config")
            dumps = load_dumps(run_dir)
            if dumps and (
                dumps[-1].run_id != config.run_id
                or dumps[-1].agent.temperament != config.agent.temperament
            ):
                raise RunError(f"{run_dir}: the tick dumps do not belong to {CONFIG_FILE}")
            if dumps:
                records = [d.record for d in dumps]
                agent = dumps[-1].agent
                world = ClearingWorld.restore(config.world, dumps[-1].world)
        else:
            try:
                run_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise RunError(f"{run_dir}: cannot create: {_reason(e)}") from None
            _write_run_file(run_dir / CONFIG_FILE, config.model_dump_json())

    rt = Runtime(config=config, world=world, model=model)
    async with AsyncExitStack() as stack:
        if model is not None:
            # keep the model's HTTP client open for the whole run instead of one per call
            await stack.enter_async_context(model)
        while not (
            _finished(config, agent.tick, world.done)
            or (stop_after_tick is not None and agent.tick >= stop_after_tick)
        ):
            # let concurrent runs in the same event loop take turns, even when no phase awaits I/O
            await asyncio.sleep(0)
            agent, record = await _tick(agent, rt, phases)
            records.append(record)
            if run_dir is not None:
                write_dump(run_dir, _dump(config, agent, world, record))

    return RunResult(config=config, records=records, agent=agent, world=world.snapshot())


# --- replay -------------------------------------------------------------------------------------


async def replay(run_dir: Path) -> int:
    """Re-run every stored tick with the recorded model-phase outputs; return the ticks verified."""
    return await _replay(run_dir, _REAL_PHASES)


def _recorded(record: TickRecord, run_dir: Path, base: _Phases) -> _Phases:
    """`base` with Appraise and Deliberate returning what `record` holds. Evaluate is code and
    runs for real."""

    def recorded_pass(ws: Workspace) -> PassRecord:
        if ws.pass_index >= len(record.passes):
            raise ReplayMismatch(
                run_dir, record.tick, f"the replay needs pass {ws.pass_index}, not in the record"
            )
        return record.passes[ws.pass_index]

    async def appraise(ws: Workspace, rt: Runtime) -> Workspace:
        return ws.model_copy(update={"appraisal": record.appraisal})

    async def deliberate(ws: Workspace, rt: Runtime) -> Workspace:
        return ws.model_copy(update={"decision": recorded_pass(ws).decision})

    return dataclasses.replace(base, appraise=appraise, deliberate=deliberate)


async def _replay(run_dir: Path, base: _Phases) -> int:
    run_dir = Path(run_dir)
    config = read_config(run_dir)
    dumps = load_dumps(run_dir)
    agent, world = _initial(config)
    # No model: the model-backed phases return their recorded outputs, so replay makes no calls.
    rt = Runtime(config=config, world=world, model=None)
    for stored in dumps:
        if world.done:
            raise ReplayMismatch(run_dir, stored.tick, "the world ended before this tick")
        agent, record = await _tick(agent, rt, _recorded(stored.record, run_dir, base))
        rebuilt = _dump(config, agent, world, record)
        if rebuilt != stored:
            differing = [
                f for f in TickDump.model_fields if getattr(rebuilt, f) != getattr(stored, f)
            ]
            raise ReplayMismatch(run_dir, stored.tick, f"mismatch in {', '.join(differing)}")
    return len(dumps)
