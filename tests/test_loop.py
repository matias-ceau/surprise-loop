import asyncio
import dataclasses
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from surprise_loop.config import (
    MODEL_ENV_VAR,
    AgentSpec,
    Backends,
    ConfigError,
    ConflictParams,
    MemoryParams,
    Params,
    RunConfig,
    Runtime,
    initial_agent_state,
)
from surprise_loop.loop import _Phases, _run, _tick, run, tick
from surprise_loop.phases import deliberate, reconcile
from surprise_loop.phases.perceive import build_percept
from surprise_loop.state import Conflict, GoalName, Mood, Traits, Workspace
from surprise_loop.world import PhaseKind, PhaseSpec, SelfAdminWorld, WorldConfig

Phase = Callable[[Workspace, Runtime], Awaitable[Workspace]]

TRAITS = Traits(reward_sensitivity=1.0, surprise_learning_rate=0.3, mood_inertia=0.5)
PRIORITIES = {GoalName.SEEK_REWARD: 1.0, GoalName.EXPLORE: 0.5, GoalName.CONSERVE: 0.5}
# Conflicts disabled so the real Reconcile never asks for a second pass, whatever its rule.
PARAMS = Params(conflict=ConflictParams(enabled=False), memory=MemoryParams(retrieval_k=50))


def schedule(*steps: int) -> list[PhaseSpec]:
    """FR, then PR, then extinction phases with the given step counts (cycling)."""
    kinds = [
        lambda n: PhaseSpec(kind=PhaseKind.FIXED_RATIO, steps=n, ratio=2),
        lambda n: PhaseSpec(kind=PhaseKind.PROGRESSIVE_RATIO, steps=n, pr_start=1, pr_increment=1),
        lambda n: PhaseSpec(kind=PhaseKind.EXTINCTION, steps=n),
    ]
    return [kinds[i % 3](n) for i, n in enumerate(steps)]


def make_config(
    *steps: int,
    ticks: int | None = None,
    seed: int = 3,
    backends: Backends | None = None,
    model: str | None = None,
) -> RunConfig:
    return RunConfig(
        run_id="loop-test",
        seed=seed,
        ticks=ticks,
        world=WorldConfig(active_lever="random", reward_magnitude=1.0, schedule=schedule(*steps)),
        agent=AgentSpec(
            name="a", traits=TRAITS, mood=Mood(valence=0.0, arousal=0.0), goal_priorities=PRIORITIES
        ),
        params=PARAMS,
        backends=backends or Backends(),
        model=model,
    )


def make_rt(config: RunConfig) -> Runtime:
    return Runtime(config=config, world=SelfAdminWorld(config.world, config.seed))


def conflict_reconcile(passes: tuple[int, ...] = (0,), ticks: set[int] | None = None) -> Phase:
    """The real Reconcile, plus a conflict on the given passes of the given ticks (all if None)."""

    async def reconcile_with_conflict(ws: Workspace, rt: Runtime) -> Workspace:
        ws = await reconcile(ws, rt)
        if ws.pass_index not in passes or (ticks is not None and ws.agent.tick + 1 not in ticks):
            return ws
        assert ws.decision is not None
        conflict = Conflict(kind="habit_vs_goal", action=ws.decision.action, detail="injected")
        last = ws.passes[-1].model_copy(update={"conflict": conflict})
        return ws.model_copy(update={"conflict": conflict, "passes": (*ws.passes[:-1], last)})

    return reconcile_with_conflict


# --- tick ---------------------------------------------------------------------------------------


def test_tick_returns_one_pass_and_advances_tick_by_one() -> None:
    config = make_config(10)
    rt = make_rt(config)
    agent = initial_agent_state(config)
    start = rt.world.observe()

    new_agent, record = asyncio.run(tick(agent, rt))

    assert new_agent.tick == agent.tick + 1 == 1
    assert record.tick == 1
    assert len(record.passes) == 1
    assert record.passes[0].tick == 1
    assert record.passes[0].pass_index == 0
    assert record.observation == start
    assert rt.world.snapshot().step == 1
    assert len(new_agent.memory.episodes) == 1


def test_tick_counts_from_the_agent_tick() -> None:
    config = make_config(10)
    rt = make_rt(config)
    agent = initial_agent_state(config).model_copy(update={"tick": 6})

    new_agent, record = asyncio.run(tick(agent, rt))

    assert new_agent.tick == 7
    assert record.tick == 7
    assert record.passes[0].tick == 7


def test_conflict_on_pass_zero_runs_a_second_pass() -> None:
    config = make_config(10)
    rt = make_rt(config)
    agent = initial_agent_state(config)
    phases = _Phases(reconcile=conflict_reconcile())

    new_agent, record = asyncio.run(_tick(agent, rt, phases))

    assert [p.pass_index for p in record.passes] == [0, 1]
    assert [p.tick for p in record.passes] == [1, 1]
    assert rt.world.snapshot().step == 2
    assert [p.outcome.info.step for p in record.passes] == [1, 2]
    assert new_agent.tick == 1
    assert len(new_agent.memory.episodes) == 2


def test_pass_one_deliberate_sees_the_rebuilt_workspace() -> None:
    config = make_config(10)
    rt = make_rt(config)
    agent = initial_agent_state(config)
    seen: list[Workspace] = []

    async def spy_deliberate(ws: Workspace, rt: Runtime) -> Workspace:
        seen.append(ws)
        return await deliberate(ws, rt)

    phases = _Phases(deliberate=spy_deliberate, reconcile=conflict_reconcile())
    _, record = asyncio.run(_tick(agent, rt, phases))

    assert [ws.pass_index for ws in seen] == [0, 1]
    pass0_ws, pass1_ws = seen
    pass0 = record.passes[0]
    assert pass0_ws.conflict is None
    assert pass1_ws.conflict is not None
    assert pass1_ws.conflict == pass0.conflict
    assert pass1_ws.observation == pass0.outcome.observation
    assert pass1_ws.appraisal == record.appraisal == pass0_ws.appraisal
    assert pass1_ws.decision is None
    assert pass1_ws.outcome is None
    assert pass1_ws.prediction_error is None
    assert pass1_ws.feedback is None
    assert pass1_ws.passes == (pass0,)
    # The percept is rebuilt on the agent after pass 0, so it recalls the pass-0 episode.
    assert pass1_ws.agent.tick == 0
    assert len(pass1_ws.agent.memory.episodes) == 1
    assert pass1_ws.percept == build_percept(
        pass0.outcome.observation, pass1_ws.agent, config.params
    )
    assert pass1_ws.percept is not None
    assert pass1_ws.percept.recalled[-1].step == pass0.outcome.info.step
    assert pass1_ws.percept.recalled[-1].tick == 1
    assert pass0_ws.percept is not None
    assert pass0_ws.percept.recalled == []


def test_conflict_on_the_worlds_last_step_yields_one_pass() -> None:
    config = make_config(1)
    rt = make_rt(config)
    agent = initial_agent_state(config)

    new_agent, record = asyncio.run(_tick(agent, rt, _Phases(reconcile=conflict_reconcile())))

    assert rt.world.done
    assert len(record.passes) == 1
    assert record.passes[0].conflict is not None
    assert new_agent.tick == 1


def test_conflict_on_pass_one_is_ignored() -> None:
    config = make_config(10)
    rt = make_rt(config)
    agent = initial_agent_state(config)
    phases = _Phases(reconcile=conflict_reconcile(passes=(0, 1)))

    new_agent, record = asyncio.run(_tick(agent, rt, phases))

    assert len(record.passes) == 2
    assert record.passes[1].conflict is not None
    assert rt.world.snapshot().step == 2
    assert new_agent.tick == 1


def test_tick_leaves_its_input_agent_untouched() -> None:
    config = make_config(10)
    rt = make_rt(config)
    agent = initial_agent_state(config)
    before = agent.model_copy(deep=True)

    asyncio.run(_tick(agent, rt, _Phases(reconcile=conflict_reconcile())))

    assert agent == before


def test_default_phases_are_the_real_ones() -> None:
    from surprise_loop import phases as real

    defaults = _Phases()
    for field in dataclasses.fields(_Phases):
        assert getattr(defaults, field.name) is getattr(real, field.name)


# --- run ----------------------------------------------------------------------------------------


def test_run_stops_after_config_ticks() -> None:
    result = asyncio.run(run(make_config(100, 100, ticks=100)))

    assert len(result.records) == 100
    assert [r.tick for r in result.records] == list(range(1, 101))
    assert result.agent.tick == 100
    assert result.world.step == sum(len(r.passes) for r in result.records)
    assert result.world.phase_index < 2


def test_run_without_tick_limit_stops_when_the_world_is_done() -> None:
    config = make_config(12, 10, 8)

    result = asyncio.run(run(config))

    assert result.world.phase_index == len(config.world.schedule)
    assert sum(len(r.passes) for r in result.records) == 30
    assert result.agent.tick == len(result.records)


def test_run_with_conflicts_counts_ticks_not_steps() -> None:
    config = make_config(30, ticks=5)

    result = asyncio.run(_run(config, None, phases=_Phases(reconcile=conflict_reconcile())))

    assert len(result.records) == 5
    assert all(len(r.passes) == 2 for r in result.records)
    assert result.world.step == 10
    assert result.agent.tick == 5


def test_run_stop_after_tick_stops_early() -> None:
    result = asyncio.run(run(make_config(30, ticks=20), stop_after_tick=7))

    assert len(result.records) == 7
    assert result.agent.tick == 7


def test_run_returns_the_config_with_resolved_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MODEL_ENV_VAR, "test")
    config = make_config(5, backends=Backends(deliberate="llm"))

    result = asyncio.run(run(config, stop_after_tick=0))

    assert result.records == []
    assert result.config == config.model_copy(update={"model": "test"})


def test_run_with_an_unknown_model_name_fails_before_tick_one(tmp_path: Path) -> None:
    config = make_config(5, backends=Backends(deliberate="llm"), model="no-such-model")
    run_dir = tmp_path / "run"

    with pytest.raises(ConfigError, match="no-such-model"):
        asyncio.run(run(config, run_dir))

    assert not run_dir.exists()


def test_run_with_llm_backend_and_no_model_fails_before_tick_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
    config = make_config(5, backends=Backends(appraise="llm"))
    run_dir = tmp_path / "run"

    with pytest.raises(ConfigError):
        asyncio.run(run(config, run_dir))

    assert not run_dir.exists()


def test_run_is_deterministic_in_memory() -> None:
    config = make_config(15, 15, 15)

    first = asyncio.run(run(config))
    second = asyncio.run(run(config))

    assert first == second
