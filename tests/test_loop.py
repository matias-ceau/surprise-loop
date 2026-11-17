"""One tick and one run: the passes of a tick, the override's second pass, when a run stops."""

import asyncio
import dataclasses
import re
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest
from bodies import ILO_TEXT, TAVI_TEXT

from surprise_loop.config import (
    MODEL_ENV_VAR,
    Backends,
    ConfigError,
    MemoryParams,
    OverrideParams,
    Params,
    RunConfig,
    Runtime,
    initial_agent_state,
    load_config,
)
from surprise_loop.loop import _Phases, _run, _tick, run, tick
from surprise_loop.phases import deliberate, reconcile
from surprise_loop.phases.perceive import build_percept
from surprise_loop.state import Override, Percept, Workspace
from surprise_loop.world import Action, Channel, ClearingWorld, WorldConfig

Phase = Callable[[Workspace, Runtime], Awaitable[Workspace]]

ROOT = Path(__file__).resolve().parents[1]
AFTERNOON = ROOT / "experiments" / "clearing_afternoon.yaml"
DAY = ROOT / "experiments" / "clearing_day.yaml"

# The real override rule disabled, so every two-pass tick here comes from an injected override.
PARAMS = Params(override=OverrideParams(enabled=False), memory=MemoryParams(retrieval_k=50))
# What an injected override offers: REST is offered at every place.
INJECTED = Override(channel=Channel.ENERGY, actions=[Action.REST], detail="injected")


def afternoon(agent: int = 0) -> RunConfig:
    """The afternoon config's run for its `agent`-th body (0 is ilo, 1 is tavi), as it is."""
    return load_config(AFTERNOON).runs()[agent]


def make_config(
    *,
    ticks: int | None = None,
    start_minute: int = 780,
    backends: Backends | None = None,
    model: str | None = None,
) -> RunConfig:
    """Ilo's afternoon run with the override rule off, from `start_minute`."""
    return afternoon().model_copy(
        update={
            "ticks": ticks,
            "world": WorldConfig(start_minute=start_minute),
            "params": PARAMS,
            "backends": backends or Backends(),
            "model": model,
        }
    )


def make_rt(config: RunConfig) -> Runtime:
    world = ClearingWorld(config.world, config.seed, config.agent.body, config.start_place)
    return Runtime(config=config, world=world)


def override_reconcile(passes: tuple[int, ...] = (0,)) -> Phase:
    """The real Reconcile, plus the injected override on the given passes."""

    async def reconcile_with_override(ws: Workspace, rt: Runtime) -> Workspace:
        ws = await reconcile(ws, rt)
        if ws.pass_index not in passes:
            return ws
        last = ws.passes[-1].model_copy(update={"override": INJECTED})
        return ws.model_copy(update={"override": INJECTED, "passes": (*ws.passes[:-1], last)})

    return reconcile_with_override


# --- tick ---------------------------------------------------------------------------------------


def test_tick_returns_one_pass_and_advances_tick_by_one() -> None:
    config = make_config()
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
    config = make_config()
    rt = make_rt(config)
    agent = initial_agent_state(config).model_copy(update={"tick": 6})

    new_agent, record = asyncio.run(tick(agent, rt))

    assert new_agent.tick == 7
    assert record.tick == 7
    assert record.passes[0].tick == 7
    assert new_agent.memory.episodes[-1].tick == 7


@pytest.mark.parametrize("agent", [0, 1], ids=["ilo", "tavi"])
def test_first_pass_of_the_afternoon_runs_reads_the_fixed_text(agent: int) -> None:
    config = afternoon(agent)

    result = asyncio.run(run(config, stop_after_tick=1))

    assert result.records[0].passes[0].scene.text() == [ILO_TEXT, TAVI_TEXT][agent]


def test_override_on_pass_zero_runs_a_second_pass() -> None:
    config = make_config()
    rt = make_rt(config)
    agent = initial_agent_state(config)
    phases = _Phases(reconcile=override_reconcile())

    new_agent, record = asyncio.run(_tick(agent, rt, phases))

    assert [p.pass_index for p in record.passes] == [0, 1]
    assert [p.tick for p in record.passes] == [1, 1]
    assert rt.world.snapshot().step == 2
    assert [p.outcome.info.step for p in record.passes] == [1, 2]
    assert new_agent.tick == 1
    assert [e.tick for e in new_agent.memory.episodes] == [1, 1]


def test_pass_one_deliberate_sees_the_rebuilt_workspace() -> None:
    config = make_config()
    rt = make_rt(config)
    agent = initial_agent_state(config)
    seen: list[Workspace] = []

    async def spy_deliberate(ws: Workspace, rt: Runtime) -> Workspace:
        seen.append(ws)
        return await deliberate(ws, rt)

    phases = _Phases(deliberate=spy_deliberate, reconcile=override_reconcile())
    _, record = asyncio.run(_tick(agent, rt, phases))

    assert [ws.pass_index for ws in seen] == [0, 1]
    pass0_ws, pass1_ws = seen
    pass0 = record.passes[0]
    assert pass0_ws.override is None
    assert pass1_ws.override == pass0.override == INJECTED
    assert pass1_ws.observation == pass0.outcome.observation
    assert pass1_ws.appraisal == record.appraisal == pass0_ws.appraisal
    assert pass1_ws.decision is None
    assert pass1_ws.outcome is None
    assert pass1_ws.comparison is None
    assert pass1_ws.feedback is None
    assert pass1_ws.passes == (pass0,)
    # The percept is rebuilt on the agent after pass 0, with only the override's actions offered,
    # so it recalls the pass-0 episode.
    assert pass1_ws.agent.tick == 0
    assert len(pass1_ws.agent.memory.episodes) == 1
    assert pass1_ws.percept == build_percept(
        pass0.outcome.observation, pass1_ws.agent, config.params, offers=INJECTED.actions
    )
    assert pass1_ws.percept is not None
    assert [o.action for o in pass1_ws.percept.offers] == INJECTED.actions
    assert record.passes[1].offers == INJECTED.actions
    assert record.passes[1].offers != pass0.outcome.observation.offers
    assert pass1_ws.percept.recalled[-1].step == pass0.outcome.info.step
    assert pass1_ws.percept.recalled[-1].tick == 1
    assert pass0_ws.percept is not None
    assert pass0_ws.percept.recalled == []


def test_override_on_the_worlds_last_step_yields_one_pass() -> None:
    config = make_config(start_minute=1190)
    rt = make_rt(config)
    agent = initial_agent_state(config)

    new_agent, record = asyncio.run(_tick(agent, rt, _Phases(reconcile=override_reconcile())))

    assert rt.world.done
    assert len(record.passes) == 1
    assert record.passes[0].override is not None
    assert new_agent.tick == 1


def test_override_on_pass_one_is_ignored() -> None:
    config = make_config()
    rt = make_rt(config)
    agent = initial_agent_state(config)
    phases = _Phases(reconcile=override_reconcile(passes=(0, 1)))

    new_agent, record = asyncio.run(_tick(agent, rt, phases))

    assert len(record.passes) == 2
    assert record.passes[1].override is not None
    assert rt.world.snapshot().step == 2
    assert new_agent.tick == 1


def test_tick_leaves_its_input_agent_untouched() -> None:
    config = make_config()
    rt = make_rt(config)
    agent = initial_agent_state(config)
    before = agent.model_copy(deep=True)

    asyncio.run(_tick(agent, rt, _Phases(reconcile=override_reconcile())))

    assert agent == before


def test_default_phases_are_the_real_ones() -> None:
    from surprise_loop import phases as real

    defaults = _Phases()
    for field in dataclasses.fields(_Phases):
        assert getattr(defaults, field.name) is getattr(real, field.name)


# --- run ----------------------------------------------------------------------------------------


def test_run_stops_after_config_ticks() -> None:
    result = asyncio.run(run(make_config(ticks=10)))

    assert len(result.records) == 10
    assert [r.tick for r in result.records] == list(range(1, 11))
    assert result.agent.tick == 10
    assert result.world.step == sum(len(r.passes) for r in result.records) == 10


def test_run_without_tick_limit_stops_when_the_world_is_done() -> None:
    config = make_config()

    result = asyncio.run(run(config))

    assert ClearingWorld.restore(config.world, result.world).done
    steps = (config.world.end_minute - config.world.start_minute) // config.world.tick_minutes
    assert result.world.step == sum(len(r.passes) for r in result.records) == steps
    assert result.agent.tick == len(result.records)


def test_run_with_overrides_counts_ticks_not_steps() -> None:
    config = make_config(ticks=5)

    result = asyncio.run(_run(config, None, phases=_Phases(reconcile=override_reconcile())))

    assert len(result.records) == 5
    assert all(len(r.passes) == 2 for r in result.records)
    assert result.world.step == 10
    assert result.agent.tick == 5


def test_every_pass_and_episode_carries_the_number_of_its_tick() -> None:
    config = make_config(ticks=12)

    result = asyncio.run(_run(config, None, phases=_Phases(reconcile=override_reconcile())))

    for record in result.records:
        assert [p.tick for p in record.passes] == [record.tick] * len(record.passes)
    episodes = result.agent.memory.episodes
    assert [e.tick for e in episodes] == [r.tick for r in result.records for _ in r.passes]


def test_run_stop_after_tick_stops_early() -> None:
    result = asyncio.run(run(make_config(ticks=20), stop_after_tick=7))

    assert len(result.records) == 7
    assert result.agent.tick == 7


def test_run_returns_the_config_with_resolved_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MODEL_ENV_VAR, "test")
    config = make_config(backends=Backends(deliberate="llm"))

    result = asyncio.run(run(config, stop_after_tick=0))

    assert result.records == []
    assert result.config == config.model_copy(update={"model": "test"})


def test_run_with_an_unknown_model_name_fails_before_tick_one(tmp_path: Path) -> None:
    config = make_config(backends=Backends(deliberate="llm"), model="no-such-model")
    run_dir = tmp_path / "run"

    with pytest.raises(ConfigError, match="no-such-model"):
        asyncio.run(run(config, run_dir))

    assert not run_dir.exists()


def test_run_with_llm_backend_and_no_model_fails_before_tick_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
    config = make_config(backends=Backends(appraise="llm"))
    run_dir = tmp_path / "run"

    with pytest.raises(ConfigError):
        asyncio.run(run(config, run_dir))

    assert not run_dir.exists()


def test_run_is_deterministic_in_memory() -> None:
    config = afternoon()

    first = asyncio.run(run(config))
    second = asyncio.run(run(config))

    assert first == second


# --- what a model would read --------------------------------------------------------------------


MACHINE_LIKE = re.compile(r"[0-9_]")


@pytest.mark.parametrize("agent", [0, 1], ids=["ilo", "tavi"])
def test_a_full_stub_day_shows_no_digit_and_no_identifier(agent: int) -> None:
    config = load_config(DAY).runs()[agent]
    percepts: list[Percept] = []

    async def spy_deliberate(ws: Workspace, rt: Runtime) -> Workspace:
        assert ws.percept is not None
        percepts.append(ws.percept)
        return await deliberate(ws, rt)

    result = asyncio.run(_run(config, None, phases=_Phases(deliberate=spy_deliberate)))

    assert ClearingWorld.restore(config.world, result.world).done
    assert len(percepts) == sum(len(r.passes) for r in result.records)
    texts = [r.appraisal.reading for r in result.records]
    for percept in percepts:
        texts += [percept.scene.text(), percept.pull_line, *(o.phrase for o in percept.offers)]
    assert [t for t in texts if MACHINE_LIKE.search(t)] == []
