"""One tick and one run: the seven phases once per tick, when a run stops, and what it leaves."""

import asyncio
import dataclasses
import math
from pathlib import Path

import pytest
from creatures import run_config

from surprise_loop.config import (
    MODEL_ENV_VAR,
    Backends,
    ConfigError,
    RunConfig,
    Runtime,
    initial_agent_state,
)
from surprise_loop.loop import _Phases, run, tick
from surprise_loop.reasoner import reasoner_beliefs
from surprise_loop.world import GridWorld, frame


def make_rt(config: RunConfig) -> tuple[Runtime, GridWorld]:
    world = GridWorld(config.world, config.seed)
    return Runtime(config=config, world=world), world


# --- tick ---------------------------------------------------------------------------------------


def test_tick_advances_the_count_and_records_the_felt_record_before_it() -> None:
    config = run_config()
    rt, world = make_rt(config)
    agent = initial_agent_state(config, world.observe())

    new_agent, record = asyncio.run(tick(agent, rt))

    assert new_agent.tick == agent.tick + 1 == 1
    assert record.tick == 1
    assert record.felt_before == agent.felt
    assert record.felt == new_agent.felt
    assert world.snapshot().step == 1
    assert [e.tick for e in new_agent.memory.episodes] == [1]


def test_tick_counts_from_the_agent_tick() -> None:
    config = run_config()
    rt, world = make_rt(config)
    agent = initial_agent_state(config, world.observe()).model_copy(update={"tick": 6})

    new_agent, record = asyncio.run(tick(agent, rt))

    assert new_agent.tick == record.tick == 7
    assert new_agent.memory.episodes[-1].tick == 7


def test_tick_leaves_its_input_agent_untouched() -> None:
    config = run_config()
    rt, world = make_rt(config)
    agent = initial_agent_state(config, world.observe())
    before = agent.model_copy(deep=True)

    asyncio.run(tick(agent, rt))

    assert agent == before


def test_default_phases_are_the_real_ones_in_the_order_of_a_tick() -> None:
    from surprise_loop import phases as real

    names = [f.name for f in dataclasses.fields(_Phases)]
    assert names == [
        "perceive",
        "deliberate",
        "act",
        "monitor",
        "evaluate",
        "appraise",
        "reconcile",
    ]
    defaults = _Phases()
    for name in names:
        assert getattr(defaults, name) is getattr(real, name)


# --- run ----------------------------------------------------------------------------------------


def test_run_stops_after_config_ticks() -> None:
    result = asyncio.run(run(run_config(ticks=30)))

    assert len(result.records) == 30
    assert [r.tick for r in result.records] == list(range(1, 31))
    assert result.agent.tick == 30
    assert result.world.step == 30


def test_run_without_tick_limit_stops_when_the_world_is_done() -> None:
    config = run_config(ticks=None)

    result = asyncio.run(run(config))

    assert len(result.records) == config.world.steps
    assert result.world.step == config.world.steps
    assert GridWorld.restore(config.world, result.world).done


def test_run_stop_after_tick_stops_early() -> None:
    result = asyncio.run(run(run_config(ticks=30), stop_after_tick=7))

    assert len(result.records) == result.agent.tick == 7


def test_run_returns_the_config_with_resolved_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MODEL_ENV_VAR, "test")
    config = run_config(backends=Backends(deliberate="llm"))

    result = asyncio.run(run(config, stop_after_tick=0))

    assert result.records == []
    assert result.config == config.model_copy(update={"model": "test"})


def test_run_with_an_unknown_model_name_fails_before_tick_one(tmp_path: Path) -> None:
    config = run_config(backends=Backends(deliberate="llm"), model="no-such-model")
    run_dir = tmp_path / "run"

    with pytest.raises(ConfigError, match="no-such-model"):
        asyncio.run(run(config, run_dir))

    assert not run_dir.exists()


def test_run_with_llm_backend_and_no_model_fails_before_tick_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
    run_dir = tmp_path / "run"

    with pytest.raises(ConfigError):
        asyncio.run(run(run_config(backends=Backends(appraise="llm")), run_dir))

    assert not run_dir.exists()


def test_the_same_config_and_seed_give_equal_results() -> None:
    config = run_config(ticks=30)

    assert asyncio.run(run(config)) == asyncio.run(run(config))


def test_a_different_seed_gives_a_different_decision_somewhere() -> None:
    one = asyncio.run(run(run_config(seed=1, ticks=30)))
    two = asyncio.run(run(run_config(seed=2, ticks=30)))

    assert [r.decision for r in one.records] != [r.decision for r in two.records]


# --- the two maps after a day -------------------------------------------------------------------


@pytest.mark.parametrize("agent", ["plain", "wary"])
def test_after_a_full_stand_in_run_the_own_map_is_the_reasoners(agent: str) -> None:
    config = run_config(agent, ticks=None)

    result = asyncio.run(run(config))

    assert len(result.records) == config.world.steps
    west, south, east, north = frame(config.world)
    own, reasoner = result.agent.own_map, result.agent.reasoner
    for x in range(west, east + 1):
        for y in range(south, north + 1):
            mine, its = own.at((x, y)), reasoner_beliefs(reasoner, (x, y))
            assert mine.keys() == its.keys()
            for belief in mine:
                assert math.isclose(mine[belief], its[belief], rel_tol=0, abs_tol=1e-9), (
                    (x, y),
                    belief,
                )
