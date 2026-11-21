"""The seven phases of a tick, run by hand on the stand-ins."""

import asyncio
import copy
from typing import Any

import pytest
from creatures import run_config

from surprise_loop.config import (
    Backends,
    ConfigError,
    MemoryParams,
    Params,
    Runtime,
    initial_agent_state,
)
from surprise_loop.feeling import next_stir, stir_step, tone_step
from surprise_loop.lens import sense
from surprise_loop.phases import (
    act,
    appraise,
    deliberate,
    evaluate,
    monitor,
    perceive,
    reconcile,
)
from surprise_loop.reasoner import reason, reasoner_beliefs
from surprise_loop.state import Workspace
from surprise_loop.world import Belief, GridWorld, derive_rng, frame, key

PHASES = (perceive, deliberate, act, monitor, evaluate, appraise, reconcile)


def start(agent: str = "plain", seed: int = 1, **changes: Any) -> tuple[Workspace, Runtime]:
    config = run_config(agent, seed, **changes)
    world = GridWorld(config.world, config.seed)
    rt = Runtime(config=config, world=world)
    return Workspace(agent=initial_agent_state(config, world.observe())), rt


def run(phase, ws: Workspace, rt: Runtime) -> Workspace:
    return asyncio.run(phase(ws, rt))


def one_tick(ws: Workspace, rt: Runtime) -> Workspace:
    """All seven phases, then the next tick's workspace, as the loop will build it."""
    for phase in PHASES:
        ws = run(phase, ws, rt)
    return ws


def next_ws(ws: Workspace) -> Workspace:
    return Workspace(agent=ws.agent.model_copy(update={"tick": ws.agent.tick + 1}))


@pytest.mark.parametrize("deliberate_with", ["rule", "blind"])
def test_each_phase_returns_a_copy_and_only_act_moves_the_world(deliberate_with: str) -> None:
    ws, rt = start(backends=Backends(deliberate=deliberate_with))
    for _ in range(3):
        for phase in PHASES:
            before = copy.deepcopy(ws)
            world_before = rt.world.snapshot()
            new = run(phase, ws, rt)
            assert isinstance(new, Workspace)
            assert new is not ws
            assert ws == before
            if phase is act:
                assert rt.world.snapshot() != world_before
            else:
                assert rt.world.snapshot() == world_before
            ws = new
        ws = next_ws(ws)


def test_monitor_feels_the_outcome() -> None:
    ws, rt = start()
    for phase in (perceive, deliberate, act):
        ws = run(phase, ws, rt)
    after = run(monitor, ws, rt)
    signal = ws.outcome.signal
    config = rt.config
    levels, smell, damp = sense(
        signal, config.agent.temperament, derive_rng(config.seed, "sense", 1), config.params.lens
    )
    stir = next_stir(ws.agent.stir, after.comparison.surprise, config.params.feeling)
    f = after.felt_after
    assert f.at == signal.position
    assert f.moved == signal.moved
    assert (f.levels, f.smell, f.damp) == (levels, smell, damp)
    assert f.tone == tone_step(levels, config.params.feeling)
    assert after.stir == stir
    assert f.stir == stir_step(stir, config.params.feeling)


def test_evaluate_only_carries_the_reasoner() -> None:
    ws, rt = start()
    for phase in (perceive, deliberate, act, monitor):
        ws = run(phase, ws, rt)
    after = run(evaluate, ws, rt)
    assert after.agent == ws.agent
    expected = reason(
        ws.agent.reasoner,
        1,
        ws.decision.direction,
        ws.agent.felt,
        ws.felt_after,
        rt.config.world.body,
        rt.config.params.reasoner,
    )
    assert (after.reasoner, after.reasoner_changed) == expected


def test_on_the_reasoner_backend_the_own_map_follows_the_reasoner() -> None:
    ws, rt = start()
    west, south, east, north = frame(rt.config.world)
    for _ in range(12):
        ws = one_tick(ws, rt)
        agent = ws.agent
        for x in range(west, east + 1):
            for y in range(south, north + 1):
                own = agent.own_map.at((x, y))
                theirs = reasoner_beliefs(agent.reasoner, (x, y))
                assert own == pytest.approx(theirs, abs=1e-9)
        ws = next_ws(ws)


def test_reconcile_adds_one_try_and_keeps_a_short_memory() -> None:
    ws, rt = start(params=Params(memory=MemoryParams(recent_k=3)))
    for tick in range(1, 9):
        tries_before = dict(ws.agent.tries)
        ws = one_tick(ws, rt)
        target = key(ws.outcome.info.target)
        added = {k: v - tries_before.get(k, 0) for k, v in ws.agent.tries.items()}
        assert {k: v for k, v in added.items() if v} == {target: 1}
        episodes = ws.agent.memory.episodes
        assert len(episodes) == min(tick, 3)
        assert episodes[-1].tick == tick
        assert episodes[-1].salience == ws.comparison.surprise
        assert ws.record.tick == tick
        assert ws.agent.tick == tick - 1  # the loop advances it
        ws = next_ws(ws)


def test_reconcile_with_no_memory() -> None:
    ws, rt = start(params=Params(memory=MemoryParams(recent_k=0)))
    for _ in range(3):
        ws = one_tick(ws, rt)
        assert ws.agent.memory.episodes == []
        ws = next_ws(ws)


def test_the_record_of_a_tick() -> None:
    ws, rt = start()
    before = ws.agent
    ws = one_tick(ws, rt)
    r = ws.record
    assert r.felt_before == before.felt
    assert r.rule_best == ws.percept.rule.best
    assert (r.decision, r.outcome, r.felt, r.comparison) == (
        ws.decision,
        ws.outcome,
        ws.felt_after,
        ws.comparison,
    )
    assert r.reasoner_changed == ws.reasoner_changed
    assert r.appraisal == ws.appraisal
    assert r.measures.tiles_tried == 1
    assert all(0 <= v <= 1 for c in ws.appraisal.changes for v in c.beliefs.values())
    assert set(Belief) == set(ws.appraisal.changes[0].beliefs)


def test_an_llm_deliberate_backend_is_not_built_yet() -> None:
    ws, rt = start(backends=Backends(deliberate="llm"))
    ws = run(perceive, ws, rt)
    with pytest.raises(ConfigError):
        run(deliberate, ws, rt)


def test_an_llm_appraise_backend_is_not_built_yet() -> None:
    ws, rt = start(backends=Backends(appraise="llm"))
    for phase in (perceive, deliberate, act, monitor, evaluate):
        ws = run(phase, ws, rt)
    with pytest.raises(ConfigError):
        run(appraise, ws, rt)
