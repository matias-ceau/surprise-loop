"""Bounded re-entry driven by the real override rule in Reconcile."""

import asyncio
from pathlib import Path

from surprise_loop.config import (
    OverrideParams,
    Params,
    RunConfig,
    Runtime,
    initial_agent_state,
    load_config,
)
from surprise_loop.loop import _Phases, _tick, run
from surprise_loop.phases import deliberate, reconcile
from surprise_loop.phases.perceive import build_percept
from surprise_loop.state import Band, Decision, Expectation, Sureness, Workspace
from surprise_loop.world import Action, Channel, ClearingWorld, WorldConfig, offers_at

ROOT = Path(__file__).resolve().parents[1]
AFTERNOON = ROOT / "experiments" / "clearing_afternoon.yaml"

# Low enough that ilo's hot, tired afternoon body is urgent on every channel the scripted LOOK
# leaves unserved, and any error at or above zero counts as worse than expected.
URGENT = Params(override=OverrideParams(enabled=True, urgent_felt=0.5, worse_margin=0.0))


def make_config(start_minute: int = 780, params: Params = URGENT) -> RunConfig:
    """Ilo's afternoon run, from `start_minute`, with `params`."""
    return (
        load_config(AFTERNOON)
        .runs()[0]
        .model_copy(update={"world": WorldConfig(start_minute=start_minute), "params": params})
    )


def make_rt(config: RunConfig) -> Runtime:
    world = ClearingWorld(config.world, config.seed, config.agent.body, config.start_place)
    return Runtime(config=config, world=world)


async def look_expecting_comfort(ws: Workspace, rt: Runtime) -> Workspace:
    """Always LOOK (it serves no need), expecting the body to sit exactly at its set point: every
    strained channel then comes out worse than expected."""
    setpoint = ws.agent.temperament.setpoint
    bands = {c: Band(lo=setpoint.of(c), hi=setpoint.of(c)) for c in Channel}
    expectation = Expectation(words=None, bands=bands, world=None, sureness=Sureness.SURE)
    decision = Decision(action=Action.LOOK, expectation=expectation, why="scripted")
    return ws.model_copy(update={"decision": decision})


# --- one tick -----------------------------------------------------------------------------------


def test_override_on_pass_zero_gives_exactly_two_passes_even_if_pass_one_would_override() -> None:
    config = make_config()
    rt = make_rt(config)
    agent = initial_agent_state(config)
    reconciled: list[Workspace] = []

    async def spy_reconcile(ws: Workspace, rt: Runtime) -> Workspace:
        reconciled.append(ws)
        return await reconcile(ws, rt)

    phases = _Phases(deliberate=look_expecting_comfort, reconcile=spy_reconcile)
    new_agent, record = asyncio.run(_tick(agent, rt, phases))

    assert [p.pass_index for p in record.passes] == [0, 1]
    assert record.passes[0].override is not None
    assert record.passes[1].override is None
    assert rt.world.snapshot().step == 2
    assert new_agent.tick == agent.tick + 1 == 1
    assert len(new_agent.memory.episodes) == 2
    # Pass 1 met the rule's conditions: the same Reconcile input as pass 0 would override.
    as_pass_zero = reconciled[1].model_copy(update={"pass_index": 0, "override": None})
    assert asyncio.run(reconcile(as_pass_zero, rt)).override is not None


def test_pass_one_deliberate_receives_only_serving_offers_and_the_pass_zero_episode() -> None:
    config = make_config()
    rt = make_rt(config)
    agent = initial_agent_state(config)
    seen: list[Workspace] = []

    async def spy_deliberate(ws: Workspace, rt: Runtime) -> Workspace:
        seen.append(ws)
        if ws.pass_index == 0:
            return await look_expecting_comfort(ws, rt)
        return await deliberate(ws, rt)

    _, record = asyncio.run(_tick(agent, rt, _Phases(deliberate=spy_deliberate)))

    assert [ws.pass_index for ws in seen] == [0, 1]
    pass0_ws, pass1_ws = seen
    pass0, pass1 = record.passes
    override = pass0.override
    assert override is not None
    assert pass0_ws.override is None
    assert pass1_ws.override == override
    observation = pass0.outcome.observation
    assert pass1_ws.percept == build_percept(
        observation, pass1_ws.agent, config.params, offers=override.actions
    )
    assert pass1_ws.percept is not None
    assert [o.action for o in pass1_ws.percept.offers] == override.actions
    # What Deliberate saw is what the record keeps, and it is cut from the place's full offers.
    assert pass1.offers == override.actions
    assert pass1.offers != offers_at(observation.place)
    assert pass1.decision.action in override.actions
    recalled = pass1_ws.percept.recalled[-1]
    assert (recalled.tick, recalled.step, recalled.action) == (1, 1, Action.LOOK)


def test_override_with_the_real_phases_also_re_enters() -> None:
    config = make_config()
    rt = make_rt(config)
    agent = initial_agent_state(config)
    seen: list[Workspace] = []

    async def spy_deliberate(ws: Workspace, rt: Runtime) -> Workspace:
        seen.append(ws)
        return await deliberate(ws, rt)

    phases = _Phases(deliberate=spy_deliberate)
    two_pass = None
    while two_pass is None and not rt.world.done:
        agent, record = asyncio.run(_tick(agent, rt, phases))
        if len(record.passes) == 2:
            two_pass = record
    assert two_pass is not None
    override = two_pass.passes[0].override
    assert override is not None
    assert seen[-1].pass_index == 1
    assert seen[-1].override == override
    assert two_pass.passes[1].offers == override.actions
    assert two_pass.passes[1].offers != two_pass.passes[0].outcome.observation.offers


def test_override_on_the_worlds_last_step_gives_one_pass() -> None:
    config = make_config(start_minute=1170)  # three world steps
    rt = make_rt(config)
    agent = initial_agent_state(config)
    phases = _Phases(deliberate=look_expecting_comfort)

    records = []
    while not rt.world.done:
        agent, record = asyncio.run(_tick(agent, rt, phases))
        records.append(record)

    assert [len(r.passes) for r in records] == [2, 1]
    last = records[-1]
    assert last.passes[0].override is not None
    assert last.passes[0].outcome.info.step == 3
    assert agent.tick == 2


def test_override_disabled_gives_one_pass_per_tick() -> None:
    params = URGENT.model_copy(
        update={"override": URGENT.override.model_copy(update={"enabled": False})}
    )
    result = asyncio.run(run(make_config(params=params)))
    assert all(len(r.passes) == 1 for r in result.records)
    assert all(p.override is None for r in result.records for p in r.passes)


def test_every_override_cuts_pass_one_to_its_actions() -> None:
    result = asyncio.run(run(make_config()))

    two_pass = [r for r in result.records if len(r.passes) == 2]
    assert two_pass
    for record in two_pass:
        pass0, pass1 = record.passes
        assert pass0.override is not None
        assert pass1.offers == pass0.override.actions
        assert pass1.offers != pass0.outcome.observation.offers
