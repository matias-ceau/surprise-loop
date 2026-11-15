"""Bounded re-entry driven by the real habit-vs-goal conflict rule in Reconcile."""

import asyncio
from pathlib import Path

import pytest

from surprise_loop.config import (
    AgentSpec,
    AppraiseParams,
    Backends,
    ConflictParams,
    DeliberateParams,
    DispositionParams,
    EvaluateParams,
    GoalParams,
    MemoryParams,
    MetricsParams,
    MonitorParams,
    MoodParams,
    Params,
    RunConfig,
    Runtime,
    initial_agent_state,
)
from surprise_loop.loop import _Phases, _tick, replay, run
from surprise_loop.phases import deliberate, reconcile
from surprise_loop.phases.perceive import build_percept
from surprise_loop.state import Decision, GoalName, Mood, Prediction, Traits, Workspace
from surprise_loop.world import Action, Event, PhaseKind, PhaseSpec, SelfAdminWorld, WorldConfig

TRAITS = Traits(reward_sensitivity=1.0, surprise_learning_rate=0.5, mood_inertia=0.5)
PRIORITIES = {GoalName.SEEK_REWARD: 1.0, GoalName.EXPLORE: 0.5, GoalName.CONSERVE: 0.5}
SEEK = GoalName.SEEK_REWARD

# Every group pinned. patience 1 and threshold 0 make every unrewarded press with a positive
# expected reward revise seek_reward downward; alignment_weight 0 keeps every bias >= 0, and
# habit_threshold 0 then lets each such revision on pass 0 raise a conflict.
PARAMS = Params(
    monitor=MonitorParams(confidence_floor=0.5, event_mismatch_weight=0.5),
    mood=MoodParams(arousal_gain=1.0),
    memory=MemoryParams(retrieval_k=10, base_salience=0.1),
    goals=GoalParams(
        ema_rate=1.0,
        error_threshold=0.0,
        patience=1,
        revision_step=0.02,
        min_priority=0.1,
        max_priority=2.0,
    ),
    dispositions=DispositionParams(rate=0.1, bound=1.0, alignment_weight=0.0),
    conflict=ConflictParams(enabled=True, habit_threshold=0.0),
    deliberate=DeliberateParams(
        temperature=0.1,
        prior_expected_reward=0.5,
        prior_weight=1.0,
        action_costs={a: 0.0 for a in Action},
        approach_weight=0.0,
        vigor_weight=0.0,
        explore_value=0.0,
        rest_value=0.0,
        confidence_halfsat=3.0,
    ),
    appraise=AppraiseParams(reward_gain=0.5),
    evaluate=EvaluateParams(scale=0.2),
    metrics=MetricsParams(quit_steps=10),
)


def make_config(
    *phases: PhaseSpec, params: Params = PARAMS, run_id: str = "reentry", seed: int = 5
) -> RunConfig:
    schedule = list(phases) or [
        PhaseSpec(kind=PhaseKind.FIXED_RATIO, steps=30, ratio=2),
        PhaseSpec(kind=PhaseKind.EXTINCTION, steps=20),
    ]
    return RunConfig(
        run_id=run_id,
        seed=seed,
        ticks=None,
        world=WorldConfig(active_lever="press_a", reward_magnitude=1.0, schedule=schedule),
        agent=AgentSpec(
            name="a", traits=TRAITS, mood=Mood(valence=0, arousal=0), goal_priorities=PRIORITIES
        ),
        params=params,
        backends=Backends(),
        model=None,
    )


def make_rt(config: RunConfig) -> Runtime:
    return Runtime(config=config, world=SelfAdminWorld(config.world, config.seed))


async def press_b_confidently(ws: Workspace, rt: Runtime) -> Workspace:
    """Always press the inactive lever expecting a reward of 1 with full confidence."""
    prediction = Prediction(expected_event=Event.REWARD, expected_reward=1.0, confidence=1.0)
    decision = Decision(action=Action.PRESS_B, prediction=prediction, rationale="scripted")
    return ws.model_copy(update={"decision": decision})


def files(run_dir: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(run_dir.iterdir())}


# --- one tick -----------------------------------------------------------------------------------


def test_conflict_on_pass_zero_gives_exactly_two_passes_even_if_pass_one_would_conflict() -> None:
    config = make_config()
    rt = make_rt(config)
    agent = initial_agent_state(config)
    reconciled: list[Workspace] = []

    async def spy_reconcile(ws: Workspace, rt: Runtime) -> Workspace:
        reconciled.append(ws)
        return await reconcile(ws, rt)

    phases = _Phases(deliberate=press_b_confidently, reconcile=spy_reconcile)
    new_agent, record = asyncio.run(_tick(agent, rt, phases))

    assert [p.pass_index for p in record.passes] == [0, 1]
    assert record.passes[0].conflict is not None
    assert record.passes[0].conflict.action == Action.PRESS_B
    assert record.passes[1].conflict is None
    assert rt.world.snapshot().step == 2
    assert new_agent.tick == agent.tick + 1 == 1
    assert new_agent.goals.revisions == 2
    # Pass 1 met the rule's conditions: the same Reconcile input as pass 0 would conflict.
    assert record.passes[1].goal_revised == SEEK
    as_pass_zero = reconciled[1].model_copy(update={"pass_index": 0, "conflict": None})
    assert asyncio.run(reconcile(as_pass_zero, rt)).conflict is not None


def test_pass_one_deliberate_receives_the_conflict_and_the_pass_zero_episode() -> None:
    config = make_config()
    rt = make_rt(config)
    agent = initial_agent_state(config)
    seen: list[Workspace] = []

    async def spy_deliberate(ws: Workspace, rt: Runtime) -> Workspace:
        seen.append(ws)
        return await press_b_confidently(ws, rt)

    _, record = asyncio.run(_tick(agent, rt, _Phases(deliberate=spy_deliberate)))

    assert [ws.pass_index for ws in seen] == [0, 1]
    pass0_ws, pass1_ws = seen
    assert pass0_ws.conflict is None
    assert pass1_ws.conflict == record.passes[0].conflict is not None
    assert pass1_ws.percept == build_percept(
        record.passes[0].outcome.observation, pass1_ws.agent, config.params
    )
    assert pass1_ws.percept is not None
    recalled = pass1_ws.percept.recalled[-1]
    assert (recalled.tick, recalled.step, recalled.action) == (1, 1, Action.PRESS_B)


def test_conflict_with_the_real_deliberate_also_re_enters() -> None:
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
    assert seen[-1].pass_index == 1
    assert seen[-1].conflict == two_pass.passes[0].conflict


def test_conflict_on_the_worlds_last_step_gives_one_pass() -> None:
    config = make_config(PhaseSpec(kind=PhaseKind.EXTINCTION, steps=3))
    rt = make_rt(config)
    agent = initial_agent_state(config)
    phases = _Phases(deliberate=press_b_confidently)

    records = []
    while not rt.world.done:
        agent, record = asyncio.run(_tick(agent, rt, phases))
        records.append(record)

    assert [len(r.passes) for r in records] == [2, 1]
    last = records[-1]
    assert last.passes[0].conflict is not None
    assert last.passes[0].outcome.info.step == 3
    assert agent.tick == 2


def test_conflict_rule_disabled_gives_one_pass_per_tick() -> None:
    params = PARAMS.model_copy(update={"conflict": ConflictParams(enabled=False)})
    config = make_config(params=params)
    result = asyncio.run(run(config))
    assert all(len(r.passes) == 1 for r in result.records)
    assert all(p.conflict is None for r in result.records for p in r.passes)
    assert sum(1 for r in result.records for p in r.passes if p.goal_revised) > 0


# --- resume and replay --------------------------------------------------------------------------


CUT = 10


def test_resumed_run_with_two_pass_ticks_on_both_sides_is_byte_identical(tmp_path: Path) -> None:
    config = make_config()
    whole, split = tmp_path / "whole", tmp_path / "split"

    full = asyncio.run(run(config, whole))
    asyncio.run(run(config, split, stop_after_tick=CUT))
    resumed = asyncio.run(run(config, split, resume=True))

    two_pass = [r.tick for r in full.records if len(r.passes) == 2]
    assert any(t <= CUT for t in two_pass)
    assert any(t > CUT for t in two_pass)
    assert full.world.step > CUT
    assert files(split) == files(whole)
    assert resumed == full
    assert asyncio.run(replay(split)) == len(full.records)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_replay_passes_on_runs_with_real_two_pass_ticks(tmp_path: Path, seed: int) -> None:
    config = make_config(seed=seed)
    result = asyncio.run(run(config, tmp_path))
    assert any(len(r.passes) == 2 for r in result.records)
    assert asyncio.run(replay(tmp_path)) == len(result.records)
