"""Monitor: the confidence-weighted prediction error, on hand-built workspaces."""

import asyncio

import pytest

from surprise_loop.config import (
    AgentSpec,
    Backends,
    MonitorParams,
    Params,
    RunConfig,
    Runtime,
    initial_agent_state,
)
from surprise_loop.phases import monitor
from surprise_loop.state import Decision, GoalName, Mood, Prediction, Traits, Workspace
from surprise_loop.world import (
    Action,
    Event,
    Observation,
    Outcome,
    PhaseKind,
    PhaseSpec,
    SelfAdminWorld,
    StepInfo,
    WorldConfig,
)

WORLD = WorldConfig(
    active_lever="press_a",
    reward_magnitude=1.0,
    schedule=[PhaseSpec(kind=PhaseKind.FIXED_RATIO, steps=5, ratio=1)],
)
TRAITS = Traits(reward_sensitivity=1.0, surprise_learning_rate=0.5, mood_inertia=0.5)
PRIORITIES = {GoalName.SEEK_REWARD: 1.0, GoalName.EXPLORE: 0.5, GoalName.CONSERVE: 0.5}
FLOOR = 0.2
MISMATCH = 0.4
PARAMS = Params(monitor=MonitorParams(confidence_floor=FLOOR, event_mismatch_weight=MISMATCH))


def make_rt(params: Params = PARAMS) -> Runtime:
    config = RunConfig(
        run_id="monitor",
        seed=1,
        ticks=5,
        world=WORLD,
        agent=AgentSpec(
            name="a", traits=TRAITS, mood=Mood(valence=0, arousal=0), goal_priorities=PRIORITIES
        ),
        params=params,
        backends=Backends(),
        model=None,
    )
    return Runtime(config=config, world=SelfAdminWorld(WORLD, 1))


def workspace(
    rt: Runtime, expected: float, expected_event: Event, confidence: float, reward: float
) -> Workspace:
    event = Event.REWARD if reward > 0 else Event.NO_REWARD
    observation = Observation(step=1, last_action=Action.PRESS_A, event=event, reward=reward)
    info = StepInfo(
        step=1,
        phase_index=0,
        phase_kind=PhaseKind.FIXED_RATIO,
        ratio_required=1,
        active_press=True,
        ratio_completed=1 if reward > 0 else None,
    )
    prediction = Prediction(
        expected_event=expected_event, expected_reward=expected, confidence=confidence
    )
    return Workspace(
        agent=initial_agent_state(rt.config),
        observation=Observation(step=0, last_action=None, event=Event.START, reward=0.0),
        decision=Decision(action=Action.PRESS_A, prediction=prediction),
        outcome=Outcome(observation=observation, info=info),
    )


@pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("mismatch", [False, True])
@pytest.mark.parametrize(("expected", "reward"), [(0.3, 1.0), (0.8, 0.0)])
def test_monitor_matches_formula(
    confidence: float, mismatch: bool, expected: float, reward: float
) -> None:
    rt = make_rt()
    actual_event = Event.REWARD if reward > 0 else Event.NO_REWARD
    other_event = Event.NO_REWARD if reward > 0 else Event.REWARD
    ws = workspace(rt, expected, other_event if mismatch else actual_event, confidence, reward)

    pe = asyncio.run(monitor(ws, rt)).prediction_error

    raw = reward - expected
    w = FLOOR + (1 - FLOOR) * confidence
    assert pe is not None
    assert pe.raw_signed == pytest.approx(raw)
    assert pe.signed == pytest.approx(w * raw)
    assert pe.unsigned == pytest.approx(w * (abs(raw) + (MISMATCH if mismatch else 0)))
    assert pe.event_mismatch is mismatch


def test_monitor_weights_at_hand_values() -> None:
    rt = make_rt()
    # confidence 0 -> w = floor; confidence 1 -> w = 1; confidence 0.5 -> w = 0.6
    cases = {0.0: 0.2, 0.5: 0.6, 1.0: 1.0}
    for confidence, w in cases.items():
        ws = workspace(rt, 0.5, Event.NO_REWARD, confidence, 1.0)
        pe = asyncio.run(monitor(ws, rt)).prediction_error
        assert pe is not None
        assert pe.signed == pytest.approx(w * 0.5)
        assert pe.unsigned == pytest.approx(w * (0.5 + 0.4))


def test_monitor_changes_only_prediction_error_and_leaves_input_untouched() -> None:
    rt = make_rt()
    ws = workspace(rt, 0.3, Event.NO_REWARD, 0.5, 1.0)
    before = ws.model_copy(deep=True)

    result = asyncio.run(monitor(ws, rt))

    assert ws == before
    assert result.model_copy(update={"prediction_error": None}) == ws
