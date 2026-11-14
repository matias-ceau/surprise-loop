import math
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from surprise_loop.state import (
    GOAL_OF_ACTION,
    AgentState,
    Appraisal,
    Conflict,
    Decision,
    Dispositions,
    Episode,
    Feedback,
    GoalName,
    Goals,
    Memory,
    Mood,
    PassRecord,
    Percept,
    Prediction,
    PredictionError,
    TickDump,
    TickRecord,
    Traits,
    Workspace,
)
from surprise_loop.world import Action, Event, Observation, Outcome, PhaseKind, StepInfo, WorldState

NAN, INF = math.nan, math.inf


def traits() -> Traits:
    return Traits(reward_sensitivity=1.0, surprise_learning_rate=0.5, mood_inertia=0.5)


def prediction() -> Prediction:
    return Prediction(expected_event=Event.REWARD, expected_reward=1.0, confidence=0.5)


def prediction_error() -> PredictionError:
    return PredictionError(raw_signed=-1.0, signed=-0.5, unsigned=0.75, event_mismatch=True)


def episode(tick: int = 0) -> Episode:
    return Episode(
        tick=tick,
        step=tick + 1,
        action=Action.PRESS_A,
        prediction=prediction(),
        event=Event.NO_REWARD,
        reward=0.0,
        prediction_error=prediction_error(),
        alignment=0.2,
        critique="fine",
        salience=0.85,
    )


def goals() -> Goals:
    return Goals(
        priorities={GoalName.SEEK_REWARD: 1.0, GoalName.EXPLORE: 0.5, GoalName.CONSERVE: 0.5},
        error_ema={GoalName.SEEK_REWARD: -0.1, GoalName.EXPLORE: 0.0, GoalName.CONSERVE: 0.0},
        error_streak={GoalName.SEEK_REWARD: -2, GoalName.EXPLORE: 0, GoalName.CONSERVE: 1},
        revisions=1,
    )


def dispositions() -> Dispositions:
    return Dispositions(biases={a: 0.1 for a in Action})


def agent_state() -> AgentState:
    return AgentState(
        traits=traits(),
        mood=Mood(valence=-0.2, arousal=0.3),
        memory=Memory(episodes=[episode(0), episode(1)]),
        goals=goals(),
        dispositions=dispositions(),
        tick=2,
    )


def observation() -> Observation:
    return Observation(step=3, last_action=Action.PRESS_A, event=Event.NO_REWARD, reward=0.0)


def outcome() -> Outcome:
    info = StepInfo(
        step=3,
        phase_index=0,
        phase_kind=PhaseKind.FIXED_RATIO,
        ratio_required=5,
        active_press=True,
        ratio_completed=None,
    )
    return Outcome(observation=observation(), info=info)


def pass_record(pass_index: int = 0) -> PassRecord:
    return PassRecord(
        tick=2,
        pass_index=pass_index,
        decision=Decision(action=Action.PRESS_A, prediction=prediction(), rationale="go"),
        outcome=outcome(),
        prediction_error=prediction_error(),
        feedback=Feedback(alignment=0.3, critique="ok"),
        goal_revised=GoalName.SEEK_REWARD,
        conflict=Conflict(kind="habit_vs_goal", action=Action.PRESS_A, detail="habit"),
    )


def tick_record() -> TickRecord:
    return TickRecord(
        tick=2,
        observation=observation(),
        appraisal=Appraisal(valence=0.1, arousal=0.2, note="calm"),
        passes=[pass_record(0), pass_record(1)],
    )


def world_state() -> WorldState:
    return WorldState(
        seed=7,
        step=3,
        active_lever=Action.PRESS_A,
        phase_index=0,
        step_in_phase=3,
        presses_toward_ratio=3,
        reinforcers_in_phase=0,
        last_observation=observation(),
    )


def test_goal_of_action_covers_every_action() -> None:
    assert GOAL_OF_ACTION == {
        Action.PRESS_A: GoalName.SEEK_REWARD,
        Action.PRESS_B: GoalName.SEEK_REWARD,
        Action.EXPLORE: GoalName.EXPLORE,
        Action.REST: GoalName.CONSERVE,
    }


# (model, factory of a valid instance, field, out-of-range values); NaN and ±inf are added below
BOUNDED: list[tuple[type[BaseModel], Any, str, list[Any]]] = [
    (Traits, traits, "reward_sensitivity", [0.0, -1.0]),
    (Traits, traits, "surprise_learning_rate", [0.0, 1.01, -0.1]),
    (Traits, traits, "mood_inertia", [1.0, -0.01]),
    (Mood, lambda: Mood(valence=0.0, arousal=0.0), "valence", [-1.01, 1.01]),
    (Mood, lambda: Mood(valence=0.0, arousal=0.0), "arousal", [-0.01, 1.01]),
    (Prediction, prediction, "confidence", [-0.01, 1.01]),
    (Prediction, prediction, "expected_reward", []),
    (PredictionError, prediction_error, "raw_signed", []),
    (PredictionError, prediction_error, "signed", []),
    (PredictionError, prediction_error, "unsigned", [-0.01]),
    (Appraisal, lambda: Appraisal(valence=0.0, arousal=0.0), "valence", [-1.01, 1.01]),
    (Appraisal, lambda: Appraisal(valence=0.0, arousal=0.0), "arousal", [-0.01, 1.01]),
    (Feedback, lambda: Feedback(alignment=0.0), "alignment", [-1.01, 1.01]),
    (Episode, episode, "alignment", [-1.01, 1.01]),
    (Episode, episode, "salience", [-0.01, 1.01]),
    (Episode, episode, "reward", []),
    (Goals, goals, "revisions", [-1]),
    (AgentState, agent_state, "tick", [-1]),
    (PassRecord, pass_record, "pass_index", [-1, 2]),
]

CASES = [
    pytest.param(model, make, field, bad, id=f"{model.__name__}.{field}={bad}")
    for model, make, field, bads in BOUNDED
    for bad in [*bads, NAN, INF, -INF]
]


@pytest.mark.parametrize(("model", "make", "field", "bad"), CASES)
def test_bounded_fields_reject_out_of_range_nan_and_inf(
    model: type[BaseModel], make: Any, field: str, bad: Any
) -> None:
    data = make().model_dump()
    model.model_validate(data)  # the baseline is valid
    data[field] = bad
    with pytest.raises(ValidationError):
        model.model_validate(data)


@pytest.mark.parametrize(
    ("model", "make", "field"),
    [
        (Goals, goals, "priorities"),
        (Goals, goals, "error_ema"),
        (Dispositions, dispositions, "biases"),
    ],
)
def test_dict_values_reject_nan_and_inf(model: type[BaseModel], make: Any, field: str) -> None:
    for bad in (NAN, INF):
        data = make().model_dump()
        key = next(iter(data[field]))
        data[field][key] = bad
        with pytest.raises(ValidationError):
            model.model_validate(data)


@pytest.mark.parametrize(
    "make",
    [traits, prediction, prediction_error, episode, goals, dispositions, agent_state, pass_record],
)
def test_unknown_fields_are_rejected(make: Any) -> None:
    instance = make()
    data = instance.model_dump()
    data["surprise"] = 1
    with pytest.raises(ValidationError, match="surprise"):
        type(instance).model_validate(data)


@pytest.mark.parametrize("field", ["priorities", "error_ema", "error_streak"])
def test_goals_dicts_must_contain_every_goal(field: str) -> None:
    data = goals().model_dump()
    del data[field][GoalName.EXPLORE]
    with pytest.raises(ValidationError, match="explore"):
        Goals.model_validate(data)


def test_dispositions_must_contain_every_action() -> None:
    biases = {a: 0.0 for a in Action if a != Action.REST}
    with pytest.raises(ValidationError, match="rest"):
        Dispositions(biases=biases)


def test_tick_record_has_one_or_two_passes() -> None:
    data = tick_record().model_dump()
    TickRecord.model_validate({**data, "passes": data["passes"][:1]})
    for passes in ([], data["passes"] + data["passes"][:1]):
        with pytest.raises(ValidationError):
            TickRecord.model_validate({**data, "passes": passes})


def test_text_fields_default_to_empty() -> None:
    assert Appraisal(valence=0.0, arousal=0.0).note == ""
    assert Decision(action=Action.REST, prediction=prediction()).rationale == ""
    assert Feedback(alignment=0.0).critique == ""


def test_agent_state_survives_json_round_trip() -> None:
    state = agent_state()
    assert AgentState.model_validate_json(state.model_dump_json()) == state


def test_tick_dump_survives_json_round_trip() -> None:
    dump = TickDump(
        run_id="exp-a-s7", tick=2, agent=agent_state(), world=world_state(), record=tick_record()
    )
    assert TickDump.model_validate_json(dump.model_dump_json()) == dump


def test_memory_last_k_returns_the_last_k_in_order() -> None:
    episodes = [episode(t) for t in range(5)]
    memory = Memory(episodes=episodes)
    assert memory.last_k(2) == episodes[3:]
    assert memory.last_k(5) == episodes
    assert memory.last_k(0) == []


def test_memory_last_k_returns_all_when_k_exceeds_length() -> None:
    episodes = [episode(t) for t in range(3)]
    assert Memory(episodes=episodes).last_k(10) == episodes


def test_memory_last_k_of_empty_memory_is_empty() -> None:
    assert Memory(episodes=[]).last_k(3) == []


def test_workspace_is_frozen_and_has_documented_defaults() -> None:
    ws = Workspace(agent=agent_state(), observation=observation())
    assert ws.pass_index == 0
    assert ws.passes == ()
    for name in (
        "percept",
        "appraisal",
        "conflict",
        "decision",
        "outcome",
        "prediction_error",
        "feedback",
    ):
        assert getattr(ws, name) is None
    with pytest.raises(ValidationError):
        ws.pass_index = 1  # type: ignore[misc]


def test_state_models_are_frozen() -> None:
    state = agent_state()
    with pytest.raises(ValidationError):
        state.tick = 3  # type: ignore[misc]
    with pytest.raises(ValidationError):
        state.mood.valence = 0.5  # type: ignore[misc]


def test_workspace_copies_carry_passes_as_a_tuple() -> None:
    ws = Workspace(agent=agent_state(), observation=observation())
    percept = Percept(
        event=Event.NO_REWARD, reward=0.0, last_action=Action.PRESS_A, recalled=[episode()]
    )
    later = ws.model_copy(update={"percept": percept, "passes": (pass_record(),)})
    assert ws.percept is None and ws.passes == ()
    assert later.passes == (pass_record(),)
