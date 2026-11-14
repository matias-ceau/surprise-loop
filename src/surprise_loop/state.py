"""The agent's mental state and the records of each tick.

CoreAgent is the only owner of mental state (`AgentState`). Every model here is frozen: phases
take state in and return a copy, never mutating their input.
"""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from surprise_loop.world import Action, Event, Observation, Outcome, WorldState


class GoalName(StrEnum):
    SEEK_REWARD = "seek_reward"
    EXPLORE = "explore"
    CONSERVE = "conserve"


GOAL_OF_ACTION: dict[Action, GoalName] = {
    Action.PRESS_A: GoalName.SEEK_REWARD,
    Action.PRESS_B: GoalName.SEEK_REWARD,
    Action.EXPLORE: GoalName.EXPLORE,
    Action.REST: GoalName.CONSERVE,
}

Signed = Annotated[float, Field(ge=-1, le=1)]
Unit = Annotated[float, Field(ge=0, le=1)]
Count = Annotated[int, Field(ge=0)]


def _require_keys[K, V](value: dict[K, V], keys: type[StrEnum]) -> dict[K, V]:
    """Raise unless `value` has an entry for every member of `keys`."""
    missing = [k.value for k in keys if k not in value]
    if missing:
        raise ValueError(f"missing keys: {', '.join(missing)}")
    return value


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


class Traits(_Frozen):
    reward_sensitivity: float = Field(gt=0)
    surprise_learning_rate: float = Field(gt=0, le=1)
    mood_inertia: float = Field(ge=0, lt=1)


class Mood(_Frozen):
    valence: Signed
    arousal: Unit


class Prediction(_Frozen):
    expected_event: Event
    expected_reward: float
    confidence: Unit


class PredictionError(_Frozen):
    raw_signed: float
    signed: float
    unsigned: float = Field(ge=0)
    event_mismatch: bool


class Appraisal(_Frozen):
    valence: Signed
    arousal: Unit
    note: str = ""


class Decision(_Frozen):
    action: Action
    prediction: Prediction
    rationale: str = ""


class Feedback(_Frozen):
    alignment: Signed
    critique: str = ""


class Conflict(_Frozen):
    kind: Literal["habit_vs_goal"]
    action: Action
    detail: str


class Episode(_Frozen):
    tick: Count
    step: Count
    action: Action
    prediction: Prediction
    event: Event
    reward: float
    prediction_error: PredictionError
    alignment: Signed
    critique: str = ""
    salience: Unit


class Memory(_Frozen):
    episodes: list[Episode]

    def last_k(self, k: int) -> list[Episode]:
        """The most recent `k` episodes, oldest first; all of them when `k` exceeds the length."""
        if k < 0:
            raise ValueError("k must be >= 0")
        return self.episodes[max(len(self.episodes) - k, 0) :]


class Goals(_Frozen):
    priorities: dict[GoalName, float]
    error_ema: dict[GoalName, float]
    # signed: +n / -n consecutive over-threshold passes of that sign
    error_streak: dict[GoalName, int]
    revisions: Count

    @field_validator("priorities", "error_ema", "error_streak")
    @classmethod
    def _every_goal(cls, value: dict[GoalName, float]) -> dict[GoalName, float]:
        return _require_keys(value, GoalName)


class Dispositions(_Frozen):
    biases: dict[Action, float]

    @field_validator("biases")
    @classmethod
    def _every_action(cls, value: dict[Action, float]) -> dict[Action, float]:
        return _require_keys(value, Action)


class AgentState(_Frozen):
    traits: Traits
    mood: Mood
    memory: Memory
    goals: Goals
    dispositions: Dispositions
    tick: Count  # completed ticks


class Percept(_Frozen):
    event: Event
    reward: float
    last_action: Action | None
    recalled: list[Episode]


class PassRecord(_Frozen):
    tick: Count
    pass_index: int = Field(ge=0, le=1)
    decision: Decision
    outcome: Outcome
    prediction_error: PredictionError
    feedback: Feedback
    goal_revised: GoalName | None
    conflict: Conflict | None


class TickRecord(_Frozen):
    tick: Count
    observation: Observation
    appraisal: Appraisal
    passes: list[PassRecord] = Field(min_length=1, max_length=2)


class TickDump(_Frozen):
    run_id: str
    tick: Count
    agent: AgentState
    world: WorldState
    record: TickRecord


class Workspace(_Frozen):
    """What one tick's phases pass along; each phase returns an updated copy."""

    agent: AgentState
    observation: Observation
    pass_index: int = Field(default=0, ge=0, le=1)
    percept: Percept | None = None
    appraisal: Appraisal | None = None
    conflict: Conflict | None = None
    decision: Decision | None = None
    outcome: Outcome | None = None
    prediction_error: PredictionError | None = None
    feedback: Feedback | None = None
    passes: tuple[PassRecord, ...] = ()
