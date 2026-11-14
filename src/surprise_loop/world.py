"""The self-administration world.

The World is the only owner of environment state: schedules, ratio counters and which lever
is active. It is deterministic under a seed. The agent only ever sees `Observation`; `StepInfo`
(phase, requirement, which lever is active) is for logging and metrics, never for the agent.
"""

import random
from enum import StrEnum
from typing import Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Action(StrEnum):
    PRESS_A = "press_a"
    PRESS_B = "press_b"
    EXPLORE = "explore"
    REST = "rest"


class Event(StrEnum):
    START = "start"
    REWARD = "reward"
    NO_REWARD = "no_reward"
    EXPLORED = "explored"
    RESTED = "rested"


class PhaseKind(StrEnum):
    FIXED_RATIO = "fixed_ratio"
    PROGRESSIVE_RATIO = "progressive_ratio"
    EXTINCTION = "extinction"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


class Observation(_Frozen):
    step: int
    last_action: Action | None
    event: Event
    reward: float


class StepInfo(_Frozen):
    step: int
    phase_index: int
    phase_kind: PhaseKind
    ratio_required: int | None
    active_press: bool
    ratio_completed: int | None


class Outcome(_Frozen):
    observation: Observation
    info: StepInfo


class PhaseSpec(_Frozen):
    kind: PhaseKind
    steps: int = Field(ge=1)
    ratio: int | None = None
    pr_start: int | None = None
    pr_increment: int | None = None

    @model_validator(mode="after")
    def _check_parameters(self) -> Self:
        fr_ok = self.ratio is not None and self.ratio >= 1
        pr_ok = (
            self.pr_start is not None
            and self.pr_start >= 1
            and self.pr_increment is not None
            and self.pr_increment >= 0
        )
        has_fr = self.ratio is not None
        has_pr = self.pr_start is not None or self.pr_increment is not None
        match self.kind:
            case PhaseKind.FIXED_RATIO:
                valid = fr_ok and not has_pr
            case PhaseKind.PROGRESSIVE_RATIO:
                valid = pr_ok and not has_fr
            case PhaseKind.EXTINCTION:
                valid = not has_fr and not has_pr
        if not valid:
            raise ValueError(
                "fixed_ratio needs ratio >= 1 only; progressive_ratio needs pr_start >= 1 and "
                "pr_increment >= 0 only; extinction takes none of them"
            )
        return self


class WorldConfig(_Frozen):
    active_lever: Literal["press_a", "press_b", "random"]
    reward_magnitude: float = Field(gt=0)
    schedule: list[PhaseSpec] = Field(min_length=1)


class WorldState(_Frozen):
    seed: int
    step: int
    active_lever: Action
    phase_index: int
    step_in_phase: int
    presses_toward_ratio: int
    reinforcers_in_phase: int
    last_observation: Observation


class WorldDone(RuntimeError):
    """Raised by `step()` once the schedule has ended."""


class World(Protocol):
    @property
    def done(self) -> bool: ...
    def observe(self) -> Observation: ...
    def step(self, action: Action) -> Outcome: ...
    def snapshot(self) -> WorldState: ...


class SelfAdminWorld:
    """Operant chamber with two levers; one is active and reinforced under the schedule."""

    def __init__(self, config: WorldConfig, seed: int) -> None:
        self._config = config
        self._state = WorldState(
            seed=seed,
            step=0,
            active_lever=_draw_active_lever(config, seed),
            phase_index=0,
            step_in_phase=0,
            presses_toward_ratio=0,
            reinforcers_in_phase=0,
            last_observation=Observation(step=0, last_action=None, event=Event.START, reward=0.0),
        )

    @classmethod
    def restore(cls, config: WorldConfig, state: WorldState) -> "SelfAdminWorld":
        world = cls.__new__(cls)
        world._config = config
        world._state = state
        return world

    @property
    def done(self) -> bool:
        return self._state.phase_index >= len(self._config.schedule)

    def observe(self) -> Observation:
        return self._state.last_observation

    def snapshot(self) -> WorldState:
        return self._state

    def step(self, action: Action) -> Outcome:
        if self.done:
            raise WorldDone("the schedule has ended")
        s = self._state
        phase = self._config.schedule[s.phase_index]
        required = _requirement(phase, s.reinforcers_in_phase)
        active_press = action == s.active_lever
        presses = s.presses_toward_ratio + 1 if active_press else s.presses_toward_ratio
        reinforced = active_press and required is not None and presses >= required

        if reinforced:
            event, reward = Event.REWARD, self._config.reward_magnitude
            presses, reinforcers = 0, s.reinforcers_in_phase + 1
        else:
            event, reward = _UNREWARDED_EVENT[action], 0.0
            reinforcers = s.reinforcers_in_phase

        step = s.step + 1
        observation = Observation(step=step, last_action=action, event=event, reward=reward)
        info = StepInfo(
            step=step,
            phase_index=s.phase_index,
            phase_kind=phase.kind,
            ratio_required=required,
            active_press=active_press,
            ratio_completed=required if reinforced else None,
        )

        phase_index, step_in_phase = s.phase_index, s.step_in_phase + 1
        if step_in_phase == phase.steps:
            phase_index, step_in_phase, presses, reinforcers = phase_index + 1, 0, 0, 0

        self._state = s.model_copy(
            update={
                "step": step,
                "phase_index": phase_index,
                "step_in_phase": step_in_phase,
                "presses_toward_ratio": presses,
                "reinforcers_in_phase": reinforcers,
                "last_observation": observation,
            }
        )
        return Outcome(observation=observation, info=info)


_UNREWARDED_EVENT = {
    Action.PRESS_A: Event.NO_REWARD,
    Action.PRESS_B: Event.NO_REWARD,
    Action.EXPLORE: Event.EXPLORED,
    Action.REST: Event.RESTED,
}


def _draw_active_lever(config: WorldConfig, seed: int) -> Action:
    if config.active_lever != "random":
        return Action(config.active_lever)
    return random.Random(f"{seed}:active_lever").choice([Action.PRESS_A, Action.PRESS_B])


def _requirement(phase: PhaseSpec, reinforcers: int) -> int | None:
    match phase.kind:
        case PhaseKind.FIXED_RATIO:
            return phase.ratio
        case PhaseKind.PROGRESSIVE_RATIO:
            return phase.pr_start + reinforcers * phase.pr_increment
        case PhaseKind.EXTINCTION:
            return None
