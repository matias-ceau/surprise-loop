"""The agent's mental state, what it makes of a moment, and the records of each tick.

The agent owns its mental state. Every model here is frozen: phases take state in and return a
copy, never mutating their input.
"""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from surprise_loop.world import (
    Action,
    Channel,
    Event,
    FactKind,
    FactName,
    Frozen,
    Levels,
    Observation,
    OpenQuestion,
    Outcome,
    Place,
    Unit,
    WorldState,
)

Signed = Annotated[float, Field(ge=-1, le=1)]
Count = Annotated[int, Field(ge=0)]
PassIndex = Annotated[int, Field(ge=0, le=1)]
Sign = Literal["+", "-", "0"]  # which way a fact's value points: toward comfort, away, neither


def require_keys[K, V](value: dict[K, V], keys: type[StrEnum]) -> dict[K, V]:
    """Raise unless `value` has an entry for every member of `keys`."""
    missing = [k.value for k in keys if k not in value]
    if missing:
        raise ValueError(f"missing keys: {', '.join(missing)}")
    return value


class Temperament(Frozen):
    """What makes one body differ from another. Fixed for a run."""

    setpoint: Levels
    tightness: dict[Channel, float]
    trust: dict[Channel, Unit]
    habit: dict[FactKind, float]
    baseline: Signed

    @field_validator("tightness")
    @classmethod
    def _tightness_positive(cls, value: dict[Channel, float]) -> dict[Channel, float]:
        require_keys(value, Channel)
        slack = [c.value for c, v in value.items() if v <= 0]
        if slack:
            raise ValueError(f"tightness must be > 0: {', '.join(slack)}")
        return value

    @field_validator("trust")
    @classmethod
    def _every_channel(cls, value: dict[Channel, float]) -> dict[Channel, float]:
        return require_keys(value, Channel)

    @field_validator("habit")
    @classmethod
    def _every_kind(cls, value: dict[FactKind, float]) -> dict[FactKind, float]:
        return require_keys(value, FactKind)


class Feeling(Frozen):
    discomfort: float
    valence: Signed
    arousal: Unit
    mood: Signed
    phasic: float


class ChangeWord(StrEnum):
    MUCH_LESS = "much_less"
    BIT_LESS = "bit_less"
    SAME = "same"
    BIT_MORE = "bit_more"
    MUCH_MORE = "much_more"


class Sureness(StrEnum):
    GUESSING = "guessing"
    FAIRLY_SURE = "fairly_sure"
    SURE = "sure"


class Band(Frozen):
    """A range in channel units; None is open on that side."""

    lo: float | None
    hi: float | None


class Expectation(Frozen):
    words: dict[Channel, ChangeWord] | None  # what a model said; None for a stub
    bands: dict[Channel, Band]  # channel units, every channel
    world: bool | None  # the answer to QUESTION_OF[action]; None is no expectation
    sureness: Sureness

    @field_validator("bands")
    @classmethod
    def _every_channel(cls, value: dict[Channel, Band]) -> dict[Channel, Band]:
        return require_keys(value, Channel)


class Comparison(Frozen):
    body: dict[Channel, float]  # (a) signed error in channel units, 0 inside the band
    # the same in felt units: for heat, how much hotter than expected the body came out; for
    # water and energy, how much more need than expected. Not "worse": for a body that is too
    # cold, hotter is better
    body_felt: dict[Channel, float]
    world: bool | None  # (b) True is a mismatch, False held, None nothing to compare
    better: float  # (c) positive is better for the body than expected
    size: float = Field(ge=0)


class SurpriseNote(Frozen):
    """The last surprise worth voicing: a body surprise or a world surprise. No surprise is no
    note (None), never an empty one."""

    channel: Channel | None = None
    worse: bool | None = None
    question: OpenQuestion | None = None
    actual: bool | None = None

    @model_validator(mode="after")
    def _one_pair(self) -> "SurpriseNote":
        body = (self.channel, self.worse)
        world = (self.question, self.actual)
        unset = (None, None)
        if not ((None not in body and world == unset) or (None not in world and body == unset)):
            raise ValueError(
                "a surprise note is a body pair (channel, worse) or a world pair "
                "(question, actual); no surprise is no note"
            )
        return self


class FeltFact(Frozen):
    name: FactName
    about: Place | None
    weight: float
    value: float
    sign: Sign
    grade: int
    noticed: bool
    phrase: str | None


class BodyLine(Frozen):
    channel: Channel
    grade: int
    felt: float
    phrase: str


class FeltScene(Frozen):
    place: Place
    opening: str
    facts: list[FeltFact]  # heaviest first
    body: list[BodyLine]
    surprise: str | None
    felt: dict[Channel, float]  # raw felt strength per channel

    def text(self) -> str:
        """The opening, the noticed phrases, the body sentences and the surprise, in that order."""
        parts = [self.opening]
        parts += [f.phrase for f in self.facts if f.noticed and f.phrase is not None]
        parts += [line.phrase for line in self.body]
        if self.surprise is not None:
            parts.append(self.surprise)
        return " ".join(parts)


class Offer(Frozen):
    action: Action
    phrase: str


class Appraisal(Frozen):
    reading: str
    about: Channel | None  # the channel the reading is about, if any


class Decision(Frozen):
    action: Action
    expectation: Expectation
    why: str = ""


class Feedback(Frozen):
    judged_step: Count | None  # the step of the episode judged; None when nothing was judged
    score: Signed | None
    line: str = ""


class Override(Frozen):
    """An urgent need the chosen action did not serve; pass 1 is offered only `actions`."""

    channel: Channel
    actions: list[Action] = Field(min_length=1)  # the offered actions that serve the channel
    detail: str


class Episode(Frozen):
    tick: Count
    step: Count
    place: Place
    action: Action
    event: Event
    expectation: Expectation
    comparison: Comparison
    need_before: dict[Channel, float]  # needs at decision time
    salience: Unit
    scene: str
    reading: str
    regulation: str = ""
    score: Signed | None = None

    @field_validator("need_before")
    @classmethod
    def _every_channel(cls, value: dict[Channel, float]) -> dict[Channel, float]:
        return require_keys(value, Channel)


class Memory(Frozen):
    episodes: list[Episode]

    def last_k(self, k: int) -> list[Episode]:
        """The most recent `k` episodes, oldest first; all of them when `k` exceeds the length."""
        if k < 0:
            raise ValueError("k must be >= 0")
        return self.episodes[max(len(self.episodes) - k, 0) :]

    def salient_k(self, k: int) -> list[Episode]:
        """The most recent `k // 2` episodes plus the `k - k // 2` most salient episodes before
        them (ties go to the more recent), oldest first; all of them when `k` exceeds the length."""
        if k < 0:
            raise ValueError("k must be >= 0")
        n = len(self.episodes)
        if n <= k:
            return list(self.episodes)
        window_start = n - k // 2
        by_salience = sorted(
            range(window_start), key=lambda i: (self.episodes[i].salience, i), reverse=True
        )
        chosen = sorted(by_salience[: k - k // 2]) + list(range(window_start, n))
        return [self.episodes[i] for i in chosen]


class Pulls(Frozen):
    """How much each action at each place has helped: the slowest channel."""

    values: dict[Place, dict[Action, float]]

    @field_validator("values")
    @classmethod
    def _every_place_and_action(
        cls, value: dict[Place, dict[Action, float]]
    ) -> dict[Place, dict[Action, float]]:
        require_keys(value, Place)
        for place, per_action in value.items():
            try:
                require_keys(per_action, Action)
            except ValueError as e:
                raise ValueError(f"{place.value}: {e}") from None
        return value


class AgentState(Frozen):
    temperament: Temperament
    belief: Levels
    feeling: Feeling
    memory: Memory
    pulls: Pulls
    attention: Channel | None
    voiced: SurpriseNote | None
    tick: Count  # completed ticks


class Percept(Frozen):
    scene: FeltScene
    offers: list[Offer]
    recalled: list[Episode]
    pull_line: str


class PassRecord(Frozen):
    tick: Count
    pass_index: PassIndex
    scene: FeltScene  # what Deliberate saw on this pass
    offers: list[Action]
    decision: Decision
    outcome: Outcome
    comparison: Comparison
    feedback: Feedback
    belief_after: Levels
    feeling_after: Feeling
    override: Override | None


class TickRecord(Frozen):
    tick: Count
    observation: Observation
    appraisal: Appraisal
    passes: list[PassRecord] = Field(min_length=1, max_length=2)


class TickDump(Frozen):
    run_id: str
    tick: Count
    agent: AgentState
    world: WorldState
    record: TickRecord


class Workspace(Frozen):
    """What one tick's phases pass along; each phase returns an updated copy."""

    agent: AgentState
    observation: Observation
    pass_index: PassIndex = 0
    percept: Percept | None = None
    appraisal: Appraisal | None = None
    override: Override | None = None
    decision: Decision | None = None
    outcome: Outcome | None = None
    comparison: Comparison | None = None
    feedback: Feedback | None = None
    passes: tuple[PassRecord, ...] = ()
