"""The state models one tick passes around.

The creature's side of the loop: its temperament, what it felt, what it expected, how the two
compared, its own map of the world and the careful reasoner's, its memory, and the workspace a
tick's phases fill in turn. Every model is frozen; a phase returns a copy.
"""

import math
from collections.abc import Iterable
from enum import StrEnum
from typing import Annotated

from pydantic import AfterValidator, Field, field_validator, model_validator

from surprise_loop.world import (
    Belief,
    Channel,
    Direction,
    Frozen,
    Kind,
    Outcome,
    Position,
    Unit,
    WorldState,
    key,
    require_keys,
)

Count = Annotated[int, Field(ge=0)]
# a place on a five-word ladder
Step5 = Annotated[int, Field(ge=0, le=4)]
# a place on a three-word ladder
Step3 = Annotated[int, Field(ge=0, le=2)]
NonNegative = Annotated[float, Field(ge=0)]

PRESUMES_TOLERANCE = 1e-6


def every(keys: type[StrEnum]) -> AfterValidator:
    """Annotate a dict field keyed by an enum: it must hold every member of `keys`."""
    return AfterValidator(lambda value: require_keys(value, keys))


ChannelSteps = Annotated[dict[Channel, Step5], every(Channel)]
Beliefs = Annotated[dict[Belief, Unit], every(Belief)]
Shares = Annotated[dict[Kind, Unit], every(Kind)]
ChannelWeights = Annotated[dict[Channel, NonNegative], every(Channel)]


def _per_channel(value: float) -> dict[Channel, float]:
    return dict.fromkeys(Channel, value)


class Temperament(Frozen):
    # the share of each kind of tile this creature presumes an untried tile to be
    presumes: Shares
    # how loudly each channel is felt, and how much noise each reading carries
    gain: ChannelWeights = Field(default_factory=lambda: _per_channel(1.0))
    noise: ChannelWeights = Field(default_factory=lambda: _per_channel(0.02))
    blur: Unit = 0.0

    @field_validator("presumes")
    @classmethod
    def _sums_to_one(cls, value: dict[Kind, float]) -> dict[Kind, float]:
        total = math.fsum(value.values())
        if abs(total - 1) > PRESUMES_TOLERANCE:
            raise ValueError(f"the shares must sum to 1, not {total}")
        return value


class Sureness(StrEnum):
    GUESSING = "guessing"
    FAIRLY_SURE = "fairly sure"
    SURE = "sure"


class FeltRecord(Frozen):
    moved: bool | None
    levels: ChannelSteps
    smell: Step3
    damp: Step3
    tone: Step5
    stir: Step3
    at: Position


class Expectation(Frozen):
    moved: bool
    levels: ChannelSteps
    smell: Step3
    damp: Step3
    sure: Sureness


class Comparison(Frozen):
    moved_miss: bool
    misses: Annotated[dict[Channel, int], every(Channel)]
    smell_miss: int
    damp_miss: int
    surprise: Unit
    worse: float


class Change(Frozen):
    at: Position
    beliefs: Beliefs


class CreatureMap(Frozen):
    # what a tile with no entry is believed to be
    untried: Beliefs
    # by key(position)
    tiles: dict[str, Beliefs]

    def at(self, position: Position) -> dict[Belief, float]:
        return dict(self.tiles.get(key(position), self.untried))

    def changed(self, changes: Iterable[Change]) -> "CreatureMap":
        """A copy with each change applied in order; a later change to the same spot wins."""
        tiles = dict(self.tiles)
        for change in changes:
            tiles[key(change.at)] = dict(change.beliefs)
        return self.model_copy(update={"tiles": tiles})


class Touch(Frozen):
    step: Count
    at: Position


class ReasonerState(Frozen):
    presumes: Shares
    # tiles whose shares differ from `presumes`
    tiles: dict[str, Shares]
    # blocked targets of the last max(sick_delays) steps, oldest first
    touched: tuple[Touch, ...]
    # steps of the last max(sick_delays) steps at which sickness rose
    onsets: tuple[Count, ...]


class RuleChoice(Frozen):
    # non-empty, in Direction order, without repeats
    best: tuple[Direction, ...]
    choice: Direction

    @model_validator(mode="after")
    def _ordered(self) -> "RuleChoice":
        if not self.best:
            raise ValueError("best needs at least one direction")
        if list(self.best) != [d for d in Direction if d in self.best]:
            raise ValueError("best must be in Direction order without repeats")
        if self.choice not in self.best:
            raise ValueError("choice must be one of best")
        return self


class Decision(Frozen):
    direction: Direction
    expectation: Expectation
    why: str


class Appraisal(Frozen):
    reading: str
    changes: tuple[Change, ...]


class Measures(Frozen):
    tiles_tried: Count
    # over the tiles tried so far
    start_error: float
    own_error: float
    reasoner_error: float
    gap: float
    # over every tile of the layout
    own_error_all: float
    reasoner_error_all: float
    gap_all: float
    false_harm: Count
    false_safe: Count
    reasoner_false_harm: Count
    reasoner_false_safe: Count
    # keys of tried tiles where the creature's own map is on the wrong side, sorted
    wrong: tuple[str, ...]


class Episode(Frozen):
    tick: Count
    direction: Direction
    expectation: Expectation
    felt: FeltRecord
    comparison: Comparison
    reading: str
    salience: Unit


class Memory(Frozen):
    # at most memory.recent_k, oldest first
    episodes: list[Episode]


class AgentState(Frozen):
    temperament: Temperament
    felt: FeltRecord
    stir: Unit
    own_map: CreatureMap
    reasoner: ReasonerState
    tries: dict[str, Count]
    memory: Memory
    # completed ticks
    tick: Count


class Percept(Frozen):
    recent: tuple[Episode, ...]
    rule: RuleChoice


class TickRecord(Frozen):
    tick: Count
    felt_before: FeltRecord
    rule_best: tuple[Direction, ...]
    decision: Decision
    outcome: Outcome
    felt: FeltRecord
    comparison: Comparison
    reasoner_changed: tuple[Position, ...]
    appraisal: Appraisal
    measures: Measures


class TickDump(Frozen):
    run_id: str
    tick: Count
    agent: AgentState
    world: WorldState
    record: TickRecord


class Workspace(Frozen):
    agent: AgentState
    percept: Percept | None = None
    decision: Decision | None = None
    outcome: Outcome | None = None
    felt_after: FeltRecord | None = None
    stir: float | None = None
    comparison: Comparison | None = None
    reasoner: ReasonerState | None = None
    reasoner_changed: tuple[Position, ...] | None = None
    appraisal: Appraisal | None = None
    record: TickRecord | None = None
