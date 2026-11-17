"""The agent's mental state and what it makes of a moment.

CoreAgent is the only owner of mental state. Every model here is frozen: phases take state in and
return a copy, never mutating their input.
"""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, field_validator

from surprise_loop.world import (
    Channel,
    FactKind,
    FactName,
    Frozen,
    Levels,
    OpenQuestion,
    Place,
    Unit,
)

Signed = Annotated[float, Field(ge=-1, le=1)]
Count = Annotated[int, Field(ge=0)]


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
    body_felt: dict[Channel, float]  # the same in felt units; positive is more strain than expected
    world: bool | None  # (b) True is a mismatch, False held, None nothing to compare
    better: float  # (c) positive is better for the body than expected
    size: float = Field(ge=0)


class SurpriseNote(Frozen):
    """The last surprise worth voicing: a body surprise or a world surprise."""

    channel: Channel | None = None
    worse: bool | None = None
    question: OpenQuestion | None = None
    actual: bool | None = None


class FeltFact(Frozen):
    name: FactName
    about: Place | None
    weight: float
    value: float
    sign: Literal["+", "-", "0"]
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
