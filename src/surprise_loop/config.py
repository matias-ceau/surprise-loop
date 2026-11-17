"""Parameters. Every number that shapes behaviour lives here or in a body's `Temperament`."""

from typing import Annotated

from pydantic import Field

from surprise_loop.world import Frozen, Unit

NonNegative = Annotated[float, Field(ge=0)]
Positive = Annotated[float, Field(gt=0)]


class LensParams(Frozen):
    # sharpness of the softmax over fact relevance, and how arousal sharpens it
    gain_base: NonNegative = 0.5
    gain_arousal: NonNegative = 1.5
    # how many facts get through: fewer when stirred up, never fewer than min_noticed
    min_noticed: int = Field(default=3, ge=1)
    drop_per_arousal: NonNegative = 4.0
    # extra relevance for a fact bearing on the channel the last reading was about
    attention_bonus: NonNegative = 0.3
    # how many body channels may speak, at or above narrow_arousal and below it
    narrow_arousal: Unit = 0.5
    body_lines_narrow: int = Field(default=2, ge=0)
    body_lines_wide: int = Field(default=3, ge=0)
    # the least body grade that gets a sentence
    body_speak_grade: int = Field(default=2, ge=1)
    # grade edges for a fact's value and a channel's felt strength; the first value edge is
    # also where a value stops being neutral
    value_edges: tuple[Positive, ...] = (0.25, 1.0, 2.0, 3.0)
    body_edges: tuple[Positive, ...] = (0.5, 1.0, 2.0, 3.0)


class FeelingParams(Frozen):
    # how strongly standing discomfort lowers valence
    tone_gain: NonNegative = 0.12
    # the phasic lift from a better-than-expected outcome, and its decay per tick
    phasic_gain: NonNegative = 0.3
    phasic_decay: Unit = 0.5
    # arousal's target: surprise size, plus the loudest need relative to strain_full
    surprise_gain: NonNegative = 0.5
    strain_gain: NonNegative = 0.6
    strain_full: Positive = 3.0
    arousal_inertia: Unit = 0.5
    mood_inertia: Unit = 0.9
