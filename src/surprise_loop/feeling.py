"""The definitions of feeling: the one place felt strength, need, discomfort and affect live.

Felt strength is signed and in felt units: positive is more strain (too hot, thirsty, tired).
Need is the felt strength with water and energy clipped at 0 from below: having drunk more than
enough is not a need. Heat keeps its sign, because too cold is a need as well.
"""

import math

from surprise_loop.config import FeelingParams
from surprise_loop.state import Comparison, Feeling, Temperament
from surprise_loop.world import Channel, Levels


def felt(levels: Levels, t: Temperament) -> dict[Channel, float]:
    """Signed felt strength per channel."""
    out: dict[Channel, float] = {}
    for c in Channel:
        offset = levels.of(c) - t.setpoint.of(c)
        out[c] = t.tightness[c] * (offset if c is Channel.HEAT else -offset)
    return out


def needs(levels: Levels, t: Temperament) -> dict[Channel, float]:
    """Felt strength, with water and energy clipped at 0 from below."""
    return {c: v if c is Channel.HEAT else max(0, v) for c, v in felt(levels, t).items()}


def discomfort(levels: Levels, t: Temperament) -> float:
    return sum(abs(v) for v in needs(levels, t).values())


def grade(x: float, edges: tuple[float, ...]) -> int:
    """How many edges |x| reaches."""
    return sum(abs(x) >= e for e in edges)


def feel(
    belief: Levels,
    t: Temperament,
    previous: Feeling,
    comparison: Comparison | None,
    p: FeelingParams,
) -> Feeling:
    """The next feeling: tone from standing discomfort, a phasic lift or drop from how the last
    outcome compared with what was expected, arousal from surprise and strain, mood from valence."""
    need = needs(belief, t)
    d = sum(abs(v) for v in need.values())
    phasic = p.phasic_decay * previous.phasic
    size = 0
    if comparison is not None:
        phasic += p.phasic_gain * math.tanh(comparison.better)
        size = comparison.size
    valence = min(1, max(-1, t.baseline - math.tanh(p.tone_gain * d) + phasic))
    strain = min(1, max(abs(v) for v in need.values()) / p.strain_full)
    target = min(1, p.surprise_gain * size + p.strain_gain * strain)
    arousal = p.arousal_inertia * previous.arousal + (1 - p.arousal_inertia) * target
    mood = p.mood_inertia * previous.mood + (1 - p.mood_inertia) * valence
    return Feeling(discomfort=d, valence=valence, arousal=arousal, mood=mood, phasic=phasic)
