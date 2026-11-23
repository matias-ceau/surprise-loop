"""The summary feeling, defined in one place: the tone of the felt levels, and stir.

Tone weighs every channel's felt step, pleasure for and the others against. Stir follows the size
of the surprise, so it rises when expectations fail and settles when they hold.
"""

from surprise_loop.config import FeelingParams
from surprise_loop.lens import ladder_step
from surprise_loop.phrases import FEEL_WORDS
from surprise_loop.world import Channel

# the top step of the five-word ladder, which a tone sum is scaled by
TOP_STEP = len(FEEL_WORDS) - 1


def valence(c: Channel) -> int:
    """+1 for a channel that feels good when high (pleasure), -1 for the others."""
    return 1 if c is Channel.PLEASURE else -1


def tone_step(levels: dict[Channel, int], p: FeelingParams) -> int:
    total = sum(valence(c) * p.tone_weights[c] * levels[c] for c in Channel)
    value = min(1.0, max(-1.0, total / TOP_STEP))
    return ladder_step(value, p.tone_edges)


def next_stir(stir: float, surprise: float, p: FeelingParams) -> float:
    return (1 - p.stir_rate) * stir + p.stir_rate * surprise


def stir_step(stir: float, p: FeelingParams) -> int:
    return ladder_step(stir, p.stir_edges)
