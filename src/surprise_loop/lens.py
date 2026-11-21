"""The senses: what one creature feels of its true body, and the words-ladder step of each.

The temperament bends the true numbers in a fixed order: blur within each group of channels,
then gain, then noise, then a clamp to 0..1. Smell and damp are counts the temperament leaves
alone.
"""

import random
from statistics import fmean

from surprise_loop.config import LensParams
from surprise_loop.state import Temperament
from surprise_loop.world import Channel, Signal

# channels that blur into one another; pleasure blurs with nothing
BLUR_GROUPS: tuple[tuple[Channel, ...], ...] = (
    (Channel.HUNGER, Channel.THIRST),
    (Channel.ACHE, Channel.SICKNESS, Channel.UNEASE),
)

# the strongest smell or damp has three words: none, faint, strong
NEAR_TOP = 2


def felt_levels(
    body: dict[Channel, float], t: Temperament, rng: random.Random
) -> dict[Channel, float]:
    """The body as this temperament feels it. Draws one gauss per channel, in `Channel` order."""
    levels = dict(body)
    for group in BLUR_GROUPS:
        mean = fmean(body[c] for c in group)
        for c in group:
            levels[c] = (1 - t.blur) * body[c] + t.blur * mean
    felt = {}
    for c in Channel:
        x = levels[c] * t.gain[c] + rng.gauss(0, t.noise[c])
        felt[c] = min(1.0, max(0.0, x))
    return felt


def ladder_step(level: float, edges: tuple[float, ...]) -> int:
    """The number of edges at or below `level`."""
    return sum(edge <= level for edge in edges)


def near_step(count: int) -> int:
    return min(count, NEAR_TOP)


def sense(
    signal: Signal, t: Temperament, rng: random.Random, p: LensParams
) -> tuple[dict[Channel, int], int, int]:
    """The felt ladder step of every channel, then smell and damp."""
    felt = felt_levels(signal.body, t, rng)
    levels = {c: ladder_step(felt[c], p.edges) for c in Channel}
    return levels, near_step(signal.smell), near_step(signal.damp)
