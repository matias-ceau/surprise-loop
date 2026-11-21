"""Monitor: how what was felt compares with what was expected."""

from surprise_loop.config import MonitorParams
from surprise_loop.feeling import TOP_STEP, valence
from surprise_loop.lens import NEAR_TOP
from surprise_loop.state import Comparison, Expectation
from surprise_loop.world import Channel


def compare(
    e: Expectation, moved: bool, levels: dict[Channel, int], smell: int, damp: int, p: MonitorParams
) -> Comparison:
    """The misses, the size of the surprise, and how much worse for the body than expected."""
    misses = {c: levels[c] - e.levels[c] for c in Channel}
    smell_miss, damp_miss = smell - e.smell, damp - e.damp
    moved_miss = moved != e.moved
    raw = (
        p.moved_weight * moved_miss
        + sum(p.channel_weight[c] * abs(misses[c]) / TOP_STEP for c in Channel)
        + p.sense_weight * (abs(smell_miss) + abs(damp_miss)) / NEAR_TOP
    )
    surprise = min(1.0, p.sure_factor[e.sure] * raw / p.scale)
    worse = sum(-valence(c) * misses[c] for c in Channel) / TOP_STEP
    return Comparison(
        moved_miss=moved_miss,
        misses=misses,
        smell_miss=smell_miss,
        damp_miss=damp_miss,
        surprise=surprise,
        worse=worse,
    )
