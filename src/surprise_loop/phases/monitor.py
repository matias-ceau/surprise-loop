"""Monitor: what the body feels after the step, and how that compares with what was expected."""

from surprise_loop.config import MonitorParams, Runtime, derive_rng
from surprise_loop.feeling import TOP_STEP, next_stir, stir_step, tone_step, valence
from surprise_loop.lens import NEAR_TOP, sense
from surprise_loop.phases.heuristics import require
from surprise_loop.state import Comparison, Expectation, FeltRecord, Workspace
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


async def monitor(ws: Workspace, rt: Runtime) -> Workspace:
    agent = ws.agent
    decision = require(ws.decision, "decision", "monitor")
    signal = require(ws.outcome, "outcome", "monitor").signal
    params = rt.config.params
    rng = derive_rng(rt.config.seed, "sense", agent.tick + 1)
    levels, smell, damp = sense(signal, agent.temperament, rng, params.lens)
    moved = bool(signal.moved)
    comparison = compare(decision.expectation, moved, levels, smell, damp, params.monitor)
    stir = next_stir(agent.stir, comparison.surprise, params.feeling)
    felt_after = FeltRecord(
        moved=moved,
        levels=levels,
        smell=smell,
        damp=damp,
        tone=tone_step(levels, params.feeling),
        stir=stir_step(stir, params.feeling),
        at=signal.position,
    )
    return ws.model_copy(update={"felt_after": felt_after, "stir": stir, "comparison": comparison})
