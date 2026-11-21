"""Reconcile: the creature after this tick, its map rewritten and the episode remembered, and the
record of the tick. The loop, not this phase, advances the tick count."""

from surprise_loop.config import Runtime
from surprise_loop.phases.heuristics import measures, require
from surprise_loop.state import Episode, Memory, TickRecord, Workspace
from surprise_loop.world import key


async def reconcile(ws: Workspace, rt: Runtime) -> Workspace:
    agent = ws.agent
    tick = agent.tick + 1
    percept = require(ws.percept, "percept", "reconcile")
    decision = require(ws.decision, "decision", "reconcile")
    outcome = require(ws.outcome, "outcome", "reconcile")
    felt_after = require(ws.felt_after, "felt_after", "reconcile")
    stir = require(ws.stir, "stir", "reconcile")
    comparison = require(ws.comparison, "comparison", "reconcile")
    reasoner = require(ws.reasoner, "reasoner", "reconcile")
    reasoner_changed = require(ws.reasoner_changed, "reasoner_changed", "reconcile")
    appraisal = require(ws.appraisal, "appraisal", "reconcile")

    target = key(outcome.info.target)
    episode = Episode(
        tick=tick,
        direction=decision.direction,
        expectation=decision.expectation,
        felt=felt_after,
        comparison=comparison,
        reading=appraisal.reading,
        salience=comparison.surprise,
    )
    recent_k = rt.config.params.memory.recent_k
    episodes = [*agent.memory.episodes, episode]
    new_agent = agent.model_copy(
        update={
            "own_map": agent.own_map.changed(appraisal.changes),
            "tries": agent.tries | {target: agent.tries.get(target, 0) + 1},
            "felt": felt_after,
            "stir": stir,
            "reasoner": reasoner,
            "memory": Memory(episodes=episodes[-recent_k:] if recent_k else []),
        }
    )
    record = TickRecord(
        tick=tick,
        felt_before=agent.felt,
        rule_best=percept.rule.best,
        decision=decision,
        outcome=outcome,
        felt=felt_after,
        comparison=comparison,
        reasoner_changed=reasoner_changed,
        appraisal=appraisal,
        measures=measures(new_agent, rt.config.world, rt.config.params.metrics),
    )
    return ws.model_copy(update={"agent": new_agent, "record": record})
