"""Reconcile: apply the pass's outcome to mental state.

Minimal for now: store the episode and the pass record. Mood, goals, dispositions and conflict
detection arrive with the full Reconcile.
"""

from surprise_loop.config import Runtime
from surprise_loop.phases.heuristics import clamp, require
from surprise_loop.state import Episode, Memory, PassRecord, Workspace


async def reconcile(ws: Workspace, rt: Runtime) -> Workspace:
    decision = require(ws.decision, "decision", "reconcile")
    outcome = require(ws.outcome, "outcome", "reconcile")
    error = require(ws.prediction_error, "prediction_error", "reconcile")
    feedback = require(ws.feedback, "feedback", "reconcile")
    tick = ws.agent.tick + 1
    base_salience = rt.config.params.memory.base_salience

    episode = Episode(
        tick=tick,
        step=outcome.info.step,
        action=decision.action,
        prediction=decision.prediction,
        event=outcome.observation.event,
        reward=outcome.observation.reward,
        prediction_error=error,
        alignment=feedback.alignment,
        critique=feedback.critique,
        salience=clamp(base_salience + error.unsigned, 0, 1),
    )
    record = PassRecord(
        tick=tick,
        pass_index=ws.pass_index,
        decision=decision,
        outcome=outcome,
        prediction_error=error,
        feedback=feedback,
        goal_revised=None,
        conflict=None,
    )
    memory = Memory(episodes=[*ws.agent.memory.episodes, episode])
    agent = ws.agent.model_copy(update={"memory": memory})
    return ws.model_copy(update={"agent": agent, "passes": (*ws.passes, record)})
