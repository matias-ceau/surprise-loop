"""Monitor: prediction error between the predicted and the actual outcome.

Minimal for now (unweighted); the confidence-weighted version arrives with the full Reconcile.
"""

from surprise_loop.config import Runtime
from surprise_loop.phases.heuristics import require
from surprise_loop.state import PredictionError, Workspace


async def monitor(ws: Workspace, rt: Runtime) -> Workspace:
    decision = require(ws.decision, "decision", "monitor")
    outcome = require(ws.outcome, "outcome", "monitor")
    prediction = decision.prediction
    observed = outcome.observation
    raw = observed.reward - prediction.expected_reward
    error = PredictionError(
        raw_signed=raw,
        signed=raw,
        unsigned=abs(raw),
        event_mismatch=observed.event != prediction.expected_event,
    )
    return ws.model_copy(update={"prediction_error": error})
