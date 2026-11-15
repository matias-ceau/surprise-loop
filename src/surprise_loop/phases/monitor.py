"""Monitor: confidence-weighted prediction error between the predicted and the actual outcome.

`raw = reward − expected_reward`; `w = floor + (1 − floor)·confidence`; `signed = w·raw`;
`unsigned = w·(|raw| + event_mismatch_weight·[event ≠ expected_event])`.
"""

from surprise_loop.config import Runtime
from surprise_loop.phases.heuristics import require
from surprise_loop.state import PredictionError, Workspace


async def monitor(ws: Workspace, rt: Runtime) -> Workspace:
    decision = require(ws.decision, "decision", "monitor")
    outcome = require(ws.outcome, "outcome", "monitor")
    params = rt.config.params.monitor
    prediction = decision.prediction
    observed = outcome.observation

    raw = observed.reward - prediction.expected_reward
    mismatch = observed.event != prediction.expected_event
    weight = params.confidence_floor + (1 - params.confidence_floor) * prediction.confidence
    surprise = abs(raw) + (params.event_mismatch_weight if mismatch else 0)
    error = PredictionError(
        raw_signed=raw,
        signed=weight * raw,
        unsigned=weight * surprise,
        event_mismatch=mismatch,
    )
    return ws.model_copy(update={"prediction_error": error})
