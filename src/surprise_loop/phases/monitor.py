"""Monitor: compare what arrived with what Deliberate expected. The core signal."""

from surprise_loop.config import Runtime
from surprise_loop.phases.heuristics import compare, require
from surprise_loop.state import Workspace


async def monitor(ws: Workspace, rt: Runtime) -> Workspace:
    decision = require(ws.decision, "decision", "monitor")
    outcome = require(ws.outcome, "outcome", "monitor")
    comparison = compare(
        decision, outcome.observation, ws.agent.temperament, rt.config.params.monitor
    )
    return ws.model_copy(update={"comparison": comparison})
