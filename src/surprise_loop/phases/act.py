"""Act: send the decided direction to the World. The only phase that steps the world."""

from surprise_loop.config import Runtime
from surprise_loop.phases.heuristics import require
from surprise_loop.state import Workspace


async def act(ws: Workspace, rt: Runtime) -> Workspace:
    decision = require(ws.decision, "decision", "act")
    outcome = rt.world.step(decision.direction)
    return ws.model_copy(update={"outcome": outcome})
