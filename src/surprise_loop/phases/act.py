"""Act: send the decided action to the World. The only place the world is stepped."""

from surprise_loop.config import Runtime
from surprise_loop.phases.heuristics import require
from surprise_loop.state import Workspace


async def act(ws: Workspace, rt: Runtime) -> Workspace:
    decision = require(ws.decision, "decision", "act")
    outcome = rt.world.step(decision.action)
    return ws.model_copy(update={"outcome": outcome})
