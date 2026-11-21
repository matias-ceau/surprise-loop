"""Evaluate: the careful reasoner reads the step. It changes only the reasoner's state carried on
the workspace."""

from surprise_loop.config import Runtime
from surprise_loop.phases.heuristics import require
from surprise_loop.reasoner import reason
from surprise_loop.state import Workspace


async def evaluate(ws: Workspace, rt: Runtime) -> Workspace:
    agent = ws.agent
    decision = require(ws.decision, "decision", "evaluate")
    felt_after = require(ws.felt_after, "felt_after", "evaluate")
    state, changed = reason(
        agent.reasoner,
        agent.tick + 1,
        decision.direction,
        agent.felt,
        felt_after,
        rt.config.world.body,
        rt.config.params.reasoner,
    )
    return ws.model_copy(update={"reasoner": state, "reasoner_changed": changed})
