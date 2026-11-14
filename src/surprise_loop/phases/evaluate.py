"""Evaluate: external critique of the action against goals. Returns feedback only."""

import math

from surprise_loop.config import Runtime
from surprise_loop.phases.heuristics import clamp, goal_utilities, require
from surprise_loop.state import Feedback, Workspace


async def evaluate_stub(ws: Workspace, rt: Runtime) -> Workspace:
    percept = require(ws.percept, "percept", "evaluate")
    decision = require(ws.decision, "decision", "evaluate")
    params = rt.config.params
    utilities = goal_utilities(ws.agent, percept.recalled, params.deliberate)
    mean = sum(utilities.values()) / len(utilities)
    advantage = (utilities[decision.action] - mean) / params.evaluate.scale
    feedback = Feedback(alignment=clamp(math.tanh(advantage), -1, 1))
    return ws.model_copy(update={"feedback": feedback})


async def evaluate(ws: Workspace, rt: Runtime) -> Workspace:
    backend = rt.config.backends.evaluate
    if backend == "stub":
        return await evaluate_stub(ws, rt)
    raise NotImplementedError(f'evaluate backend "{backend}" is not implemented yet')
