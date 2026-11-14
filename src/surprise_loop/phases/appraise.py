"""Appraise: an affective reading of the percept given mood and goals."""

from surprise_loop.config import Runtime
from surprise_loop.phases.heuristics import clamp, require
from surprise_loop.state import Appraisal, GoalName, Workspace


async def appraise_stub(ws: Workspace, rt: Runtime) -> Workspace:
    percept = require(ws.percept, "percept", "appraise")
    agent = ws.agent
    lift = (
        rt.config.params.appraise.reward_gain
        * agent.traits.reward_sensitivity
        * agent.goals.priorities[GoalName.SEEK_REWARD]
        * percept.reward
    )
    appraisal = Appraisal(
        valence=clamp(agent.mood.valence + lift, -1, 1),
        arousal=clamp(agent.mood.arousal, 0, 1),
    )
    return ws.model_copy(update={"appraisal": appraisal})


async def appraise(ws: Workspace, rt: Runtime) -> Workspace:
    backend = rt.config.backends.appraise
    if backend == "stub":
        return await appraise_stub(ws, rt)
    raise NotImplementedError(f'appraise backend "{backend}" is not implemented yet')
