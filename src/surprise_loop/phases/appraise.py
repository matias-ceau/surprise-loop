"""Appraise: the creature's reading of the step, and the changes it makes to its own map."""

from surprise_loop.agents import (
    AppraiseDeps,
    appraise_agent,
    appraise_prompt,
    creature_view,
    run_agent,
    to_appraisal,
)
from surprise_loop.config import Runtime
from surprise_loop.phases.heuristics import require
from surprise_loop.reasoner import reasoner_beliefs
from surprise_loop.state import Appraisal, Change, Workspace


async def appraise(ws: Workspace, rt: Runtime) -> Workspace:
    if rt.config.backends.appraise == "llm":
        appraisal = await _by_model(ws, rt)
    else:
        reasoner = require(ws.reasoner, "reasoner", "appraise")
        changed = require(ws.reasoner_changed, "reasoner_changed", "appraise")
        changes = tuple(Change(at=at, beliefs=reasoner_beliefs(reasoner, at)) for at in changed)
        appraisal = Appraisal(reading="", changes=changes)
    return ws.model_copy(update={"appraisal": appraisal})


async def _by_model(ws: Workspace, rt: Runtime) -> Appraisal:
    """The model's reading of the step just taken. The workspace's agent is still the one before
    the step, so the map it is shown is the map before this tick's changes."""
    params = rt.config.params
    percept = require(ws.percept, "percept", "appraise")
    prompt = appraise_prompt(
        creature_view(ws.agent, percept.recent, params),
        require(ws.decision, "decision", "appraise"),
        require(ws.felt_after, "felt_after", "appraise"),
        require(ws.comparison, "comparison", "appraise"),
        params,
    )
    deps = AppraiseDeps(max_changes=params.appraise.max_changes, reach=params.rule.reach)
    return to_appraisal(await run_agent(appraise_agent, prompt, ws, rt, deps))
