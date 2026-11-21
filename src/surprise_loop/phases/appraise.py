"""Appraise: the creature's reading of the step, and the changes it makes to its own map."""

from surprise_loop.config import ConfigError, Runtime
from surprise_loop.phases.heuristics import require
from surprise_loop.reasoner import reasoner_beliefs
from surprise_loop.state import Appraisal, Change, Workspace


async def appraise(ws: Workspace, rt: Runtime) -> Workspace:
    if rt.config.backends.appraise != "reasoner":
        raise ConfigError("the llm backend of appraise arrives in Task 10")
    reasoner = require(ws.reasoner, "reasoner", "appraise")
    changed = require(ws.reasoner_changed, "reasoner_changed", "appraise")
    changes = tuple(
        Change(
            at=at,
            # a sum of shares can overshoot 1 by a rounding error
            beliefs={b: min(1.0, max(0.0, v)) for b, v in reasoner_beliefs(reasoner, at).items()},
        )
        for at in changed
    )
    return ws.model_copy(update={"appraisal": Appraisal(reading="", changes=changes)})
