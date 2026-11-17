"""Appraise: an affective reading of the scene, and which need it is about."""

from surprise_loop.config import ConfigError, Runtime
from surprise_loop.phases.heuristics import clipped, loudest, require, too_much
from surprise_loop.phrases import STUB_READINGS
from surprise_loop.state import Appraisal, Workspace


async def appraise_stub(ws: Workspace, rt: Runtime) -> Workspace:
    """About the loudest need when it reaches `act_threshold`, else about nothing; the reading is
    the stub's one line for that need."""
    percept = require(ws.percept, "percept", "appraise")
    need = clipped(percept.scene.felt)
    top = loudest(need)[0]
    loud = abs(need[top]) >= rt.config.params.deliberate.act_threshold
    about = top if loud else None
    # with nothing loud, either key of (None, ...) holds the same line
    reading = STUB_READINGS[(about, too_much(top, need[top]))]
    return ws.model_copy(update={"appraisal": Appraisal(reading=reading, about=about)})


async def appraise(ws: Workspace, rt: Runtime) -> Workspace:
    if rt.config.backends.appraise == "llm":
        raise ConfigError("appraise is not built for a model yet: use the stub backend")
    return await appraise_stub(ws, rt)
