"""Evaluate (code): judge the episode `window` back by how its served need changed since.
Returns feedback only; Reconcile applies it. An episode whose action did not take place (it
stumbled, or a collapse replaced it with rest) is not judged."""

from surprise_loop.config import Runtime
from surprise_loop.feeling import needs
from surprise_loop.phases.heuristics import clamp, require
from surprise_loop.phrases import REGULATION_LINES
from surprise_loop.state import Feedback, Workspace
from surprise_loop.world import SERVES, Event

# events after which the chosen action did not take place
NOT_DONE = (Event.STUMBLED, Event.COLLAPSED)


def judged_index(n: int, window: int) -> int | None:
    """The index of the episode to judge among `n` stored ones, if there are enough."""
    return n - window if n >= window else None


async def evaluate(ws: Workspace, rt: Runtime) -> Workspace:
    outcome = require(ws.outcome, "outcome", "evaluate")
    p = rt.config.params.evaluate
    episodes = ws.agent.memory.episodes
    feedback = Feedback(judged_step=None, score=None, line="")
    i = judged_index(len(episodes), p.window)
    if i is not None:
        episode = episodes[i]
        channel = SERVES[episode.action]
        if episode.score is None and channel is not None and episode.event not in NOT_DONE:
            now = abs(needs(outcome.observation.signal, ws.agent.temperament)[channel])
            score = clamp((abs(episode.need_before[channel]) - now) / p.score_scale, -1, 1)
            verdict = 1 if score >= p.helped else -1 if score <= -p.helped else 0
            feedback = Feedback(
                judged_step=episode.step,
                score=score,
                line=REGULATION_LINES[(episode.action, verdict)],
            )
    return ws.model_copy(update={"feedback": feedback})
