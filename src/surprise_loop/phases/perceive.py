"""Perceive: read the world observation into a percept, with recalled episodes."""

from surprise_loop.config import Params, Runtime
from surprise_loop.state import AgentState, Percept, Workspace
from surprise_loop.world import Observation


def build_percept(observation: Observation, agent: AgentState, params: Params) -> Percept:
    """What crosses into the agent: the event, reward and last action, plus last-K recall."""
    return Percept(
        event=observation.event,
        reward=observation.reward,
        last_action=observation.last_action,
        recalled=agent.memory.last_k(params.memory.retrieval_k),
    )


async def perceive(ws: Workspace, rt: Runtime) -> Workspace:
    percept = build_percept(ws.observation, ws.agent, rt.config.params)
    return ws.model_copy(update={"percept": percept})
