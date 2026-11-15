"""Perceive: read the world observation into a percept, with recalled episodes."""

from surprise_loop.config import Params, Runtime
from surprise_loop.state import AgentState, Percept, Workspace
from surprise_loop.world import Observation


def build_percept(observation: Observation, agent: AgentState, params: Params) -> Percept:
    """What crosses into the agent: the event, reward and last action, plus the episodes recalled
    by the configured retrieval (`last_k` or `salient_k`) with `retrieval_k`."""
    memory, k = agent.memory, params.memory.retrieval_k
    recalled = memory.salient_k(k) if params.memory.retrieval == "salient_k" else memory.last_k(k)
    return Percept(
        event=observation.event,
        reward=observation.reward,
        last_action=observation.last_action,
        recalled=recalled,
    )


async def perceive(ws: Workspace, rt: Runtime) -> Workspace:
    percept = build_percept(ws.observation, ws.agent, rt.config.params)
    return ws.model_copy(update={"percept": percept})
