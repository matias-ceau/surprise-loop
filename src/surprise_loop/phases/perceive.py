"""Perceive: the observation through this body's lens, the offers in words, recalled episodes
and the gut pull."""

from surprise_loop.config import Params, Runtime
from surprise_loop.lens import felt_scene
from surprise_loop.phrases import OFFER_PHRASES, PULL_PHRASES
from surprise_loop.state import AgentState, Offer, Percept, Workspace
from surprise_loop.world import Action, Event, Observation


def build_percept(
    observation: Observation,
    agent: AgentState,
    params: Params,
    offers: list[Action] | None = None,
) -> Percept:
    """The felt scene at the observation's place, the offers (the given list, else the
    observation's) with their phrases, the episodes recalled by the configured retrieval, and the
    strongest pull among the offers at this place when it is at least `pulls.speak`."""
    scene = felt_scene(
        observation.place,
        observation.facts,
        agent.belief,
        agent.temperament,
        agent.feeling.arousal,
        agent.attention,
        agent.voiced,
        1 if observation.event is Event.LOOKED else 0,
        params.lens,
    )
    actions = observation.offers if offers is None else offers
    memory, k = agent.memory, params.memory.retrieval_k
    recalled = memory.salient_k(k) if params.memory.retrieval == "salient_k" else memory.last_k(k)

    here = agent.pulls.values[observation.place]
    pull_line = ""
    if actions:
        strongest = max(actions, key=lambda a: abs(here[a]))  # ties: the earlier offer
        if abs(here[strongest]) >= params.pulls.speak:
            pull_line = PULL_PHRASES[(strongest, here[strongest] > 0)]
    return Percept(
        scene=scene,
        offers=[Offer(action=a, phrase=OFFER_PHRASES[(observation.place, a)]) for a in actions],
        recalled=recalled,
        pull_line=pull_line,
    )


async def perceive(ws: Workspace, rt: Runtime) -> Workspace:
    percept = build_percept(ws.observation, ws.agent, rt.config.params)
    return ws.model_copy(update={"percept": percept})
