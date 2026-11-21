"""Perceive: what the creature brings to this tick, and where the rule would go from here."""

from surprise_loop.config import Runtime, derive_rng
from surprise_loop.phases.heuristics import rule_choice
from surprise_loop.state import Percept, Workspace


async def perceive(ws: Workspace, rt: Runtime) -> Workspace:
    agent = ws.agent
    tick = agent.tick + 1
    rule = rule_choice(
        agent.felt,
        agent.own_map,
        agent.tries,
        rt.config.params.rule,
        derive_rng(rt.config.seed, "rule", tick),
    )
    percept = Percept(recent=tuple(agent.memory.episodes), rule=rule)
    return ws.model_copy(update={"percept": percept})
