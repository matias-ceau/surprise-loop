"""Deliberate: choose a direction and say what the body will feel after it."""

from surprise_loop.config import ConfigError, Runtime, derive_rng
from surprise_loop.phases.heuristics import blind_decision, require, rule_expectation
from surprise_loop.state import Decision, Workspace


async def deliberate(ws: Workspace, rt: Runtime) -> Workspace:
    agent = ws.agent
    tick = agent.tick + 1
    backend = rt.config.backends.deliberate
    if backend == "rule":
        choice = require(ws.percept, "percept", "deliberate").rule.choice
        expectation = rule_expectation(
            choice, agent.felt, agent.own_map, agent.tries, rt.config.params.rule
        )
        decision = Decision(direction=choice, expectation=expectation, why="")
    elif backend == "blind":
        decision = blind_decision(agent.felt, derive_rng(rt.config.seed, "blind", tick))
    else:
        raise ConfigError("the llm backend of deliberate arrives in Task 10")
    return ws.model_copy(update={"decision": decision})
