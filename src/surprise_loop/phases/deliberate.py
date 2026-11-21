"""Deliberate: choose a direction and say what the body will feel after it."""

from surprise_loop.agents import (
    creature_view,
    deliberate_agent,
    deliberate_prompt,
    run_agent,
    to_decision,
)
from surprise_loop.config import Runtime, derive_rng
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
        recent = require(ws.percept, "percept", "deliberate").recent
        view = creature_view(agent, recent, rt.config.params)
        output = await run_agent(deliberate_agent, deliberate_prompt(view), ws, rt, None)
        decision = to_decision(output)
    return ws.model_copy(update={"decision": decision})
