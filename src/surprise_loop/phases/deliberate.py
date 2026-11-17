"""Deliberate: choose an action and commit an expectation of how the body will feel after it.

The stubs are hand-written rules: `reactive` serves the loudest need it can see a way to serve,
`lookahead` plans a few steps on its own crude model of its body, `blind` picks at random. Each
commits a band around its one-step forecast and no expectation about the world.
"""

from surprise_loop.config import ConfigError, DeliberateParams, Runtime, derive_rng
from surprise_loop.feeling import discomfort, felt
from surprise_loop.phases.heuristics import (
    bands_from_forecast,
    clipped,
    forecast,
    heat_pull,
    reactive_choice,
    require,
    rule_choice,
)
from surprise_loop.state import Decision, Expectation, Sureness, Temperament, Workspace
from surprise_loop.world import GO_TARGET, Action, Fact, Levels, Place, _offers_at

STUB_WHY = "A rule chose this."


def _pull(facts: list[Fact], here: Place, place: Place) -> float:
    """The heat pull of `place` as the facts seen `here` tell it."""
    return heat_pull(facts, None if place is here else place)


def _ends_in(action: Action, place: Place) -> Place:
    return GO_TARGET.get(action, place)


def lookahead_choice(
    belief: Levels,
    t: Temperament,
    here: Place,
    facts: list[Fact],
    offers: list[Action],
    p: DeliberateParams,
) -> Action:
    """For each offer: that action, then `horizon - 1` steps by the reactive rule on the forecast
    belief (every target counted as noticed, every action its place allows offered), summing
    discomfort; the smallest sum wins, ties to the earlier offer."""
    best, best_total = offers[0], None
    for first in offers:
        b, place, action, total = belief, here, first, 0.0
        for step in range(p.horizon):
            if step > 0:
                need = clipped(felt(b, t))
                action = rule_choice(need, place, _offers_at(place), lambda _: True, p)
            place = _ends_in(action, place)
            b = forecast(b, _pull(facts, here, place), action, p.model)
            total += discomfort(b, t)
        if best_total is None or total < best_total:
            best, best_total = first, total
    return best


async def deliberate_stub(ws: Workspace, rt: Runtime) -> Workspace:
    percept = require(ws.percept, "percept", "deliberate")
    p = rt.config.params.deliberate
    agent = ws.agent
    offers = [o.action for o in percept.offers]
    here, facts = ws.observation.place, ws.observation.facts

    match p.stub:
        case "reactive":
            action = reactive_choice(percept.scene, offers, p)
        case "lookahead":
            action = lookahead_choice(agent.belief, agent.temperament, here, facts, offers, p)
        case "blind":
            rng = derive_rng(rt.config.seed, agent.tick + 1, ws.pass_index, "blind")
            action = rng.choice(offers)

    expected = forecast(agent.belief, _pull(facts, here, _ends_in(action, here)), action, p.model)
    expectation = Expectation(
        words=None,
        bands=bands_from_forecast(expected, rt.config.params.expectation),
        world=None,
        sureness=Sureness.FAIRLY_SURE,
    )
    decision = Decision(action=action, expectation=expectation, why=STUB_WHY)
    return ws.model_copy(update={"decision": decision})


async def deliberate(ws: Workspace, rt: Runtime) -> Workspace:
    if rt.config.backends.deliberate == "llm":
        raise ConfigError("deliberate is not built for a model yet: use the stub backend")
    return await deliberate_stub(ws, rt)
