"""Deliberate: choose an action and commit an expectation of how the body will feel after it.

The stubs are hand-written rules: `reactive` serves the loudest need it can see a way to serve,
`lookahead` plans a few steps on its own crude model of its body, `blind` picks at random. Each
commits a band around its one-step forecast and no expectation about the world.

On a model, the offers are lettered in a seeded shuffle of the tick and pass; the model answers
with a letter, a ladder phrase per channel, a yes or no for the open question the chosen action
settles, how sure it is and why. The phrases become bands around the present felt strength.
"""

from surprise_loop.agents import (
    EXPECT,
    LETTERS,
    SURENESS,
    DeliberateOutput,
    deliberate_agent,
    deliberate_prompt,
    recalled_line,
    run_agent,
)
from surprise_loop.config import DeliberateParams, Params, Runtime, derive_rng
from surprise_loop.feeling import discomfort, felt, needs
from surprise_loop.phases.heuristics import (
    bands_from_forecast,
    bands_from_words,
    clipped,
    forecast,
    heat_pull,
    reactive_choice,
    require,
    rule_choice,
    too_much,
    voiced_note,
)
from surprise_loop.phrases import OVERRIDE_LINES
from surprise_loop.state import (
    AgentState,
    Decision,
    Expectation,
    Offer,
    Sureness,
    Temperament,
    Workspace,
)
from surprise_loop.world import GO_TARGET, QUESTION_OF, Action, Fact, Levels, Place, offers_at

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
                action = rule_choice(need, place, offers_at(place), lambda _: True, p)
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


def offer_order(seed: int, tick: int, pass_index: int, actions: list[Action]) -> list[Action]:
    """The offered actions in the order they are lettered: a shuffle seeded by the run's seed,
    the tick and the pass, so it changes from tick to tick and repeats on a rerun."""
    order = list(actions)
    derive_rng(seed, tick, pass_index, "offers").shuffle(order)
    return order


def override_line(ws: Workspace) -> str:
    """On pass 1, the line that tells the model its body overrules the plan, keyed by the urgent
    need and its direction on the belief the override was decided on; else empty."""
    if ws.pass_index == 0 or ws.override is None:
        return ""
    channel = ws.override.channel
    need = needs(ws.agent.belief, ws.agent.temperament)[channel]
    return OVERRIDE_LINES[(channel, too_much(channel, need))]


def llm_prompt(ws: Workspace, order: list[Action], params: Params) -> str:
    """The Deliberate prompt of this pass, with the offers lettered in `order`."""
    percept = require(ws.percept, "percept", "deliberate")
    appraisal = require(ws.appraisal, "appraisal", "deliberate")
    phrase = {o.action: o.phrase for o in percept.offers}
    threshold = params.reconcile.voice_threshold
    recalled = [
        recalled_line(e, voiced_note(e.action, e.comparison, e.expectation.world, threshold))
        for e in percept.recalled
    ]
    return deliberate_prompt(
        percept.scene.text(),
        appraisal.reading,
        [Offer(action=a, phrase=phrase[a]) for a in order],
        recalled,
        percept.pull_line,
        override_line(ws),
        second_pass=ws.pass_index == 1,
    )


def decision_from(
    out: DeliberateOutput, order: list[Action], agent: AgentState, params: Params
) -> Decision:
    """The model's answer as a decision: the letter through `order`, the phrases as bands around
    the present belief, and the yes or no only when the chosen action settles an open question."""
    action = order[LETTERS.index(out.letter)]
    words = out.words()
    world = EXPECT[out.expect] if QUESTION_OF[action] is not None else None
    expectation = Expectation(
        words=words,
        bands=bands_from_words(words, agent.belief, agent.temperament, params.expectation),
        world=world,
        sureness=SURENESS[out.sure],
    )
    return Decision(action=action, expectation=expectation, why=out.why)


async def deliberate_llm(ws: Workspace, rt: Runtime) -> Workspace:
    percept = require(ws.percept, "percept", "deliberate")
    params = rt.config.params
    tick = ws.agent.tick + 1
    order = offer_order(rt.config.seed, tick, ws.pass_index, [o.action for o in percept.offers])
    prompt = llm_prompt(ws, order, params)
    out = await run_agent(deliberate_agent, prompt, ws, rt, len(order))
    decision = decision_from(out, order, ws.agent, params)
    return ws.model_copy(update={"decision": decision})


async def deliberate(ws: Workspace, rt: Runtime) -> Workspace:
    if rt.config.backends.deliberate == "llm":
        return await deliberate_llm(ws, rt)
    return await deliberate_stub(ws, rt)
