"""Pure helpers shared by the phases: expectation bands, the comparison, the belief update, which
actions serve a need, and the stubs' crude model of their own body.

Units: a band is in channel units. Felt strength is positive for more strain; for heat that is a
higher level, for water and energy a lower one, so their bands are mirrored.
"""

from collections.abc import Callable

from surprise_loop.config import (
    BeliefParams,
    DeliberateParams,
    ExpectationParams,
    MonitorParams,
    StubModelParams,
)
from surprise_loop.feeling import discomfort, felt
from surprise_loop.state import (
    Band,
    ChangeWord,
    Comparison,
    Decision,
    FeltScene,
    SurpriseNote,
    Temperament,
)
from surprise_loop.world import (
    GO_TARGET,
    QUESTION_OF,
    Action,
    Channel,
    Event,
    Fact,
    FactName,
    Levels,
    Observation,
    OpenQuestion,
    Place,
)

# the fact on arrival that answers each open question with "yes"
_ANSWERED_BY: dict[OpenQuestion, FactName] = {
    OpenQuestion.LOG_SHADED: FactName.LOG,
    OpenQuestion.STREAM_FULL: FactName.WATER,
}
# the facts that show there is something to drink here
_DRINKABLE = (FactName.WATER, FactName.TRICKLE)


def require[T](value: T | None, name: str, phase: str) -> T:
    """`value`, or a ValueError naming the workspace field `phase` needs but lacks."""
    if value is None:
        raise ValueError(f"{phase} needs ws.{name}, which is missing")
    return value


def clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def _to_level(c: Channel, f: float, t: Temperament) -> float:
    """The level of channel `c` at felt strength `f`: the inverse of `felt`."""
    offset = f / t.tightness[c]
    return t.setpoint.of(c) + (offset if c is Channel.HEAT else -offset)


def bands_from_words(
    words: dict[Channel, ChangeWord], belief: Levels, t: Temperament, p: ExpectationParams
) -> dict[Channel, Band]:
    """Each word is a band of change in felt strength, added to the present felt strength and
    converted to channel units (water and energy mirrored, so lo and hi swap)."""
    change: dict[ChangeWord, tuple[float | None, float | None]] = {
        ChangeWord.MUCH_LESS: (None, -p.much),
        ChangeWord.BIT_LESS: (-p.much, -p.bit),
        ChangeWord.SAME: (-p.bit, p.bit),
        ChangeWord.BIT_MORE: (p.bit, p.much),
        ChangeWord.MUCH_MORE: (p.much, None),
    }
    now = felt(belief, t)
    out: dict[Channel, Band] = {}
    for c in Channel:
        lo, hi = (None if d is None else _to_level(c, now[c] + d, t) for d in change[words[c]])
        out[c] = Band(lo=lo, hi=hi) if c is Channel.HEAT else Band(lo=hi, hi=lo)
    return out


def bands_from_forecast(forecast: Levels, p: ExpectationParams) -> dict[Channel, Band]:
    """A stub's band: its forecast plus or minus the channel's tolerance."""
    return {
        c: Band(lo=forecast.of(c) - p.stub_tolerance[c], hi=forecast.of(c) + p.stub_tolerance[c])
        for c in Channel
    }


def _into(x: float, b: Band) -> float:
    if b.lo is not None and x < b.lo:
        return b.lo
    if b.hi is not None and x > b.hi:
        return b.hi
    return x


def nearest(signal: Levels, bands: dict[Channel, Band]) -> Levels:
    """Each channel of the signal clamped into its band."""
    return Levels(**{c.value: _into(signal.of(c), bands[c]) for c in Channel})


def world_truth(question: OpenQuestion, facts: list[Fact]) -> bool:
    """The answer to an open question read from the facts on arrival."""
    return any(f.name is _ANSWERED_BY[question] for f in facts)


def compare(
    decision: Decision, observation: Observation, t: Temperament, p: MonitorParams
) -> Comparison:
    """What arrived against what was expected: (a) the body per channel, (b) the open question,
    (c) whether the body is better or worse off than expected, and the overall size."""
    expectation = decision.expectation
    signal = observation.signal
    edge = nearest(signal, expectation.bands)
    body = {c: signal.of(c) - edge.of(c) for c in Channel}
    body_felt = {c: body[c] * t.tightness[c] * (1 if c is Channel.HEAT else -1) for c in Channel}
    question = QUESTION_OF[decision.action]
    world: bool | None = None
    if (
        observation.event is Event.ARRIVED
        and question is not None
        and expectation.world is not None
    ):
        world = expectation.world != world_truth(question, observation.facts)
    size = sum(abs(v) for v in body_felt.values()) + (p.world_weight if world else 0)
    return Comparison(
        body=body,
        body_felt=body_felt,
        world=world,
        better=discomfort(edge, t) - discomfort(signal, t),
        size=size,
    )


def updated_belief(
    signal: Levels, bands: dict[Channel, Band], t: Temperament, p: BeliefParams
) -> Levels:
    """From the nearest edge of the expectation toward the signal, by how much the body trusts
    its signal against the expectation; inside the band, the signal itself."""
    edge = nearest(signal, bands)
    out: dict[str, float] = {}
    for c in Channel:
        k = t.trust[c] / (t.trust[c] + p.expectation_weight[c])
        out[c.value] = edge.of(c) + k * (signal.of(c) - edge.of(c))
    return Levels(**out)


def voiced_note(
    action: Action, comparison: Comparison, expected_world: bool | None, voice_threshold: float
) -> SurpriseNote | None:
    """What a pass's surprise voices: the loudest body error if it reaches `voice_threshold`,
    else a world mismatch (what turned out true is the opposite of `expected_world`), else None.
    A body note's `worse` is set when the felt error is above zero: hotter than expected for heat
    (whichever way the body's need points), more need than expected for water and energy.
    Reconcile voices it in the next scene; a recalled episode is told with it."""
    felt_error = comparison.body_felt
    top = loudest(felt_error)[0]
    if abs(felt_error[top]) >= voice_threshold:
        return SurpriseNote(channel=top, worse=felt_error[top] > 0)
    question = QUESTION_OF[action]
    if comparison.world and question is not None and expected_world is not None:
        return SurpriseNote(question=question, actual=not expected_world)
    return None


def too_much(c: Channel, need: float) -> bool:
    """Whether a need is one of too much: heat when positive; thirst and tiredness always."""
    return c is not Channel.HEAT or need > 0


def serving_actions(channel: Channel, too_much: bool, place: Place) -> list[Action]:
    """The actions that serve a need at a place."""
    match channel:
        case Channel.HEAT if too_much:
            return [Action.REST] if place is Place.OAK else [Action.GO_OAK]
        case Channel.HEAT:
            return [Action.REST] if place is Place.MEADOW else [Action.GO_MEADOW]
        case Channel.WATER:
            return [Action.DRINK] if place is Place.STREAM else [Action.GO_STREAM]
        case Channel.ENERGY:
            return [Action.REST]


def heat_pull(facts: list[Fact], about: Place | None) -> float:
    """The sum of the heat bearings of the facts about `about` (None: about here)."""
    return sum(f.bearing.get(Channel.HEAT, 0) for f in facts if f.about is about)


def forecast(belief: Levels, pull: float, action: Action, p: StubModelParams) -> Levels:
    """The stubs' private one-step model of their own body: heat moves toward the place's pull,
    water drains (more when hot) and comes back with a drink, energy goes on a walk and comes
    back with a rest."""
    heat = belief.heat + p.heat_rate * (pull * p.heat_per_bearing - belief.heat)
    water = belief.water - p.water_drain - p.sweat_water * max(0, belief.heat)
    if action is Action.DRINK:
        water += p.drink_water
    energy = belief.energy
    if action in GO_TARGET:
        energy -= p.walk_energy
    elif action is Action.REST:
        energy += p.rest_energy
    return Levels(heat=heat, water=water, energy=energy)


def clipped(strength: dict[Channel, float]) -> dict[Channel, float]:
    """Felt strength as need: water and energy clipped at 0 from below."""
    return {c: v if c is Channel.HEAT else max(0, v) for c, v in strength.items()}


def loudest(need: dict[Channel, float]) -> list[Channel]:
    """The channels by |need|, largest first; ties in `Channel` order."""
    return sorted(Channel, key=lambda c: -abs(need[c]))


def rule_choice(
    need: dict[Channel, float],
    place: Place,
    offers: list[Action],
    qualifies: Callable[[Action], bool],
    p: DeliberateParams,
) -> Action:
    """The reactive rule: serve the loudest need at or above `act_threshold` with the first
    offered serving action that `qualifies`; else LOOK, or the first offer when LOOK is not
    offered (pass 1 of an override offers only the serving actions)."""
    for c in loudest(need):
        if abs(need[c]) < p.act_threshold:
            break
        for action in serving_actions(c, too_much(c, need[c]), place):
            if action in offers and qualifies(action):
                return action
    return Action.LOOK if Action.LOOK in offers else offers[0]


def reactive_choice(scene: FeltScene, offers: list[Action], p: DeliberateParams) -> Action:
    """The rule on the felt scene: a walk needs a noticed fact about where it leads, a drink a
    noticed WATER or TRICKLE; a rest always qualifies."""
    noticed = [f for f in scene.facts if f.noticed]

    def qualifies(action: Action) -> bool:
        if action in GO_TARGET:
            return any(f.about is GO_TARGET[action] for f in noticed)
        if action is Action.DRINK:
            return any(f.name in _DRINKABLE for f in noticed)
        return True

    return rule_choice(clipped(scene.felt), scene.place, offers, qualifies, p)
