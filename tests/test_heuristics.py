"""The shared pure functions of the phases: bands, comparison, belief, serving and the rules."""

import pytest
from bodies import ILO

from surprise_loop.config import (
    BeliefParams,
    DeliberateParams,
    ExpectationParams,
    MonitorParams,
    StubModelParams,
)
from surprise_loop.feeling import discomfort
from surprise_loop.phases.heuristics import (
    bands_from_forecast,
    bands_from_words,
    clamp,
    compare,
    forecast,
    heat_pull,
    nearest,
    reactive_choice,
    require,
    serving_actions,
    updated_belief,
)
from surprise_loop.state import (
    Band,
    ChangeWord,
    Decision,
    Expectation,
    FeltFact,
    FeltScene,
    Sureness,
    Temperament,
)
from surprise_loop.world import (
    FACT_ABOUT,
    Action,
    Channel,
    Event,
    Fact,
    FactName,
    Levels,
    Observation,
    Place,
    WorldConfig,
    facts_at,
)

HEAT, WATER, ENERGY = Channel.HEAT, Channel.WATER, Channel.ENERGY
WORLD = WorldConfig()
# pinned here so no assertion depends on the provisional defaults in config.py
EXPECT = ExpectationParams(
    bit=0.25, much=1.0, stub_tolerance={HEAT: 0.05, WATER: 0.03, ENERGY: 0.03}
)
MONITOR = MonitorParams(world_weight=0.7)
BELIEF = BeliefParams(expectation_weight={HEAT: 0.5, WATER: 0.5, ENERGY: 0.5})
MODEL = StubModelParams(
    heat_rate=0.25,
    heat_per_bearing=0.5,
    water_drain=0.004,
    sweat_water=0.01,
    drink_water=0.08,
    walk_energy=0.03,
    rest_energy=0.02,
)
DELIB = DeliberateParams(stub="reactive", act_threshold=0.5, horizon=6, model=MODEL)
BELIEF_NOW = Levels(heat=0.4, water=0.6, energy=0.5)


def band(lo: float | None, hi: float | None) -> Band:
    return Band(lo=lo, hi=hi)


def bands(heat: Band, water: Band, energy: Band) -> dict[Channel, Band]:
    return {HEAT: heat, WATER: water, ENERGY: energy}


def around(levels: Levels, half: float = 0.05) -> dict[Channel, Band]:
    return {c: band(levels.of(c) - half, levels.of(c) + half) for c in Channel}


def decision(
    expected: dict[Channel, Band], action: Action = Action.REST, world: bool | None = None
) -> Decision:
    expectation = Expectation(
        words=None, bands=expected, world=world, sureness=Sureness.FAIRLY_SURE
    )
    return Decision(action=action, expectation=expectation)


def observation(
    signal: Levels,
    place: Place = Place.MEADOW,
    event: Event = Event.RESTED,
    minute: int = 780,
    facts: list[Fact] | None = None,
) -> Observation:
    return Observation(
        step=1,
        minute=minute,
        place=place,
        last_action=None,
        event=event,
        facts=facts if facts is not None else facts_at(WORLD, place, minute, True, False),
        signal=signal,
        offers=[Action.REST, Action.LOOK],
    )


# --- require and clamp ------------------------------------------------------------------------


def test_require_returns_the_value_or_names_what_is_missing() -> None:
    assert require(3, "x", "act") == 3
    with pytest.raises(ValueError, match=r"act needs ws\.decision"):
        require(None, "decision", "act")


def test_clamp() -> None:
    assert clamp(2.0, -1, 1) == 1
    assert clamp(-2.0, -1, 1) == -1
    assert clamp(0.3, -1, 1) == 0.3


# --- bands ------------------------------------------------------------------------------------

# each word's band, as a change of felt strength
WORD_CHANGE = {
    ChangeWord.MUCH_LESS: (None, -1.0),
    ChangeWord.BIT_LESS: (-1.0, -0.25),
    ChangeWord.SAME: (-0.25, 0.25),
    ChangeWord.BIT_MORE: (0.25, 1.0),
    ChangeWord.MUCH_MORE: (1.0, None),
}


@pytest.mark.parametrize("word", list(ChangeWord))
def test_word_band_for_heat_is_the_change_over_tightness_above_the_belief(
    word: ChangeWord,
) -> None:
    lo, hi = WORD_CHANGE[word]
    got = bands_from_words(dict.fromkeys(Channel, word), BELIEF_NOW, ILO, EXPECT)[HEAT]
    t = ILO.tightness[HEAT]
    want_lo = None if lo is None else BELIEF_NOW.heat + lo / t
    want_hi = None if hi is None else BELIEF_NOW.heat + hi / t
    assert got.lo == pytest.approx(want_lo)
    assert got.hi == pytest.approx(want_hi)


@pytest.mark.parametrize("channel", [WATER, ENERGY])
@pytest.mark.parametrize("word", list(ChangeWord))
def test_word_band_for_water_and_energy_is_mirrored(word: ChangeWord, channel: Channel) -> None:
    lo, hi = WORD_CHANGE[word]
    got = bands_from_words(dict.fromkeys(Channel, word), BELIEF_NOW, ILO, EXPECT)[channel]
    t = ILO.tightness[channel]
    level = BELIEF_NOW.of(channel)
    # more strain is a lower level, so the felt change's hi edge becomes the level's lo edge
    want_lo = None if hi is None else level - hi / t
    want_hi = None if lo is None else level - lo / t
    assert got.lo == pytest.approx(want_lo)
    assert got.hi == pytest.approx(want_hi)


def test_a_less_thirsty_band_lies_above_the_present_water_level() -> None:
    got = bands_from_words(dict.fromkeys(Channel, ChangeWord.BIT_LESS), BELIEF_NOW, ILO, EXPECT)
    assert got[WATER].lo is not None and got[WATER].lo > BELIEF_NOW.water
    assert got[ENERGY].lo is not None and got[ENERGY].lo > BELIEF_NOW.energy
    assert got[HEAT].hi is not None and got[HEAT].hi < BELIEF_NOW.heat


def test_bands_from_forecast_are_the_forecast_plus_or_minus_tolerance() -> None:
    f = Levels(heat=0.3, water=0.6, energy=0.4)
    got = bands_from_forecast(f, EXPECT)
    assert got[HEAT].lo == pytest.approx(0.25) and got[HEAT].hi == pytest.approx(0.35)
    assert got[WATER].lo == pytest.approx(0.57) and got[WATER].hi == pytest.approx(0.63)
    assert got[ENERGY].lo == pytest.approx(0.37) and got[ENERGY].hi == pytest.approx(0.43)


def test_nearest_clamps_each_channel_into_its_band() -> None:
    signal = Levels(heat=0.9, water=0.1, energy=0.5)
    got = nearest(signal, bands(band(0.2, 0.3), band(0.4, None), band(None, 0.6)))
    assert got == Levels(heat=0.3, water=0.4, energy=0.5)
    open_both = nearest(signal, bands(band(None, None), band(None, 0.0), band(0.7, None)))
    assert open_both == Levels(heat=0.9, water=0.0, energy=0.7)


# --- compare ------------------------------------------------------------------------------------


def test_inside_the_band_every_error_is_zero() -> None:
    signal = Levels(heat=0.4, water=0.6, energy=0.5)
    got = compare(decision(around(signal)), observation(signal), ILO, MONITOR)
    assert got.body == dict.fromkeys(Channel, 0.0)
    assert got.body_felt == dict.fromkeys(Channel, 0.0)
    assert got.better == 0
    assert got.world is None
    assert got.size == 0


def test_a_heat_signal_above_its_band_is_a_positive_error() -> None:
    expected = bands(band(0.2, 0.3), band(0.55, 0.65), band(0.45, 0.55))
    got = compare(
        decision(expected), observation(Levels(heat=0.4, water=0.6, energy=0.5)), ILO, MONITOR
    )
    assert got.body[HEAT] == pytest.approx(0.1)
    assert got.body_felt[HEAT] == pytest.approx(0.1 * ILO.tightness[HEAT])
    assert got.body[WATER] == got.body[ENERGY] == 0


def test_a_water_signal_below_its_band_is_more_thirst_than_expected() -> None:
    expected = bands(band(0.35, 0.45), band(0.5, 0.6), band(0.45, 0.55))
    got = compare(
        decision(expected), observation(Levels(heat=0.4, water=0.4, energy=0.5)), ILO, MONITOR
    )
    assert got.body[WATER] == pytest.approx(-0.1)
    assert got.body_felt[WATER] == pytest.approx(0.1 * ILO.tightness[WATER])
    assert got.body_felt[WATER] > 0


def test_an_energy_signal_above_its_band_is_less_tired_than_expected() -> None:
    expected = bands(band(0.35, 0.45), band(0.55, 0.65), band(0.3, 0.4))
    got = compare(
        decision(expected), observation(Levels(heat=0.4, water=0.6, energy=0.5)), ILO, MONITOR
    )
    assert got.body[ENERGY] == pytest.approx(0.1)
    assert got.body_felt[ENERGY] == pytest.approx(-0.1 * ILO.tightness[ENERGY])


def arrive_at_oak(minute: int, world: bool | None, event: Event = Event.ARRIVED):
    signal = Levels(heat=0.4, water=0.6, energy=0.5)
    obs = observation(signal, place=Place.OAK, event=event, minute=minute)
    return compare(decision(around(signal), Action.GO_OAK, world), obs, ILO, MONITOR)


def test_arriving_at_a_sunlit_log_after_expecting_shade_is_a_world_mismatch() -> None:
    names = {f.name for f in facts_at(WORLD, Place.OAK, 780, True, False)}
    assert FactName.LOG_SUN in names and FactName.LOG not in names
    got = arrive_at_oak(780, world=True)
    assert got.world is True
    assert got.size == pytest.approx(MONITOR.world_weight)


def test_arriving_at_a_shaded_log_after_expecting_shade_held() -> None:
    names = {f.name for f in facts_at(WORLD, Place.OAK, 960, True, False)}
    assert FactName.LOG in names
    got = arrive_at_oak(960, world=True)
    assert got.world is False
    assert got.size == 0


def test_no_world_comparison_without_an_expectation_or_an_arrival() -> None:
    assert arrive_at_oak(780, world=None).world is None
    assert arrive_at_oak(780, world=True, event=Event.STUMBLED).world is None


def test_the_stream_question_reads_water_from_the_arrival_facts() -> None:
    signal = Levels(heat=0.4, water=0.6, energy=0.5)
    full = observation(
        signal, Place.STREAM, Event.ARRIVED, facts=facts_at(WORLD, Place.STREAM, 780, True, False)
    )
    trickle = observation(
        signal, Place.STREAM, Event.ARRIVED, facts=facts_at(WORLD, Place.STREAM, 780, False, False)
    )
    go = Action.GO_STREAM
    assert compare(decision(around(signal), go, True), full, ILO, MONITOR).world is False
    assert compare(decision(around(signal), go, True), trickle, ILO, MONITOR).world is True
    assert compare(decision(around(signal), go, False), trickle, ILO, MONITOR).world is False


def test_size_adds_world_weight_to_the_felt_error_on_a_mismatch() -> None:
    signal = Levels(heat=0.4, water=0.6, energy=0.5)
    expected = around(signal) | {HEAT: band(0.2, 0.3)}
    obs = observation(signal, place=Place.OAK, event=Event.ARRIVED, minute=780)
    got = compare(decision(expected, Action.GO_OAK, True), obs, ILO, MONITOR)
    assert got.size == pytest.approx(0.1 * ILO.tightness[HEAT] + MONITOR.world_weight)


def test_an_outcome_better_for_the_body_gives_positive_better_and_a_size() -> None:
    # less thirsty than expected: the signal is above the water band, below the set point
    expected = bands(band(0.35, 0.45), band(0.5, 0.6), band(0.45, 0.55))
    signal = Levels(heat=0.4, water=0.7, energy=0.5)
    got = compare(decision(expected), observation(signal), ILO, MONITOR)
    edge = Levels(heat=0.4, water=0.6, energy=0.5)
    assert got.better == pytest.approx(discomfort(edge, ILO) - discomfort(signal, ILO))
    assert got.better > 0
    assert got.size == pytest.approx(0.1 * ILO.tightness[WATER])


# --- belief -------------------------------------------------------------------------------------


def trusting(trust: float) -> Temperament:
    return ILO.model_copy(update={"trust": dict.fromkeys(Channel, trust)})


def test_inside_the_band_the_belief_is_the_signal() -> None:
    signal = Levels(heat=0.4, water=0.6, energy=0.5)
    assert updated_belief(signal, around(signal), ILO, BELIEF) == signal


def test_outside_the_band_the_belief_moves_from_the_edge_toward_the_signal() -> None:
    signal = Levels(heat=0.5, water=0.4, energy=0.5)
    expected = bands(band(0.2, 0.3), band(0.5, 0.6), band(0.45, 0.55))
    got = updated_belief(signal, expected, ILO, BELIEF)
    k_heat = ILO.trust[HEAT] / (ILO.trust[HEAT] + BELIEF.expectation_weight[HEAT])
    k_water = ILO.trust[WATER] / (ILO.trust[WATER] + BELIEF.expectation_weight[WATER])
    assert got.heat == pytest.approx(0.3 + k_heat * (0.5 - 0.3))
    assert got.water == pytest.approx(0.5 + k_water * (0.4 - 0.5))
    assert got.energy == 0.5


def test_higher_trust_moves_the_belief_further_toward_the_signal() -> None:
    signal = Levels(heat=0.5, water=0.6, energy=0.5)
    expected = around(signal) | {HEAT: band(0.2, 0.3)}
    low = updated_belief(signal, expected, trusting(0.1), BELIEF)
    high = updated_belief(signal, expected, trusting(0.9), BELIEF)
    assert 0.3 < low.heat < high.heat < 0.5


def test_the_belief_is_continuous_at_the_band_edge() -> None:
    expected = bands(band(0.2, 0.3), band(0.5, 0.6), band(0.45, 0.55))
    eps = 1e-9
    inside = updated_belief(Levels(heat=0.3, water=0.5, energy=0.5), expected, ILO, BELIEF)
    outside = updated_belief(
        Levels(heat=0.3 + eps, water=0.5 - eps, energy=0.5), expected, ILO, BELIEF
    )
    assert outside.heat == pytest.approx(inside.heat, abs=1e-8)
    assert outside.water == pytest.approx(inside.water, abs=1e-8)


# --- serving ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("channel", "too_much", "place", "want"),
    [
        (HEAT, True, Place.OAK, [Action.REST]),
        (HEAT, True, Place.MEADOW, [Action.GO_OAK]),
        (HEAT, True, Place.STREAM, [Action.GO_OAK]),
        (HEAT, False, Place.MEADOW, [Action.REST]),
        (HEAT, False, Place.OAK, [Action.GO_MEADOW]),
        (HEAT, False, Place.STREAM, [Action.GO_MEADOW]),
        (WATER, True, Place.STREAM, [Action.DRINK]),
        (WATER, True, Place.MEADOW, [Action.GO_STREAM]),
        (WATER, True, Place.OAK, [Action.GO_STREAM]),
        (ENERGY, True, Place.MEADOW, [Action.REST]),
        (ENERGY, True, Place.OAK, [Action.REST]),
        (ENERGY, True, Place.STREAM, [Action.REST]),
    ],
)
def test_serving_actions(channel: Channel, too_much: bool, place: Place, want: list) -> None:
    assert serving_actions(channel, too_much, place) == want


# --- heat pull and forecast -------------------------------------------------------------------


def test_heat_pull_sums_the_heat_bearings_of_the_facts_about_a_place() -> None:
    facts = facts_at(WORLD, Place.MEADOW, 780, True, False)
    # here: air +1.0, sun +1.0, breeze -0.3; the oak: its shade; the stream: its cool water
    assert heat_pull(facts, None) == pytest.approx(1.7)
    assert heat_pull(facts, Place.OAK) == pytest.approx(-1.0)
    assert heat_pull(facts, Place.STREAM) == pytest.approx(-0.3)
    assert heat_pull(facts, Place.MEADOW) == 0
    assert heat_pull([], None) == 0


@pytest.mark.parametrize(
    ("action", "water_gain", "energy_change"),
    [
        (Action.GO_OAK, 0.0, -0.03),
        (Action.DRINK, 0.08, 0.0),
        (Action.REST, 0.0, 0.02),
        (Action.LOOK, 0.0, 0.0),
    ],
)
def test_forecast_is_the_stubs_one_step_model(
    action: Action, water_gain: float, energy_change: float
) -> None:
    belief = Levels(heat=0.6, water=0.5, energy=0.4)
    got = forecast(belief, -1.0, action, MODEL)
    assert got.heat == pytest.approx(0.6 + 0.25 * (-1.0 * 0.5 - 0.6))
    assert got.water == pytest.approx(0.5 - 0.004 - 0.01 * 0.6 + water_gain)
    assert got.energy == pytest.approx(0.4 + energy_change)


def test_forecast_sweats_no_water_when_cold() -> None:
    got = forecast(Levels(heat=-0.5, water=0.5, energy=0.4), 0.0, Action.LOOK, MODEL)
    assert got.water == pytest.approx(0.5 - 0.004)


# --- the reactive rule ------------------------------------------------------------------------


def scene(
    place: Place, felt: dict[Channel, float], noticed: set[FactName], minute: int = 780
) -> FeltScene:
    facts = [
        FeltFact(
            name=f.name,
            about=FACT_ABOUT[f.name],
            weight=0.1,
            value=0.0,
            sign="0",
            grade=0,
            noticed=f.name in noticed,
            phrase="A thing." if f.name in noticed else None,
        )
        for f in facts_at(WORLD, place, minute, True, False)
    ]
    return FeltScene(place=place, opening="Here.", facts=facts, body=[], surprise=None, felt=felt)


MEADOW_OFFERS = [Action.GO_OAK, Action.GO_STREAM, Action.REST, Action.LOOK]
STREAM_OFFERS = [Action.GO_MEADOW, Action.GO_OAK, Action.DRINK, Action.REST, Action.LOOK]


def felt_of(heat: float = 0.0, water: float = 0.0, energy: float = 0.0) -> dict[Channel, float]:
    return {HEAT: heat, WATER: water, ENERGY: energy}


def test_a_hot_body_that_notices_the_oak_shade_goes_to_the_oak() -> None:
    s = scene(Place.MEADOW, felt_of(heat=2.0, water=1.0), {FactName.OAK_SHADE, FactName.STREAM})
    assert reactive_choice(s, MEADOW_OFFERS, DELIB) is Action.GO_OAK


def test_without_the_oak_shade_noticed_the_next_need_is_served() -> None:
    s = scene(Place.MEADOW, felt_of(heat=2.0, water=1.0), {FactName.STREAM})
    assert reactive_choice(s, MEADOW_OFFERS, DELIB) is Action.GO_STREAM


def test_without_the_oak_shade_noticed_and_no_other_need_it_looks() -> None:
    s = scene(Place.MEADOW, felt_of(heat=2.0), {FactName.STREAM})
    assert reactive_choice(s, MEADOW_OFFERS, DELIB) is Action.LOOK


def test_thirst_at_the_stream_drinks() -> None:
    s = scene(Place.STREAM, felt_of(water=1.5), {FactName.WATER})
    assert reactive_choice(s, STREAM_OFFERS, DELIB) is Action.DRINK


def test_the_loudest_need_is_served_first() -> None:
    s = scene(Place.STREAM, felt_of(heat=0.8, water=1.5), {FactName.WATER, FactName.OAK_SHADE_FAR})
    assert reactive_choice(s, STREAM_OFFERS, DELIB) is Action.DRINK
    s = scene(Place.STREAM, felt_of(heat=1.8, water=1.5), {FactName.WATER, FactName.OAK_SHADE_FAR})
    assert reactive_choice(s, STREAM_OFFERS, DELIB) is Action.GO_OAK


def test_tiredness_rests_whatever_is_noticed() -> None:
    s = scene(Place.MEADOW, felt_of(energy=1.0), set())
    assert reactive_choice(s, MEADOW_OFFERS, DELIB) is Action.REST


def test_nothing_above_the_threshold_looks() -> None:
    s = scene(Place.MEADOW, felt_of(heat=0.4, water=-3.0, energy=-2.0), {FactName.OAK_SHADE})
    assert reactive_choice(s, MEADOW_OFFERS, DELIB) is Action.LOOK


def test_a_cold_body_in_the_meadow_rests() -> None:
    s = scene(Place.MEADOW, felt_of(heat=-1.0), set())
    assert reactive_choice(s, MEADOW_OFFERS, DELIB) is Action.REST


def test_an_action_that_is_not_offered_is_not_chosen() -> None:
    s = scene(Place.MEADOW, felt_of(heat=2.0, energy=1.0), {FactName.OAK_SHADE})
    assert reactive_choice(s, [Action.REST], DELIB) is Action.REST


def test_without_look_on_offer_the_rule_falls_back_to_the_first_offer() -> None:
    s = scene(Place.MEADOW, felt_of(heat=2.0), set())
    assert reactive_choice(s, [Action.GO_OAK], DELIB) is Action.GO_OAK
