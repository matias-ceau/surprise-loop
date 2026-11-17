import pytest
from pydantic import ValidationError

from surprise_loop.world import (
    Action,
    BodyParams,
    BodyState,
    Channel,
    ClearingWorld,
    Event,
    FactName,
    Outcome,
    Place,
    World,
    WorldConfig,
    WorldDone,
    WorldState,
    air_at,
    facts_at,
    frozen_noon_facts,
    log_shaded_at,
    sun_at,
)

# The fact-table tests pin today's curves explicitly, so retuning the defaults does not move them.
TABLE_CONFIG = WorldConfig(
    air_curve=[(360, 18.0), (780, 30.0), (960, 30.0), (1200, 22.0)],
    place_air_offset={Place.MEADOW: 0.0, Place.OAK: -3.0, Place.STREAM: -1.5},
    air_neutral_c=25.0,
    air_scale_c=5.0,
    sun_curve=[(360, 0.2), (660, 1.0), (900, 1.0), (1200, 0.1)],
    place_sun={Place.MEADOW: 1.0, Place.OAK: 0.0, Place.STREAM: 0.6},
    sun_fact_min=0.3,
    log_sun_from=660,
    log_sun_until=930,
)
NINE = 9 * 60
ONE = 13 * 60
BODY = BodyState(heat=0.0, water=0.7, energy=0.7)

# A body whose every drift is switched off, so one rule at a time can be switched back on.
QUIET = dict(
    heat_rate=0.0,
    walk_heat=0.0,
    sweat_threshold=1.0,
    water_drain=0.0,
    base_energy=0.0,
    walk_energy=0.0,
    absorb_rate=0.0,
    heat_hot_limit=2.0,
    heat_cold_limit=-2.0,
    water_floor=0.0,
    energy_floor=0.0,
    signal_noise={c: 0.0 for c in Channel},
)


def quiet(**changes: object) -> WorldConfig:
    return TABLE_CONFIG.model_copy(update={"body": BodyParams.model_validate(QUIET | changes)})


def world_at(
    config: WorldConfig,
    *,
    place: Place = Place.MEADOW,
    minute: int = NINE,
    body: BodyState = BODY,
    stream_full: bool = True,
    seed: int = 0,
) -> ClearingWorld:
    """A world restored from a hand-built state."""
    state = ClearingWorld(config, seed, body, place).snapshot()
    state = state.model_copy(update={"minute": minute, "stream_full": stream_full})
    return ClearingWorld.restore(config, state)


def names(**kwargs: object) -> list[FactName]:
    return [f.name for f in facts_at(TABLE_CONFIG, **kwargs)]  # type: ignore[arg-type]


# --- facts --------------------------------------------------------------------------------------


@pytest.mark.parametrize("stream_full", [True, False])
def test_facts_at_the_meadow_at_one_are_the_frozen_noon_facts(stream_full: bool) -> None:
    facts = facts_at(WorldConfig(), Place.MEADOW, ONE, stream_full=stream_full, looked=False)
    assert facts == frozen_noon_facts()


F = FactName
# (place, minute, stream_full, looked) -> the expected names, without the stream and look facts
BASE = {
    (Place.MEADOW, NINE): [
        F.COOL_AIR,
        F.SUN,
        F.NO_SEAT,
        F.OAK_SHADE,
        F.STREAM,
        F.BREEZE,
        F.BIRDSONG,
    ],
    (Place.MEADOW, ONE): [F.AIR, F.SUN, F.NO_SEAT, F.OAK_SHADE, F.STREAM, F.BREEZE, F.BIRDSONG],
    (Place.OAK, NINE): [F.COOL_AIR, F.SHADE, F.LOG, F.STREAM, F.MEADOW_SUN, F.BREEZE, F.BIRDSONG],
    (Place.OAK, ONE): [F.AIR, F.SHADE, F.LOG_SUN, F.STREAM, F.MEADOW_SUN, F.BREEZE, F.BIRDSONG],
    (Place.STREAM, NINE): [
        F.COOL_AIR,
        F.SUN,
        F.NO_SEAT,
        F.OAK_SHADE_FAR,
        F.WATER,
        F.MEADOW_SUN,
        F.BREEZE,
        F.BIRDSONG,
    ],
    (Place.STREAM, ONE): [
        F.AIR,
        F.SUN,
        F.NO_SEAT,
        F.OAK_SHADE_FAR,
        F.WATER,
        F.MEADOW_SUN,
        F.BREEZE,
        F.BIRDSONG,
    ],
}


def expected_names(place: Place, minute: int, stream_full: bool, looked: bool) -> list[FactName]:
    out = list(BASE[(place, minute)])
    if place is Place.STREAM and not stream_full:
        out[out.index(F.WATER)] = F.TRICKLE
    if place is Place.MEADOW and looked:
        seen = F.OAK_LOG_SHADED if minute == NINE else F.OAK_LOG_SUNLIT
        out[out.index(F.OAK_SHADE)] = seen
    return out


@pytest.mark.parametrize("looked", [False, True])
@pytest.mark.parametrize("stream_full", [True, False])
@pytest.mark.parametrize("minute", [NINE, ONE])
@pytest.mark.parametrize("place", list(Place))
def test_fact_presence(place: Place, minute: int, stream_full: bool, looked: bool) -> None:
    got = names(place=place, minute=minute, stream_full=stream_full, looked=looked)
    assert got == expected_names(place, minute, stream_full, looked)


@pytest.mark.parametrize("minute", range(360, 1201, 30))
@pytest.mark.parametrize("place", list(Place))
def test_a_fact_list_is_in_canonical_order(place: Place, minute: int) -> None:
    order = list(FactName)
    for stream_full in (True, False):
        for looked in (True, False):
            got = names(place=place, minute=minute, stream_full=stream_full, looked=looked)
            assert got == sorted(got, key=order.index)
            assert len(set(got)) == len(got)


def test_computed_bearings_and_kinds() -> None:
    by_name = {f.name: f for f in facts_at(TABLE_CONFIG, Place.STREAM, NINE, True, False)}
    air = air_at(TABLE_CONFIG, Place.STREAM, NINE)
    assert by_name[F.COOL_AIR].bearing == {Channel.HEAT: pytest.approx((air - 25.0) / 5.0)}
    assert by_name[F.COOL_AIR].bearing[Channel.HEAT] < 0
    assert by_name[F.SUN].bearing == {
        Channel.HEAT: pytest.approx(sun_at(TABLE_CONFIG, Place.STREAM, NINE))
    }
    assert by_name[F.MEADOW_SUN].bearing == TABLE_CONFIG.bearings[F.MEADOW_SUN]
    meadow = {f.name: f for f in facts_at(TABLE_CONFIG, Place.MEADOW, NINE, True, True)}
    assert meadow[F.OAK_LOG_SHADED].bearing == {Channel.HEAT: -1.0, Channel.ENERGY: -0.6}
    assert meadow[F.OAK_LOG_SHADED].about is Place.OAK


def test_air_at_neutral_is_warm_air_not_cool() -> None:
    config = TABLE_CONFIG.model_copy(update={"air_neutral_c": 30.0})
    facts = facts_at(config, Place.MEADOW, ONE, True, False)
    assert facts[0].name is F.AIR
    assert facts[0].bearing == {Channel.HEAT: pytest.approx(0.0)}


def test_sun_below_the_fact_minimum_is_no_fact() -> None:
    config = TABLE_CONFIG.model_copy(update={"sun_fact_min": 0.9})
    got = [f.name for f in facts_at(config, Place.STREAM, ONE, True, False)]
    assert F.SUN not in got  # 0.6 at the stream
    assert F.MEADOW_SUN in got  # 1.0 in the meadow


def test_air_at_follows_the_curve_and_the_place_offsets() -> None:
    assert air_at(WorldConfig(), Place.MEADOW, ONE) == pytest.approx(30.0)
    for minute in (0, 360, 500, 780, 900, 1100, 1200, 1500):
        meadow = air_at(TABLE_CONFIG, Place.MEADOW, minute)
        for place in Place:
            offset = TABLE_CONFIG.place_air_offset[place]
            assert air_at(TABLE_CONFIG, place, minute) == pytest.approx(meadow + offset)
    assert air_at(TABLE_CONFIG, Place.MEADOW, 570) == pytest.approx(24.0)
    assert air_at(TABLE_CONFIG, Place.MEADOW, 0) == pytest.approx(18.0)  # flat before
    assert air_at(TABLE_CONFIG, Place.MEADOW, 1500) == pytest.approx(22.0)  # flat after


def test_sun_at_scales_open_sun_by_the_place() -> None:
    assert sun_at(TABLE_CONFIG, Place.MEADOW, 510) == pytest.approx(0.6)
    assert sun_at(TABLE_CONFIG, Place.STREAM, 510) == pytest.approx(0.36)
    assert sun_at(TABLE_CONFIG, Place.OAK, 510) == 0.0


def test_the_log_is_in_the_sun_exactly_from_until() -> None:
    config = WorldConfig(log_sun_from=600, log_sun_until=700)
    assert log_shaded_at(config, 599)
    assert not log_shaded_at(config, 600)
    assert not log_shaded_at(config, 699)
    assert log_shaded_at(config, 700)


# --- config validation --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "changes",
    [
        {"air_curve": [(360, 18.0)]},
        {"air_curve": [(360, 18.0), (360, 20.0)]},
        {"sun_curve": [(600, 0.2), (500, 1.0)]},
        {"sun_curve": []},
        {"start_minute": 600, "end_minute": 600},
        {"tick_minutes": 0},
        {"log_sun_from": 700, "log_sun_until": 600},
        {"paces": {"meadow-oak": 1.0, "meadow-stream": 4.0}},
        {"paces": {"meadow-oak": 1.0, "oak-meadow": 1.0, "meadow-stream": 4.0}},
        {"paces": {"meadow-oak": 1.0, "meadow-stream": 4.0, "oak-pond": 5.0}},
        {"paces": {"meadow-oak": 1.0, "meadow-stream": 4.0, "oak-oak": 5.0}},
        {"paces": {"meadow-oak": 1.0, "meadow-stream": 4.0, "oak-stream": 5.0, "x": 1.0}},
    ],
)
def test_world_config_rejects(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        WorldConfig.model_validate(changes)


def test_paces_may_name_a_pair_in_either_order() -> None:
    config = WorldConfig(paces={"oak-meadow": 1.0, "stream-meadow": 4.0, "stream-oak": 5.0})
    world = ClearingWorld(quiet().model_copy(update={"paces": config.paces}), 0, BODY, Place.OAK)
    assert world.step(Action.GO_STREAM).observation.place is Place.STREAM


# --- offers and actions -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("place", "offers"),
    [
        (Place.MEADOW, [Action.GO_OAK, Action.GO_STREAM, Action.REST, Action.LOOK]),
        (Place.OAK, [Action.GO_MEADOW, Action.GO_STREAM, Action.REST, Action.LOOK]),
        (
            Place.STREAM,
            [Action.GO_MEADOW, Action.GO_OAK, Action.DRINK, Action.REST, Action.LOOK],
        ),
    ],
)
def test_offers_per_place(place: Place, offers: list[Action]) -> None:
    assert ClearingWorld(WorldConfig(), 0, BODY, place).observe().offers == offers


@pytest.mark.parametrize(
    ("place", "action"),
    [(Place.MEADOW, Action.GO_MEADOW), (Place.MEADOW, Action.DRINK), (Place.OAK, Action.DRINK)],
)
def test_an_action_not_offered_raises(place: Place, action: Action) -> None:
    world = ClearingWorld(WorldConfig(), 0, BODY, place)
    before = world.snapshot()
    with pytest.raises(ValueError):
        world.step(action)
    assert world.snapshot() == before


def test_clearing_world_satisfies_the_protocol() -> None:
    world: World = ClearingWorld(WorldConfig(), 0, BODY, Place.MEADOW)
    assert not world.done


# --- the run as a whole -------------------------------------------------------------------------

SCRIPT_CYCLE = [
    Action.LOOK,
    Action.GO_OAK,
    Action.REST,
    Action.REST,
    Action.GO_STREAM,
    Action.DRINK,
    Action.DRINK,
    Action.GO_MEADOW,
    Action.REST,
]


def run_script(world: ClearingWorld, n: int | None = None) -> list[Outcome]:
    """Follow SCRIPT_CYCLE, skipping what is not offered, until the day ends or n steps."""
    out: list[Outcome] = []
    i = 0
    while not world.done and (n is None or len(out) < n):
        action = SCRIPT_CYCLE[i % len(SCRIPT_CYCLE)]
        i += 1
        if action in world.observe().offers:
            out.append(world.step(action))
    return out


def new_world(seed: int, config: WorldConfig | None = None) -> ClearingWorld:
    return ClearingWorld(config or WorldConfig(), seed, BODY, Place.MEADOW)


def test_the_same_seed_and_script_give_the_same_outcomes() -> None:
    assert run_script(new_world(3)) == run_script(new_world(3))


def test_a_different_seed_gives_a_different_signal_sequence() -> None:
    a = [o.observation.signal for o in run_script(new_world(3))]
    b = [o.observation.signal for o in run_script(new_world(4))]
    assert a != b


def test_stream_full_depends_only_on_the_seed() -> None:
    values = set()
    for seed in range(20):
        a = ClearingWorld(WorldConfig(), seed, BODY, Place.MEADOW).snapshot().stream_full
        other_body = BodyState(heat=0.5, water=0.2, energy=0.9)
        b = ClearingWorld(WorldConfig(start_minute=780), seed, other_body, Place.STREAM)
        assert b.snapshot().stream_full == a
        values.add(a)
    assert values == {True, False}


def test_observe_before_any_step_is_start() -> None:
    config = WorldConfig()
    world = ClearingWorld(config, 5, BODY, Place.OAK)
    obs = world.observe()
    assert obs.event is Event.START
    assert obs.step == 0
    assert obs.minute == config.start_minute
    assert obs.place is Place.OAK
    assert obs.last_action is None
    full = world.snapshot().stream_full
    assert obs.facts == facts_at(config, Place.OAK, config.start_minute, full, looked=False)


def test_the_signal_is_a_multiple_of_the_signal_step() -> None:
    config = WorldConfig(
        body=BodyParams(
            signal_step={Channel.HEAT: 0.07, Channel.WATER: 0.03, Channel.ENERGY: 0.11},
            signal_noise={Channel.HEAT: 0.05, Channel.WATER: 0.05, Channel.ENERGY: 0.05},
        )
    )
    world = new_world(1, config)
    observations = [world.observe()] + [o.observation for o in run_script(world)]
    for obs in observations:
        for c in Channel:
            ratio = obs.signal.of(c) / config.body.signal_step[c]
            assert ratio == pytest.approx(round(ratio), abs=1e-9)


def test_the_signal_is_the_true_value_plus_noise() -> None:
    config = quiet(signal_step={c: 1e-9 for c in Channel}, water_delay=0)
    world = ClearingWorld(config, 0, BODY, Place.MEADOW)
    obs = world.observe()
    assert obs.signal.heat == pytest.approx(BODY.heat, abs=1e-6)
    assert obs.signal.water == pytest.approx(BODY.water, abs=1e-6)
    noisy = config.model_copy(
        update={"body": config.body.model_copy(update={"signal_noise": {c: 0.1 for c in Channel}})}
    )
    signal = ClearingWorld(noisy, 0, BODY, Place.MEADOW).observe().signal
    assert signal.heat != pytest.approx(BODY.heat, abs=1e-6)


def test_step_after_the_end_raises_world_done() -> None:
    config = WorldConfig(start_minute=1140, end_minute=1200, tick_minutes=30)
    world = ClearingWorld(config, 0, BODY, Place.MEADOW)
    world.step(Action.REST)
    assert not world.done
    last = world.step(Action.REST)
    assert world.done
    assert last.observation.minute == 1200
    with pytest.raises(WorldDone):
        world.step(Action.REST)


def test_snapshot_and_restore_mid_run_continue_identically() -> None:
    config = WorldConfig()
    whole = run_script(new_world(7, config))
    actions = [o.observation.last_action for o in whole]
    cut = len(whole) // 2
    world = new_world(7, config)
    first = [world.step(a) for a in actions[:cut]]  # type: ignore[arg-type]
    state = WorldState.model_validate_json(world.snapshot().model_dump_json())
    restored = ClearingWorld.restore(config, state)
    assert restored.observe() == world.observe()
    rest = [restored.step(a) for a in actions[cut:]]  # type: ignore[arg-type]
    assert first + rest == whole


# --- one world step, rule by rule ---------------------------------------------------------------


def test_walk_moves_and_costs_energy_by_pace() -> None:
    config = quiet(walk_energy=0.01, hot_walk_factor=0.5)
    world = world_at(config, body=BODY.model_copy(update={"heat": -0.4}))
    out = world.step(Action.GO_STREAM)
    assert out.observation.event is Event.ARRIVED
    assert out.observation.place is Place.STREAM
    assert out.info.body.energy == pytest.approx(0.7 - 0.01 * 4.0)


def test_walk_costs_more_when_hot_and_heats_by_open_sun() -> None:
    config = quiet(walk_energy=0.01, hot_walk_factor=0.5, walk_heat=0.03)
    world = world_at(
        config, place=Place.OAK, minute=510, body=BODY.model_copy(update={"heat": 0.4})
    )
    out = world.step(Action.GO_STREAM)
    assert out.info.body.energy == pytest.approx(0.7 - 0.01 * 5.0 * (1 + 0.5 * 0.4))
    # the meadow's open sun at the minute the walk starts (0.6 at 08:30), whatever the places
    assert out.info.body.heat == pytest.approx(0.4 + 0.03 * 5.0 * 0.6)


def test_a_tired_body_stumbles_and_stays() -> None:
    config = quiet(walk_energy=0.01, stumble_energy=0.08, stumble_cost=0.005)
    world = world_at(config, body=BODY.model_copy(update={"energy": 0.079}))
    out = world.step(Action.GO_OAK)
    assert out.observation.event is Event.STUMBLED
    assert out.observation.place is Place.MEADOW
    assert out.info.stumbled
    assert out.info.body.energy == pytest.approx(0.079 - 0.005)
    world = world_at(config, body=BODY.model_copy(update={"energy": 0.08}))
    out = world.step(Action.GO_OAK)
    assert out.observation.event is Event.ARRIVED
    assert not out.info.stumbled


@pytest.mark.parametrize(
    ("place", "minute", "gain"),
    [
        (Place.OAK, NINE, 0.04),  # the log in shade
        (Place.OAK, ONE, 0.04 * 0.5),  # the log in the sun: no better than the ground
        (Place.MEADOW, NINE, 0.04 * 0.5),
        (Place.STREAM, NINE, 0.04 * 0.5),
    ],
)
def test_rest_quality(place: Place, minute: int, gain: float) -> None:
    config = quiet(rest_gain=0.04, ground_rest_quality=0.5)
    out = world_at(config, place=place, minute=minute).step(Action.REST)
    assert out.observation.event is Event.RESTED
    assert out.info.body.energy == pytest.approx(0.7 + gain)


@pytest.mark.parametrize(("full", "amount"), [(True, 0.25), (False, 0.25 * 0.35)])
def test_drink_full_against_trickle(full: bool, amount: float) -> None:
    config = quiet(drink_amount=0.25, absorb_rate=0.4).model_copy(update={"trickle_factor": 0.35})
    out = world_at(config, place=Place.STREAM, stream_full=full).step(Action.DRINK)
    assert out.observation.event is Event.DRANK
    assert out.info.body.gut == pytest.approx(amount * 0.6)
    assert out.info.body.water == pytest.approx(0.7 + amount * 0.4)


def test_absorption_over_steps() -> None:
    config = quiet(absorb_rate=0.3)
    world = world_at(config, body=BODY.model_copy(update={"water": 0.2, "gut": 0.4}))
    for n in range(1, 6):
        body = world.step(Action.REST).info.body
        assert body.gut == pytest.approx(0.4 * 0.7**n)
        assert body.water == pytest.approx(0.2 + 0.4 * (1 - 0.7**n))


@pytest.mark.parametrize(("heat", "sweats"), [(0.5, True), (0.3, False), (0.1, False)])
def test_sweat_above_the_threshold_only(heat: float, sweats: bool) -> None:
    config = quiet(sweat_threshold=0.3, sweat_rate=0.5, sweat_cooling=0.2, sweat_water=0.03)
    out = world_at(config, body=BODY.model_copy(update={"heat": heat})).step(Action.LOOK)
    s = 0.5 * (heat - 0.3) if sweats else 0.0
    assert out.info.body.heat == pytest.approx(heat - 0.2 * s)
    assert out.info.body.water == pytest.approx(0.7 - 0.03 * s)


def test_heat_moves_toward_the_place_target() -> None:
    config = quiet(heat_rate=0.25, air_gain=0.5, sun_gain=0.5)
    out = world_at(config, minute=ONE - 10).step(Action.REST)
    air = (air_at(config, Place.MEADOW, ONE) - 25.0) / 5.0
    target = 0.5 * air + 0.5 * sun_at(config, Place.MEADOW, ONE)
    assert out.info.body.heat == pytest.approx(0.25 * target)
    assert out.info.minute == ONE
    assert out.info.air_c == pytest.approx(air_at(config, Place.MEADOW, ONE))


def test_drain_and_base_energy() -> None:
    config = quiet(water_drain=0.004, base_energy=0.006)
    out = world_at(config).step(Action.LOOK)
    assert out.observation.event is Event.LOOKED
    assert out.info.body.water == pytest.approx(0.7 - 0.004)
    assert out.info.body.energy == pytest.approx(0.7 - 0.006)


def test_look_shows_the_log_until_the_next_step() -> None:
    world = world_at(quiet())
    looked = world.step(Action.LOOK).observation
    assert F.OAK_LOG_SHADED in [f.name for f in looked.facts]
    assert world.observe() == looked
    after = world.step(Action.REST).observation
    assert F.OAK_SHADE in [f.name for f in after.facts]


# --- the body over time -------------------------------------------------------------------------


def test_heat_now_thirst_later() -> None:
    config = WorldConfig(
        start_minute=780,
        body=BodyParams(
            signal_noise={c: 0.0 for c in Channel},
            signal_step={c: 1e-9 for c in Channel},
        ),
    )
    sun = ClearingWorld(config, 0, BODY, Place.MEADOW)
    shade = ClearingWorld(config, 0, BODY, Place.OAK)
    sun_out = [sun.step(Action.REST) for _ in range(12)]
    shade_out = [shade.step(Action.REST) for _ in range(12)]
    true_diff = [
        i
        for i, (a, b) in enumerate(zip(sun_out, shade_out, strict=True), start=1)
        if a.info.body.water != pytest.approx(b.info.body.water, abs=1e-12)
    ]
    signal_diff = [
        i
        for i, (a, b) in enumerate(zip(sun_out, shade_out, strict=True), start=1)
        if a.observation.signal.water != pytest.approx(b.observation.signal.water, abs=1e-6)
    ]
    assert true_diff and true_diff[0] <= 6
    assert sun_out[5].info.body.water < shade_out[5].info.body.water
    assert signal_diff[0] == true_diff[0] + config.body.water_delay


def test_a_drink_raises_true_water_over_three_steps() -> None:
    world = ClearingWorld(WorldConfig(), 0, BODY.model_copy(update={"water": 0.4}), Place.STREAM)
    waters = [world.step(Action.DRINK).info.body.water]
    waters += [world.step(Action.REST).info.body.water for _ in range(2)]
    assert 0.4 < waters[0] < waters[1] < waters[2]


@pytest.mark.parametrize("chosen", [Action.LOOK, Action.GO_OAK, Action.REST])
def test_reaching_a_limit_forces_rest_then_releases(chosen: Action) -> None:
    config = quiet(
        base_energy=0.01,
        energy_floor=0.03,
        rest_gain=0.1,
        ground_rest_quality=0.5,
        collapse_ticks=3,
    )
    world = world_at(config, body=BODY.model_copy(update={"energy": 0.035}))
    first = world.step(Action.LOOK)
    assert first.observation.event is Event.LOOKED
    assert not first.info.collapsed
    for _ in range(3):
        out = world.step(chosen)
        assert out.observation.event is Event.COLLAPSED
        assert out.info.collapsed
        assert out.observation.place is Place.MEADOW
        assert out.observation.offers == [Action.GO_OAK, Action.GO_STREAM, Action.REST, Action.LOOK]
    assert out.info.body.energy == pytest.approx(0.035 - 0.01 + 3 * (0.05 - 0.01))
    released = world.step(Action.LOOK)
    assert released.observation.event is Event.LOOKED
    assert not released.info.collapsed


def test_forced_rest_still_refuses_an_action_not_offered() -> None:
    config = quiet(base_energy=0.01, energy_floor=0.03)
    world = world_at(config, body=BODY.model_copy(update={"energy": 0.035}))
    world.step(Action.REST)
    with pytest.raises(ValueError):
        world.step(Action.DRINK)


@pytest.mark.parametrize(
    "body",
    [
        BodyState(heat=0.0, water=0.0, energy=0.7),
    ],
)
def test_a_body_held_at_a_limit_still_ends_the_day_on_time(body: BodyState) -> None:
    config = WorldConfig()
    world = ClearingWorld(config, 0, body, Place.MEADOW)
    outcomes = []
    while not world.done:
        outcomes.append(
            world.step(Action.GO_OAK if world.observe().place is Place.MEADOW else Action.GO_MEADOW)
        )
    assert len(outcomes) == (config.end_minute - config.start_minute) // config.tick_minutes
    assert outcomes[-1].observation.minute == config.end_minute
    assert all(o.info.collapsed for o in outcomes[1:])
    assert all(o.observation.event is Event.COLLAPSED for o in outcomes[1:])


def test_a_hot_limit_collapses_too() -> None:
    config = quiet(heat_hot_limit=1.2)
    world = world_at(config, body=BODY.model_copy(update={"heat": 1.5}))
    out = world.step(Action.REST)
    assert out.info.body.heat == pytest.approx(1.2)
    assert world.step(Action.REST).observation.event is Event.COLLAPSED
