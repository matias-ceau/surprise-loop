import pytest
from pydantic import ValidationError

from surprise_loop.state import Temperament
from surprise_loop.world import (
    FACT_ABOUT,
    FACT_KIND,
    GO_TARGET,
    QUESTION_OF,
    SERVES,
    Action,
    Channel,
    FactKind,
    FactName,
    Levels,
    OpenQuestion,
    Place,
    WorldConfig,
    frozen_noon_facts,
)


def temperament(**changes: object) -> Temperament:
    fields: dict[str, object] = {
        "setpoint": Levels(heat=-0.1, water=0.8, energy=0.7),
        "tightness": {Channel.HEAT: 3.0, Channel.WATER: 6.0, Channel.ENERGY: 6.0},
        "trust": {Channel.HEAT: 0.5, Channel.WATER: 0.3, Channel.ENERGY: 0.5},
        "habit": {FactKind.BURDEN: 0.6, FactKind.RELIEF: 0.2, FactKind.AMBIENT: 0.1},
        "baseline": -0.1,
    }
    return Temperament.model_validate(fields | changes)


def test_frozen_noon_facts_are_the_owners_example() -> None:
    facts = frozen_noon_facts()
    assert [(f.name, f.kind, f.about, f.bearing) for f in facts] == [
        (FactName.AIR, FactKind.BURDEN, None, {Channel.HEAT: pytest.approx(1.0)}),
        (FactName.SUN, FactKind.BURDEN, None, {Channel.HEAT: pytest.approx(1.0)}),
        (FactName.NO_SEAT, FactKind.BURDEN, None, {Channel.ENERGY: 1.0}),
        (FactName.OAK_SHADE, FactKind.RELIEF, Place.OAK, {Channel.HEAT: -1.0}),
        (FactName.STREAM, FactKind.RELIEF, Place.STREAM, {Channel.WATER: -1.0, Channel.HEAT: -0.3}),
        (FactName.BREEZE, FactKind.RELIEF, None, {Channel.HEAT: -0.3}),
        (FactName.BIRDSONG, FactKind.AMBIENT, None, {}),
    ]


def test_frozen_noon_facts_follow_the_config_defaults() -> None:
    config = WorldConfig()
    by_name = {f.name: f for f in frozen_noon_facts()}
    for name in (FactName.NO_SEAT, FactName.OAK_SHADE, FactName.STREAM, FactName.BREEZE):
        assert by_name[name].bearing == config.bearings[name]
    computed = {FactName.AIR, FactName.COOL_AIR, FactName.SUN}
    assert set(config.bearings) == set(FactName) - computed
    assert config.bearings[FactName.OAK_SHADE_FAR] == {Channel.HEAT: -1.0}


def test_fact_names_are_in_canonical_order() -> None:
    assert [n.value for n in FactName] == [
        "air",
        "cool_air",
        "sun",
        "shade",
        "no_seat",
        "log",
        "log_sun",
        "oak_shade",
        "oak_shade_far",
        "oak_log_shaded",
        "oak_log_sunlit",
        "water",
        "trickle",
        "stream",
        "meadow_sun",
        "breeze",
        "birdsong",
    ]


def test_levels_of_and_replace() -> None:
    levels = Levels(heat=0.5, water=0.6, energy=0.7)
    assert [levels.of(c) for c in Channel] == [0.5, 0.6, 0.7]
    replaced = levels.replace(Channel.WATER, 0.1)
    assert replaced == Levels(heat=0.5, water=0.1, energy=0.7)
    assert levels.water == 0.6  # the original is unchanged


def test_action_tables() -> None:
    assert GO_TARGET == {
        Action.GO_MEADOW: Place.MEADOW,
        Action.GO_OAK: Place.OAK,
        Action.GO_STREAM: Place.STREAM,
    }
    assert SERVES == {
        Action.GO_MEADOW: Channel.HEAT,
        Action.GO_OAK: Channel.HEAT,
        Action.GO_STREAM: Channel.WATER,
        Action.DRINK: Channel.WATER,
        Action.REST: Channel.ENERGY,
        Action.LOOK: None,
    }
    assert QUESTION_OF == {
        Action.GO_MEADOW: None,
        Action.GO_OAK: OpenQuestion.LOG_SHADED,
        Action.GO_STREAM: OpenQuestion.STREAM_FULL,
        Action.DRINK: None,
        Action.REST: None,
        Action.LOOK: None,
    }


def test_fact_tables() -> None:
    kinds = dict(FACT_KIND)
    burden, relief, ambient = FactKind.BURDEN, FactKind.RELIEF, FactKind.AMBIENT
    assert kinds == {
        FactName.AIR: burden,
        FactName.COOL_AIR: relief,
        FactName.SUN: burden,
        FactName.SHADE: relief,
        FactName.NO_SEAT: burden,
        FactName.LOG: relief,
        FactName.LOG_SUN: burden,
        FactName.OAK_SHADE: relief,
        FactName.OAK_SHADE_FAR: relief,
        FactName.OAK_LOG_SHADED: relief,
        FactName.OAK_LOG_SUNLIT: relief,
        FactName.WATER: relief,
        FactName.TRICKLE: relief,
        FactName.STREAM: relief,
        FactName.MEADOW_SUN: ambient,
        FactName.BREEZE: relief,
        FactName.BIRDSONG: ambient,
    }
    assert FACT_ABOUT == {
        FactName.AIR: None,
        FactName.COOL_AIR: None,
        FactName.SUN: None,
        FactName.SHADE: None,
        FactName.NO_SEAT: None,
        FactName.LOG: None,
        FactName.LOG_SUN: None,
        FactName.OAK_SHADE: Place.OAK,
        FactName.OAK_SHADE_FAR: Place.OAK,
        FactName.OAK_LOG_SHADED: Place.OAK,
        FactName.OAK_LOG_SUNLIT: Place.OAK,
        FactName.WATER: None,
        FactName.TRICKLE: None,
        FactName.STREAM: Place.STREAM,
        FactName.MEADOW_SUN: Place.MEADOW,
        FactName.BREEZE: None,
        FactName.BIRDSONG: None,
    }


def test_temperament_accepts_a_full_body() -> None:
    assert temperament().tightness[Channel.HEAT] == 3.0


@pytest.mark.parametrize(
    "changes",
    [
        {"tightness": {Channel.HEAT: 3.0, Channel.WATER: 6.0}},
        {"trust": {Channel.HEAT: 0.5, Channel.ENERGY: 0.5}},
        {"habit": {FactKind.BURDEN: 0.6, FactKind.RELIEF: 0.2}},
        {"tightness": {Channel.HEAT: 0.0, Channel.WATER: 6.0, Channel.ENERGY: 6.0}},
    ],
    ids=["missing tightness", "missing trust", "missing habit kind", "zero tightness"],
)
def test_temperament_rejects_incomplete_or_slack_bodies(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        temperament(**changes)
