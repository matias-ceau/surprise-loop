import pytest

from surprise_loop.config import LensParams
from surprise_loop.lens import felt_scene
from surprise_loop.phrases import FACT_PHRASES, WORLD_SURPRISE_PHRASES
from surprise_loop.state import FeltScene, SurpriseNote, Temperament
from surprise_loop.world import (
    Channel,
    Fact,
    FactKind,
    FactName,
    Levels,
    OpenQuestion,
    Place,
    frozen_noon_facts,
)

ILO = Temperament(
    setpoint=Levels(heat=-0.1, water=0.80, energy=0.70),
    tightness={Channel.HEAT: 3.0, Channel.WATER: 6.0, Channel.ENERGY: 6.0},
    trust={Channel.HEAT: 0.5, Channel.WATER: 0.3, Channel.ENERGY: 0.5},
    habit={FactKind.BURDEN: 0.6, FactKind.RELIEF: 0.2, FactKind.AMBIENT: 0.1},
    baseline=-0.1,
)
TAVI = Temperament(
    setpoint=Levels(heat=0.2, water=0.80, energy=0.70),
    tightness={Channel.HEAT: 2.5, Channel.WATER: 4.0, Channel.ENERGY: 4.0},
    trust={Channel.HEAT: 0.5, Channel.WATER: 0.5, Channel.ENERGY: 0.5},
    habit={FactKind.BURDEN: 0.1, FactKind.RELIEF: 0.1, FactKind.AMBIENT: 0.6},
    baseline=0.1,
)
ILO_BELIEF = Levels(heat=0.74, water=0.58, energy=0.25)
TAVI_BELIEF = Levels(heat=0.0, water=0.85, energy=0.75)
HOTTER = SurpriseNote(channel=Channel.HEAT, worse=True)
P = LensParams()

ILO_TEXT = (
    "You are standing in the middle of the forest clearing. There is no place to sit, and you "
    "need one. It is really hot. The sun beats down on your head. The one dark patch is under "
    "the oak, a few steps off. Your legs are heavy. Your head is thick. It is hotter than you "
    "thought it would be."
)
TAVI_TEXT = (
    "You are standing in the middle of the forest clearing. The air is pleasantly warm. The sun "
    "sits on your shoulders. It looks dim under the oak. Birds are calling. A stream runs "
    "somewhere to the west. A light breeze moves the leaves."
)


def ilo_scene(**changes: object) -> FeltScene:
    args: dict[str, object] = {
        "place": Place.MEADOW,
        "facts": frozen_noon_facts(),
        "belief": ILO_BELIEF,
        "t": ILO,
        "arousal": 0.70,
        "attention": None,
        "voiced": HOTTER,
        "extra": 0,
        "p": P,
    }
    return felt_scene(**(args | changes))  # type: ignore[arg-type]


def tavi_scene(**changes: object) -> FeltScene:
    args: dict[str, object] = {
        "place": Place.MEADOW,
        "facts": frozen_noon_facts(),
        "belief": TAVI_BELIEF,
        "t": TAVI,
        "arousal": 0.20,
        "attention": None,
        "voiced": None,
        "extra": 0,
        "p": P,
    }
    return felt_scene(**(args | changes))  # type: ignore[arg-type]


def noticed(scene: FeltScene) -> list[FactName]:
    return [f.name for f in scene.facts if f.noticed]


def by_name(scene: FeltScene) -> dict[FactName, int]:
    """Signed grade per fact."""
    signs = {"+": 1, "-": -1, "0": 0}
    return {f.name: signs[f.sign] * f.grade for f in scene.facts}


def test_ilos_fixed_text() -> None:
    assert ilo_scene().text() == ILO_TEXT


def test_tavis_fixed_text() -> None:
    assert tavi_scene().text() == TAVI_TEXT


def test_ilo_notices_four_facts() -> None:
    assert noticed(ilo_scene()) == [
        FactName.NO_SEAT,
        FactName.AIR,
        FactName.SUN,
        FactName.OAK_SHADE,
    ]


def test_tavi_notices_six_facts_and_not_the_missing_seat() -> None:
    names = noticed(tavi_scene())
    assert len(names) == 6
    assert FactName.NO_SEAT not in names


def test_air_sun_and_oak_shade_have_opposite_signs_for_the_two_bodies() -> None:
    ilo, tavi = by_name(ilo_scene()), by_name(tavi_scene())
    for name in (FactName.AIR, FactName.SUN, FactName.OAK_SHADE):
        assert ilo[name] * tavi[name] < 0


def test_scene_holds_every_fact_heaviest_first() -> None:
    scene = tavi_scene()
    assert sorted(f.name for f in scene.facts) == sorted(f.name for f in frozen_noon_facts())
    weights = [f.weight for f in scene.facts]
    assert weights == sorted(weights, reverse=True)
    assert sum(weights) == pytest.approx(1.0)
    for fact in scene.facts:
        assert (fact.phrase is None) == (not fact.noticed)


def test_felt_strengths_are_carried_raw() -> None:
    felt = ilo_scene().felt
    assert felt[Channel.HEAT] == pytest.approx(2.52)
    assert felt[Channel.ENERGY] == pytest.approx(2.70)


def test_extra_adds_exactly_one_noticed_fact() -> None:
    assert len(noticed(ilo_scene(extra=1))) == len(noticed(ilo_scene())) + 1


def test_attention_raises_the_weight_of_facts_bearing_on_that_channel() -> None:
    def stream_weight(scene: FeltScene) -> float:
        return next(f.weight for f in scene.facts if f.name is FactName.STREAM)

    assert stream_weight(tavi_scene(attention=Channel.WATER)) > stream_weight(tavi_scene())


def test_the_count_noticed_is_capped_by_the_list_length() -> None:
    two = frozen_noon_facts()[:2]
    assert len(noticed(ilo_scene(facts=two, arousal=1.0))) == 2


def test_signed_grades_move_one_way_as_believed_heat_rises() -> None:
    watched = (FactName.AIR, FactName.SUN, FactName.OAK_SHADE)
    history: dict[FactName, list[int]] = {name: [] for name in watched}
    steps = round((1.2 - -0.5) / 0.05)
    for i in range(steps + 1):
        heat = -0.5 + i * 0.05
        scene = ilo_scene(belief=ILO_BELIEF.replace(Channel.HEAT, heat))
        grades = by_name(scene)
        for name in watched:
            history[name].append(grades[name])
    for name in watched:
        series = history[name]
        rising = series[-1] >= series[0]
        pairs = zip(series, series[1:], strict=False)
        assert all((b >= a) if rising else (b <= a) for a, b in pairs), (name, series)
        assert series[0] != series[-1]


def test_a_sign_with_no_phrases_falls_back_to_neutral() -> None:
    # The stream has no phrases of its own for a negative value: a cold body hears it neutrally.
    assert "-" not in FACT_PHRASES[FactName.STREAM]
    cold = Levels(heat=-0.1 - 3.0, water=0.80, energy=0.70)
    stream = Fact(
        name=FactName.STREAM,
        kind=FactKind.RELIEF,
        about=Place.STREAM,
        bearing={Channel.WATER: -1.0, Channel.HEAT: -0.3},
    )
    scene = ilo_scene(facts=[stream], belief=cold)
    (fact,) = scene.facts
    assert fact.sign == "-"
    assert fact.phrase == FACT_PHRASES[FactName.STREAM]["0"]


def test_a_world_surprise_is_voiced() -> None:
    note = SurpriseNote(question=OpenQuestion.STREAM_FULL, actual=False)
    scene = tavi_scene(voiced=note)
    assert scene.surprise == WORLD_SURPRISE_PHRASES[(OpenQuestion.STREAM_FULL, False)]
    assert scene.text().endswith(scene.surprise)
