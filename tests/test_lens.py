import re

import pytest
from bodies import (
    HOTTER,
    ILO,
    ILO_AFTERNOON,
    ILO_AFTERNOON_AROUSAL,
    ILO_TEXT,
    TAVI,
    TAVI_AFTERNOON,
    TAVI_AFTERNOON_AROUSAL,
    TAVI_TEXT,
)

from surprise_loop.config import LensParams
from surprise_loop.lens import felt_scene
from surprise_loop.phrases import FACT_PHRASES, WORLD_SURPRISE_PHRASES
from surprise_loop.state import FeltScene, SurpriseNote
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

P = LensParams()


def ilo_scene(**changes: object) -> FeltScene:
    args: dict[str, object] = {
        "place": Place.MEADOW,
        "facts": frozen_noon_facts(),
        "belief": ILO_AFTERNOON,
        "t": ILO,
        "arousal": ILO_AFTERNOON_AROUSAL,
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
        "belief": TAVI_AFTERNOON,
        "t": TAVI,
        "arousal": TAVI_AFTERNOON_AROUSAL,
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
        scene = ilo_scene(belief=ILO_AFTERNOON.replace(Channel.HEAT, heat))
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


def test_cool_air_reads_cool_for_a_hot_body_and_bad_for_a_chilled_one() -> None:
    cool_air = Fact(
        name=FactName.COOL_AIR, kind=FactKind.RELIEF, about=None, bearing={Channel.HEAT: -1.4}
    )
    tight = ILO.tightness[Channel.HEAT]
    hot = ILO_AFTERNOON.replace(Channel.HEAT, ILO.setpoint.heat + 2.0 / tight)
    chilled = ILO_AFTERNOON.replace(Channel.HEAT, ILO.setpoint.heat - 2.0 / tight)

    (for_hot,) = ilo_scene(facts=[cool_air], belief=hot).facts
    assert for_hot.sign == "+"
    assert for_hot.phrase is not None
    assert not re.search(r"\b(warm|hot)", for_hot.phrase, re.IGNORECASE)

    (for_chilled,) = ilo_scene(facts=[cool_air], belief=chilled).facts
    assert for_chilled.sign == "-"
    assert for_chilled.phrase in FACT_PHRASES[FactName.COOL_AIR]["-"]


def test_no_cool_air_phrase_says_warm_or_hot() -> None:
    table = FACT_PHRASES[FactName.COOL_AIR]
    for phrase in [*table["+"], *table["-"], table["0"]]:
        assert not re.search(r"\b(warm|hot)", phrase, re.IGNORECASE), phrase


def test_the_far_oak_never_sounds_near() -> None:
    table = FACT_PHRASES[FactName.OAK_SHADE_FAR]
    for phrase in [*table["+"], *table["-"], table["0"]]:
        for word in ("a few steps", "close", "far", "across", "steps"):
            assert word not in phrase.lower(), (word, phrase)


# --- body sentences -----------------------------------------------------------------------------

# Tavi at grade one (felt 0.75, between the first two body edges) on one channel, the others at
# the set point; arousal low enough for every channel to have room
GRADE_ONE = 0.75


def tavi_at(channel: Channel, felt: float) -> Levels:
    """Tavi's set point with `channel` at signed felt strength `felt`."""
    offset = felt / TAVI.tightness[channel]
    level = TAVI.setpoint.of(channel) + (offset if channel is Channel.HEAT else -offset)
    return TAVI.setpoint.replace(channel, level)


def body_phrases(scene: FeltScene) -> list[str]:
    return [line.phrase for line in scene.body]


@pytest.mark.parametrize(("felt", "sentence"), [(-GRADE_ONE, "You are a little cool."),
                                                (GRADE_ONE, "You are a little warm.")])  # fmt: skip
def test_heat_at_grade_one_is_told(felt: float, sentence: str) -> None:
    scene = tavi_scene(belief=tavi_at(Channel.HEAT, felt))
    assert [line.grade for line in scene.body] == [1 if felt > 0 else -1]
    assert body_phrases(scene) == [sentence]
    assert sentence in scene.text()


@pytest.mark.parametrize("channel", [Channel.WATER, Channel.ENERGY])
def test_thirst_or_tiredness_at_grade_one_is_not_told(channel: Channel) -> None:
    scene = tavi_scene(belief=tavi_at(channel, GRADE_ONE))
    assert scene.felt[channel] == pytest.approx(GRADE_ONE)
    assert body_phrases(scene) == []


def test_heat_speaking_from_grade_two_gives_the_old_behaviour() -> None:
    p = LensParams(body_speak_grade={Channel.HEAT: 2, Channel.WATER: 2, Channel.ENERGY: 2})
    for felt in (-GRADE_ONE, GRADE_ONE):
        assert body_phrases(tavi_scene(belief=tavi_at(Channel.HEAT, felt), p=p)) == []
    assert body_phrases(tavi_scene(belief=tavi_at(Channel.HEAT, 1.5), p=p)) == [
        "You are warm under your skin."
    ]
