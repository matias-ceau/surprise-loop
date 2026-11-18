import itertools
import math

import pytest
from bodies import ILO, ILO_AFTERNOON, TAVI, TAVI_AFTERNOON

from surprise_loop.config import FeelingParams
from surprise_loop.feeling import discomfort, feel, felt, grade, needs
from surprise_loop.state import Comparison, Feeling
from surprise_loop.world import Channel, Levels

EDGES = (0.5, 1.0, 2.0, 3.0)
P = FeelingParams()


def calm(arousal: float = 0.0, phasic: float = 0.0) -> Feeling:
    return Feeling(discomfort=0.0, valence=0.0, arousal=arousal, mood=0.0, phasic=phasic)


def comparison(better: float = 0.0, size: float = 0.0) -> Comparison:
    zero = {c: 0.0 for c in Channel}
    return Comparison(body=zero, body_felt=zero, world=None, better=better, size=size)


def test_felt_for_ilos_afternoon_start() -> None:
    f = felt(ILO_AFTERNOON, ILO)
    assert f[Channel.HEAT] == pytest.approx(2.52, abs=1e-9)
    assert f[Channel.WATER] == pytest.approx(1.32, abs=1e-9)
    assert f[Channel.ENERGY] == pytest.approx(2.70, abs=1e-9)


def test_felt_for_tavis_afternoon_start() -> None:
    f = felt(TAVI_AFTERNOON, TAVI)
    assert f[Channel.HEAT] == pytest.approx(-0.5, abs=1e-9)
    assert f[Channel.WATER] == pytest.approx(-0.2, abs=1e-9)
    assert f[Channel.ENERGY] == pytest.approx(-0.2, abs=1e-9)


def test_needs_clip_water_and_energy_but_not_heat() -> None:
    n = needs(TAVI_AFTERNOON, TAVI)
    assert n[Channel.HEAT] == pytest.approx(-0.5, abs=1e-9)
    assert n[Channel.WATER] == 0.0
    assert n[Channel.ENERGY] == 0.0


def test_discomfort_is_the_sum_of_absolute_needs() -> None:
    assert discomfort(ILO_AFTERNOON, ILO) == pytest.approx(2.52 + 1.32 + 2.70, abs=1e-9)
    assert discomfort(TAVI_AFTERNOON, TAVI) == pytest.approx(0.5, abs=1e-9)


def test_grade_counts_the_edges_the_magnitude_reaches() -> None:
    assert grade(2.52, EDGES) == 3
    assert grade(0.5, EDGES) == 1
    assert grade(-3.0, EDGES) == 4
    assert grade(0.49, EDGES) == 0


def test_feel_without_comparison_is_tone_alone() -> None:
    d = discomfort(ILO_AFTERNOON, ILO)
    out = feel(ILO_AFTERNOON, ILO, calm(), None, P)
    assert out.discomfort == pytest.approx(d)
    assert out.phasic == 0.0
    assert out.valence == pytest.approx(ILO.baseline - math.tanh(P.tone_gain * d))


def test_a_better_outcome_raises_valence_and_then_decays() -> None:
    plain = feel(TAVI_AFTERNOON, TAVI, calm(), None, P)
    lifted = feel(TAVI_AFTERNOON, TAVI, calm(), comparison(better=1.0), P)
    assert lifted.phasic == pytest.approx(P.phasic_gain * math.tanh(1.0))
    assert lifted.valence == pytest.approx(plain.valence + lifted.phasic)
    after = feel(TAVI_AFTERNOON, TAVI, lifted, None, P)
    assert after.phasic == pytest.approx(P.phasic_decay * lifted.phasic)
    assert plain.valence < after.valence < lifted.valence


def test_arousal_moves_toward_its_target_by_one_minus_inertia() -> None:
    size = 0.4
    strain = min(1.0, 2.70 / P.strain_full)  # Ilo's loudest need is tiredness
    target = min(1.0, P.surprise_gain * size + P.strain_gain * strain)
    out = feel(ILO_AFTERNOON, ILO, calm(arousal=0.1), comparison(size=size), P)
    assert out.arousal == pytest.approx(0.1 + (1 - P.arousal_inertia) * (target - 0.1))


def test_mood_follows_valence_slowly() -> None:
    out = feel(ILO_AFTERNOON, ILO, calm(), None, P)
    assert out.mood == pytest.approx((1 - P.mood_inertia) * out.valence)


@pytest.mark.parametrize("strength", [-50.0, 50.0])
def test_feel_stays_in_range_for_extreme_strengths(strength: float) -> None:
    extreme = Levels(
        heat=ILO.setpoint.heat + strength / ILO.tightness[Channel.HEAT],
        water=ILO.setpoint.water - strength / ILO.tightness[Channel.WATER],
        energy=ILO.setpoint.energy - strength / ILO.tightness[Channel.ENERGY],
    )
    for better, size, start in itertools.product((-50.0, 50.0), (0.0, 50.0), (0.0, 1.0)):
        previous = Feeling(discomfort=0.0, valence=start, arousal=start, mood=start, phasic=start)
        out = feel(extreme, ILO, previous, comparison(better=better, size=size), P)
        assert -1 <= out.valence <= 1
        assert 0 <= out.arousal <= 1
        assert -1 <= out.mood <= 1
        assert math.isfinite(out.phasic)
