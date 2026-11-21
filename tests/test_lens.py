"""The senses: ladder steps, blur, gain and noise, and what `sense` returns."""

import random
from statistics import fmean

import pytest

from surprise_loop.config import LensParams
from surprise_loop.lens import felt_levels, ladder_step, near_step, sense
from surprise_loop.state import Temperament
from surprise_loop.world import Channel, Kind, Signal

EDGES = LensParams().edges


def temperament(
    gain: dict[Channel, float] | None = None, noise: float = 0.0, blur: float = 0.0
) -> Temperament:
    return Temperament(
        presumes=dict.fromkeys(Kind, 1 / len(Kind)),
        gain=dict.fromkeys(Channel, 1.0) | (gain or {}),
        noise=dict.fromkeys(Channel, noise),
        blur=blur,
    )


def body(**levels: float) -> dict[Channel, float]:
    return dict.fromkeys(Channel, 0.0) | {Channel(c): v for c, v in levels.items()}


@pytest.mark.parametrize(
    ("level", "step"), [(0.0, 0), (0.1, 1), (0.29, 1), (0.3, 2), (0.55, 3), (0.8, 4), (1.0, 4)]
)
def test_ladder_step_with_the_default_edges(level: float, step: int) -> None:
    assert ladder_step(level, EDGES) == step


def test_near_step_caps_at_two() -> None:
    assert [near_step(n) for n in range(4)] == [0, 1, 2, 2]


def test_plain_temperament_leaves_the_body_unchanged() -> None:
    b = body(hunger=0.6, thirst=0.2, ache=0.3, sickness=0.1, unease=0.9, pleasure=0.4)
    assert felt_levels(b, temperament(), random.Random(0)) == b


def test_full_blur_averages_within_each_group() -> None:
    b = body(hunger=0.6, thirst=0.2, ache=0.3, pleasure=0.7)
    felt = felt_levels(b, temperament(blur=1.0), random.Random(0))
    assert felt[Channel.HUNGER] == pytest.approx(0.4)
    assert felt[Channel.THIRST] == pytest.approx(0.4)
    for c in (Channel.ACHE, Channel.SICKNESS, Channel.UNEASE):
        assert felt[c] == pytest.approx(0.1)
    assert felt[Channel.PLEASURE] == pytest.approx(0.7)


def test_gain_scales_and_the_result_is_clamped() -> None:
    felt = felt_levels(body(ache=0.5), temperament({Channel.ACHE: 1.4}), random.Random(0))
    assert felt[Channel.ACHE] == pytest.approx(0.7)
    felt = felt_levels(body(ache=0.9), temperament({Channel.ACHE: 1.4}), random.Random(0))
    assert felt[Channel.ACHE] == 1.0
    noisy = temperament(noise=5.0)
    draws = [felt_levels(body(), noisy, random.Random(seed)) for seed in range(20)]
    assert all(0.0 <= v <= 1.0 for d in draws for v in d.values())
    assert any(v == 0.0 for d in draws for v in d.values())
    assert any(v == 1.0 for d in draws for v in d.values())


def test_blur_comes_before_gain() -> None:
    t = temperament({Channel.HUNGER: 2.0}, blur=1.0)
    felt = felt_levels(body(hunger=0.6, thirst=0.2), t, random.Random(0))
    assert felt[Channel.HUNGER] == pytest.approx(0.8)


def test_noise_is_seeded_and_unbiased() -> None:
    t = temperament(noise=0.1)
    b = dict.fromkeys(Channel, 0.5)
    assert felt_levels(b, t, random.Random(7)) == felt_levels(b, t, random.Random(7))
    rng = random.Random(3)
    draws = [felt_levels(b, t, rng) for _ in range(200)]
    for c in Channel:
        assert fmean(d[c] for d in draws) == pytest.approx(0.5, abs=0.03)


def test_one_draw_per_channel_even_without_noise() -> None:
    rng = random.Random(11)
    felt_levels(body(), temperament(), rng)
    expected = random.Random(11)
    for _ in Channel:
        expected.gauss(0, 1)
    assert rng.random() == expected.random()


def test_sense_returns_ladder_steps_and_passes_smell_and_damp_through() -> None:
    signal = Signal(
        moved=True,
        body=body(hunger=0.6, thirst=0.2, ache=0.05, pleasure=0.9),
        smell=3,
        damp=1,
        position=(0, 0),
    )
    t = temperament({Channel.HUNGER: 1.5}, blur=0.0)
    levels, smell, damp = sense(signal, t, random.Random(0), LensParams())
    assert levels == {
        Channel.HUNGER: 4,
        Channel.THIRST: 1,
        Channel.ACHE: 0,
        Channel.SICKNESS: 0,
        Channel.UNEASE: 0,
        Channel.PLEASURE: 4,
    }
    assert (smell, damp) == (2, 1)
    blurred = temperament(blur=1.0, noise=0.0)
    assert sense(signal, blurred, random.Random(0), LensParams())[1:] == (2, 1)
