"""The summary feeling: tone from the felt levels, and stir from the surprise."""

import pytest

from surprise_loop.config import FeelingParams
from surprise_loop.feeling import next_stir, stir_step, tone_step
from surprise_loop.world import Channel

P = FeelingParams()


def levels(**steps: int) -> dict[Channel, int]:
    return dict.fromkeys(Channel, 0) | {Channel(c): s for c, s in steps.items()}


@pytest.mark.parametrize(
    ("steps", "tone"),
    [
        ({}, 2),
        ({"hunger": 1, "thirst": 1}, 2),
        ({"hunger": 2, "thirst": 2}, 1),
        ({"pleasure": 2}, 3),
        ({"pleasure": 3}, 4),
        ({"sickness": 4, "ache": 4}, 0),
    ],
)
def test_tone_step(steps: dict[str, int], tone: int) -> None:
    assert tone_step(levels(**steps), P) == tone


def test_next_stir_follows_the_surprise() -> None:
    assert next_stir(0, 1, P) == 0.5
    assert next_stir(0.5, 0, P) == 0.25


@pytest.mark.parametrize(("stir", "step"), [(0.1, 0), (0.15, 1), (0.4, 2)])
def test_stir_step(stir: float, step: int) -> None:
    assert stir_step(stir, P) == step
