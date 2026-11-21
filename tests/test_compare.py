"""The comparison of an expectation with what was felt."""

import itertools

import pytest

from surprise_loop.config import MonitorParams
from surprise_loop.phases.monitor import compare
from surprise_loop.state import Expectation, Sureness
from surprise_loop.world import Channel

P = MonitorParams()


def levels(**steps: int) -> dict[Channel, int]:
    return dict.fromkeys(Channel, 0) | {Channel(c): s for c, s in steps.items()}


def expect(
    moved: bool = True,
    smell: int = 0,
    damp: int = 0,
    sure: Sureness = Sureness.SURE,
    **steps: int,
) -> Expectation:
    return Expectation(moved=moved, levels=levels(**steps), smell=smell, damp=damp, sure=sure)


def test_an_expectation_met_gives_no_surprise() -> None:
    e = expect(moved=False, smell=1, damp=2, hunger=2, pleasure=3)
    c = compare(e, False, levels(hunger=2, pleasure=3), 1, 2, P)
    assert c.surprise == 0
    assert c.worse == 0
    assert not c.moved_miss
    assert all(m == 0 for m in c.misses.values())
    assert (c.smell_miss, c.damp_miss) == (0, 0)


def test_a_missed_step_and_hunger() -> None:
    c = compare(expect(hunger=1), False, levels(hunger=3), 0, 0, P)
    assert c.moved_miss
    assert c.misses[Channel.HUNGER] == 2
    assert c.surprise == pytest.approx(0.75)
    assert c.worse == pytest.approx(0.5)
    guessing = compare(expect(hunger=1, sure=Sureness.GUESSING), False, levels(hunger=3), 0, 0, P)
    assert guessing.surprise == pytest.approx(0.375)
    assert guessing.worse == pytest.approx(0.5)


def test_more_pleasure_is_better() -> None:
    c = compare(expect(pleasure=1), True, levels(pleasure=3), 0, 0, P)
    assert c.worse == pytest.approx(-0.5)


def test_smell_off_by_two_alone() -> None:
    c = compare(expect(), True, levels(), 2, 0, P)
    assert c.smell_miss == 2
    assert c.surprise == pytest.approx(0.25)
    assert c.worse == 0


def test_surprise_never_exceeds_one() -> None:
    e = expect(moved=True, smell=0, damp=2)
    for high in itertools.product((0, 4), repeat=len(Channel)):
        got = dict(zip(Channel, high, strict=True))
        assert compare(e, False, got, 2, 0, P).surprise <= 1
    worst = compare(e, False, levels(**dict.fromkeys(Channel, 4)), 2, 0, P)
    assert worst.surprise == 1
