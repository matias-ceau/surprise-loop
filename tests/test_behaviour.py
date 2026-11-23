"""Gate 1: the world is alive on the stand-ins, over seeds 1 to 20 of the two-maps experiment.

The four checks and their thresholds are fixed by the plan's section "Gates". Every run is made
once, in memory, and shared between the checks. `pytest -s` also shows a table of the 20-seed
means of a few measures per creature; only its shape is asserted.
"""

import asyncio
import math
from statistics import fmean, stdev

import pytest
from creatures import run_config

from surprise_loop.config import Backends
from surprise_loop.loop import run
from surprise_loop.metrics import RunMetrics, summarize

SEEDS = tuple(range(1, 21))
CREATURES = ("plain", "wary", "hopeful", "dull")
BLIND = "plain on the blind chooser"
TABLE = (
    "tiles_tried",
    "false_harm",
    "sickness_arrivals",
    "uncomfortable_total",
    "own_error",
    "surprised_share",
)

Runs = dict[str, list[RunMetrics]]


def _measured(agent: str, seed: int, **changes: object) -> RunMetrics:
    return summarize(asyncio.run(run(run_config(agent, seed, **changes))))


@pytest.fixture(scope="module")
def runs() -> Runs:
    """Each creature on the stand-ins, and the plain one on the blind chooser, by seed."""
    measured = {name: [_measured(name, seed) for seed in SEEDS] for name in CREATURES}
    blind = Backends(deliberate="blind")
    measured[BLIND] = [_measured("plain", seed, backends=blind) for seed in SEEDS]
    return measured


def mean(runs: Runs, name: str, field: str) -> float:
    return fmean(getattr(m, field) for m in runs[name])


def test_g1a_the_plain_creature_is_surprised_some_of_the_time(runs: Runs) -> None:
    assert 0.10 <= mean(runs, "plain", "surprised_share") <= 0.70


def test_g1b_the_reasoner_closes_at_least_half_of_the_plain_creatures_error(runs: Runs) -> None:
    assert mean(runs, "plain", "reasoner_error") <= 0.5 * mean(runs, "plain", "start_error")


def test_g1c_looking_ahead_keeps_the_plain_creature_comfortable(runs: Runs) -> None:
    rule = mean(runs, "plain", "uncomfortable_total")
    assert rule <= 0.8 * mean(runs, BLIND, "uncomfortable_total")


def test_g1d_the_hopeful_creature_tries_more_tiles_than_the_wary_one(runs: Runs) -> None:
    diffs = [
        hopeful.tiles_tried - wary.tiles_tried
        for hopeful, wary in zip(runs["hopeful"], runs["wary"], strict=True)
    ]
    standard_error = stdev(diffs) / math.sqrt(len(diffs))
    assert fmean(diffs) > 2 * standard_error


def test_the_table_of_means(runs: Runs) -> None:
    """The creatures' 20-seed means: one row per creature, a finite number in each of the six
    columns. Printed with `pytest -s`."""
    table = {name: {field: mean(runs, name, field) for field in TABLE} for name in CREATURES}
    assert list(table) == list(CREATURES)
    for row in table.values():
        assert list(row) == list(TABLE)
        assert all(isinstance(v, float) and math.isfinite(v) for v in row.values())

    width = max(len(name) for name in ("creature", *CREATURES))
    print()
    print(f"{'creature':<{width}}  " + "  ".join(f"{field:>19}" for field in TABLE))
    for name, row in table.items():
        cells = "  ".join(f"{row[field]:>19.4f}" for field in TABLE)
        print(f"{name:<{width}}  {cells}")
