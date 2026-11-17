"""The behavioural gate: a stub day in the clearing is alive and honest under the default numbers.

Every run here is a whole day of `experiments/clearing_day.yaml` (both bodies, seeds 1 to 10) on
the rule-based stubs, with no model and no tick files. Each set of runs is computed once per test
session. A criterion that names a stub runs with that stub; the others run the experiment as its
file configures it (the `reactive` stub, the default).

The criteria (the plan's Task 10, plus the controller's liveliness criterion):

- no run has more than 6 collapsed steps, and every run visits all three places;
- `reactive`: every run has at least 3 passes with a body error, and at most half its passes;
- `reactive`: at least 3 of Ilo's 10 runs have an override, and no run has two-pass ticks on more
  than a quarter of its ticks;
- `lookahead` spends at least 10% fewer uncomfortable steps than `reactive` (mean over 20 runs);
- `blind`: Ilo's and Tavi's action counts differ by less than 15% of the steps for every action;
  `reactive`: they differ by more than that for at least one action;
- the day pair has `opposite_value_steps` at least 5 in at least 8 of the 10 seeds;
- in the afternoon experiment, seed 1, Ilo has not collapsed within the first 3 steps;
- `reactive`: every run walks at least 8 times, and no action takes more than 60% of its passes.
"""

import asyncio
import statistics
from collections.abc import Callable
from pathlib import Path

import pytest

from surprise_loop.config import ExperimentConfig, RunConfig, load_config
from surprise_loop.loop import RunResult, run
from surprise_loop.metrics import flatten, opposite_value_steps, summarize
from surprise_loop.state import PassRecord
from surprise_loop.world import Action, BodyState, Channel, Event, WorldConfig

ROOT = Path(__file__).resolve().parent.parent
DAY = ROOT / "experiments" / "clearing_day.yaml"
AFTERNOON = ROOT / "experiments" / "clearing_afternoon.yaml"
SEEDS = list(range(1, 11))
AGENTS = ("ilo", "tavi")

# the plan's thresholds
MAX_COLLAPSED = 6
MIN_ERROR_PASSES = 3
MAX_ERROR_SHARE = 0.5
MIN_ILO_OVERRIDE_RUNS = 3
MAX_TWO_PASS_SHARE = 0.25
LOOKAHEAD_GAIN = 0.10
ACTION_DIFF_SHARE = 0.15
MIN_OPPOSITE_STEPS = 5
MIN_OPPOSITE_SEEDS = 8
AFTERNOON_STEPS = 3
# the controller's liveliness criterion
MIN_WALKS = 8
MAX_ACTION_SHARE = 0.6

Runs = dict[str, list[RunResult]]  # agent name -> one result per seed, in seed order


def day_runs(stub: str | None) -> Runs:
    """The day experiment over seeds 1 to 10, with `stub` (None: as the file configures it)."""
    exp = load_config(DAY)
    params = exp.params
    if stub is not None:
        deliberate = params.deliberate.model_copy(update={"stub": stub})
        params = params.model_copy(update={"deliberate": deliberate})
    exp = exp.model_copy(update={"seeds": SEEDS, "params": params})
    results = _run_all(exp.runs())
    by_agent: Runs = {name: [] for name in AGENTS}
    for result in results:
        by_agent[result.config.agent.name].append(result)
    assert all(len(results) == len(SEEDS) for results in by_agent.values())
    return by_agent


def _run_all(configs: list[RunConfig]) -> list[RunResult]:
    async def all_runs() -> list[RunResult]:
        return list(await asyncio.gather(*(run(config) for config in configs)))

    return asyncio.run(all_runs())


@pytest.fixture(scope="module")
def configured() -> Runs:
    assert load_config(DAY).params.deliberate.stub == "reactive"
    return day_runs(None)


@pytest.fixture(scope="module")
def lookahead() -> Runs:
    return day_runs("lookahead")


@pytest.fixture(scope="module")
def blind() -> Runs:
    return day_runs("blind")


def every_run(runs: Runs) -> list[RunResult]:
    return [result for name in AGENTS for result in runs[name]]


def label(result: RunResult) -> str:
    return f"{result.config.agent.name} seed {result.config.seed}"


def passes(result: RunResult) -> list[PassRecord]:
    return flatten(result.records)


def has_body_error(p: PassRecord) -> bool:
    return any(p.comparison.body[c] != 0 for c in Channel)


def action_counts(result: RunResult) -> dict[Action, int]:
    chosen = [p.decision.action for p in passes(result)]
    return {action: chosen.count(action) for action in Action}


def uncomfortable(result: RunResult) -> int:
    m = summarize(result)
    return m.uncomfortable_heat + m.uncomfortable_water + m.uncomfortable_energy


def mean_action_differences(runs: Runs) -> dict[Action, float]:
    """Per action, the mean over seeds of |Ilo's count - Tavi's count|, as a share of the mean
    number of steps of a run."""
    pairs = list(zip(runs["ilo"], runs["tavi"], strict=True))
    steps = statistics.mean(len(passes(r)) for pair in pairs for r in pair)
    return {
        action: statistics.mean(
            abs(action_counts(a)[action] - action_counts(b)[action]) for a, b in pairs
        )
        / steps
        for action in Action
    }


def at_a_limit(body: BodyState, world: WorldConfig) -> bool:
    p = world.body
    return (
        body.heat in (p.heat_cold_limit, p.heat_hot_limit)
        or body.water <= p.water_floor
        or body.energy <= p.energy_floor
    )


def failing(runs: list[RunResult], ok: Callable[[RunResult], bool]) -> list[str]:
    return [label(result) for result in runs if not ok(result)]


# --- the gate -----------------------------------------------------------------------------------


def test_no_run_collapses_for_more_than_six_steps(configured: Runs) -> None:
    collapsed = {label(r): summarize(r).collapsed_steps for r in every_run(configured)}
    assert max(collapsed.values()) <= MAX_COLLAPSED, collapsed


def test_every_run_visits_all_three_places(configured: Runs) -> None:
    def visits_all(result: RunResult) -> bool:
        places = {result.records[0].observation.place}
        places |= {p.outcome.info.place for p in passes(result)}
        return len(places) == 3

    assert failing(every_run(configured), visits_all) == []


def test_reactive_expectations_usually_hold_and_sometimes_fail(configured: Runs) -> None:
    errors = {
        label(r): (sum(has_body_error(p) for p in passes(r)), len(passes(r)))
        for r in every_run(configured)
    }
    too_few = {k: v for k, v in errors.items() if v[0] < MIN_ERROR_PASSES}
    too_many = {k: v for k, v in errors.items() if v[0] > MAX_ERROR_SHARE * v[1]}
    assert too_few == {}
    assert too_many == {}


def test_reactive_ilo_is_overridden_in_at_least_three_runs(configured: Runs) -> None:
    overridden = [r.config.seed for r in configured["ilo"] if summarize(r).overrides > 0]
    assert len(overridden) >= MIN_ILO_OVERRIDE_RUNS, overridden


def test_reactive_runs_have_two_passes_on_at_most_a_quarter_of_ticks(configured: Runs) -> None:
    def few_two_pass(result: RunResult) -> bool:
        two = sum(1 for record in result.records if len(record.passes) == 2)
        return two <= MAX_TWO_PASS_SHARE * len(result.records)

    assert failing(every_run(configured), few_two_pass) == []


def test_looking_ahead_pays(configured: Runs, lookahead: Runs) -> None:
    reactive_mean = statistics.mean(uncomfortable(r) for r in every_run(configured))
    lookahead_mean = statistics.mean(uncomfortable(r) for r in every_run(lookahead))
    assert lookahead_mean <= (1 - LOOKAHEAD_GAIN) * reactive_mean, (reactive_mean, lookahead_mean)


def test_blind_bodies_act_alike(blind: Runs) -> None:
    differences = mean_action_differences(blind)
    assert all(d < ACTION_DIFF_SHARE for d in differences.values()), differences


def test_reactive_bodies_act_differently(configured: Runs) -> None:
    differences = mean_action_differences(configured)
    assert any(d > ACTION_DIFF_SHARE for d in differences.values()), differences


def test_the_two_bodies_value_the_same_fact_oppositely(configured: Runs) -> None:
    steps = [
        opposite_value_steps(a, b)
        for a, b in zip(configured["ilo"], configured["tavi"], strict=True)
    ]
    assert sum(n >= MIN_OPPOSITE_STEPS for n in steps) >= MIN_OPPOSITE_SEEDS, steps


def test_afternoon_ilo_does_not_collapse_at_once() -> None:
    exp = load_config(AFTERNOON)
    config = next(c for c in exp.runs() if c.agent.name == "ilo" and c.seed == 1)
    first = passes(_run_all([config])[0])[:AFTERNOON_STEPS]
    assert len(first) == AFTERNOON_STEPS
    assert not any(p.outcome.info.collapsed for p in first)
    assert not any(at_a_limit(p.outcome.info.body, config.world) for p in first)


def test_reactive_runs_walk_at_least_eight_times(configured: Runs) -> None:
    def walks(result: RunResult) -> int:
        return sum(1 for p in passes(result) if p.outcome.observation.event is Event.ARRIVED)

    counts = {label(r): walks(r) for r in every_run(configured)}
    assert min(counts.values()) >= MIN_WALKS, counts


def test_no_reactive_run_is_mostly_one_action(configured: Runs) -> None:
    shares = {
        label(r): max(action_counts(r).values()) / len(passes(r)) for r in every_run(configured)
    }
    assert max(shares.values()) <= MAX_ACTION_SHARE, shares


def test_the_gate_runs_the_day_experiment_as_written() -> None:
    exp: ExperimentConfig = load_config(DAY)
    assert [a.name for a in exp.agents] == list(AGENTS)
    assert exp.backends.appraise == exp.backends.deliberate == "stub"
    assert exp.ticks is None
