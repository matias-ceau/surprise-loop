"""The behavioural gate (build step 4): the stub agent behaves like an operant subject.

Every run uses the `experiments/trait_grid.yaml` schedule and agents, stub backends, the default
`Params`, no run directory, and the same 20 fixed seeds. The runs are computed once per module.

Baseline gate, per run unless stated: more active than inactive presses in the first FR phase;
PR response rate lower in the last quarter of the phase than in the first; a breakpoint above 0,
censored in at most 2 of 20 runs; extinction response rate lower in the last third than in the
first; extinction persistence censored in at most 2 of 20 runs; a goal revision in at least 10
runs and no goal priority at a clamp at the end of any run; a two-pass tick in at least 10 runs,
and two-pass ticks under 25% of the ticks of every phase; 300 steps of steady FR1 give at most
one goal revision.

Trait gate: for each trait, the high profile differs from the low one in the stated direction by
more than two standard errors of the seed-paired difference, and breakpoint and persistence are
each censored in at most 4 of 20 runs for every low and high profile.
"""

import asyncio
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import pytest

from surprise_loop.config import ExperimentConfig, RunConfig, load_config
from surprise_loop.loop import RunResult, run
from surprise_loop.metrics import RunMetrics, flatten, summarize
from surprise_loop.state import PassRecord
from surprise_loop.world import Action, PhaseKind, PhaseSpec

GRID = Path(__file__).resolve().parents[1] / "experiments" / "trait_grid.yaml"
SEEDS = list(range(1, 21))
BASELINE = "baseline"
PRESSES = (Action.PRESS_A, Action.PRESS_B)

# (trait, metric, +1 if the high profile must score higher, -1 if lower)
CONTRASTS = [
    ("reward_sensitivity", "breakpoint", +1),
    ("surprise_learning_rate", "extinction_persistence", -1),
    ("mood_inertia", "extinction_persistence", +1),
]


@dataclass(frozen=True)
class Run:
    result: RunResult
    metrics: RunMetrics
    passes: list[PassRecord]


def _experiment() -> ExperimentConfig:
    return load_config(GRID).model_copy(update={"seeds": SEEDS})


def _execute(configs: list[RunConfig]) -> list[Run]:
    async def all_runs() -> list[RunResult]:
        return [await run(config) for config in configs]

    return [
        Run(result=r, metrics=summarize(r), passes=flatten(r.records))
        for r in asyncio.run(all_runs())
    ]


@pytest.fixture(scope="module")
def runs() -> dict[str, list[Run]]:
    """Every agent of the grid over the 20 seeds, in seed order."""
    experiment = _experiment()
    configs = experiment.runs()
    by_agent: dict[str, list[Run]] = {}
    for config, r in zip(configs, _execute(configs), strict=True):
        by_agent.setdefault(config.agent.name, []).append(r)
    return by_agent


@pytest.fixture(scope="module")
def baseline(runs: dict[str, list[Run]]) -> list[Run]:
    return runs[BASELINE]


def _phase(r: Run, kind: PhaseKind) -> list[PassRecord]:
    """The passes of the first phase of `kind` in the run."""
    index = next(
        (p.outcome.info.phase_index for p in r.passes if p.outcome.info.phase_kind == kind), None
    )
    assert index is not None, f"the run never reached a {kind.value} phase"
    return [p for p in r.passes if p.outcome.info.phase_index == index]


def _rate(passes: list[PassRecord]) -> float:
    return sum(1 for p in passes if p.outcome.info.active_press) / len(passes)


# --- the grid itself ----------------------------------------------------------------------------


def test_grid_has_the_documented_shape() -> None:
    config = load_config(GRID)
    kinds = [phase.kind for phase in config.world.schedule]
    assert kinds == [
        PhaseKind.FIXED_RATIO,
        PhaseKind.PROGRESSIVE_RATIO,
        PhaseKind.FIXED_RATIO,
        PhaseKind.EXTINCTION,
    ]
    assert config.ticks is None
    assert len(config.seeds) == 5
    assert set(config.backends.model_dump().values()) == {"stub"}
    names = [a.name for a in config.agents]
    assert names[0] == BASELINE
    base = config.agents[0].traits.model_dump()
    for trait, _, _ in CONTRASTS:
        low, high = (
            next(a for a in config.agents if a.name == f"{trait}_{level}").traits.model_dump()
            for level in ("low", "high")
        )
        assert low[trait] < base[trait] < high[trait]
        # each profile differs from the baseline in its own trait only
        assert {k: v for k, v in low.items() if k != trait} == {
            k: v for k, v in base.items() if k != trait
        }
        assert {k: v for k, v in high.items() if k != trait} == {
            k: v for k, v in base.items() if k != trait
        }
    assert len(names) == 1 + 2 * len(CONTRASTS)


def test_every_run_completes_the_schedule(runs: dict[str, list[Run]]) -> None:
    steps = sum(phase.steps for phase in load_config(GRID).world.schedule)
    for agent_runs in runs.values():
        for r in agent_runs:
            assert not r.metrics.truncated
            assert r.metrics.steps == steps


# --- baseline gate ------------------------------------------------------------------------------


def test_active_presses_exceed_inactive_presses_in_the_first_fr_phase(baseline: list[Run]) -> None:
    for r in baseline:
        first = [p for p in r.passes if p.outcome.info.phase_index == 0]
        assert first[0].outcome.info.phase_kind == PhaseKind.FIXED_RATIO
        active = sum(1 for p in first if p.outcome.info.active_press)
        inactive = sum(
            1 for p in first if p.decision.action in PRESSES and not p.outcome.info.active_press
        )
        assert active > inactive, (r.result.config.seed, active, inactive)


def test_pr_response_rate_falls_from_the_first_to_the_last_quarter(baseline: list[Run]) -> None:
    for r in baseline:
        pr = _phase(r, PhaseKind.PROGRESSIVE_RATIO)
        quarter = len(pr) // 4
        assert quarter > 0
        assert _rate(pr[-quarter:]) < _rate(pr[:quarter]), r.result.config.seed


def test_breakpoint_is_positive_and_rarely_censored(baseline: list[Run]) -> None:
    assert all(r.metrics.breakpoint > 0 for r in baseline)
    assert sum(r.metrics.breakpoint_censored for r in baseline) <= 2


def test_extinction_response_rate_falls_from_the_first_to_the_last_third(
    baseline: list[Run],
) -> None:
    for r in baseline:
        ext = _phase(r, PhaseKind.EXTINCTION)
        third = len(ext) // 3
        assert third > 0
        assert _rate(ext[-third:]) < _rate(ext[:third]), r.result.config.seed


def test_extinction_persistence_is_rarely_censored(baseline: list[Run]) -> None:
    assert sum(r.metrics.extinction_censored for r in baseline) <= 2


def test_goals_are_revised_but_never_end_at_a_clamp(baseline: list[Run]) -> None:
    assert sum(r.metrics.goal_revisions >= 1 for r in baseline) >= 10
    for r in baseline:
        bounds = r.result.config.params.goals
        for goal, priority in r.result.agent.goals.priorities.items():
            assert bounds.min_priority < priority < bounds.max_priority, (
                r.result.config.seed,
                goal,
                priority,
            )


def test_reentry_happens_but_stays_rare_in_every_phase(baseline: list[Run]) -> None:
    assert sum(r.metrics.reentries >= 1 for r in baseline) >= 10
    for r in baseline:
        ticks: dict[int, int] = {}
        two_pass: dict[int, int] = {}
        for record in r.result.records:
            phase = record.passes[0].outcome.info.phase_index
            ticks[phase] = ticks.get(phase, 0) + 1
            two_pass[phase] = two_pass.get(phase, 0) + (len(record.passes) == 2)
        for phase, n in ticks.items():
            assert two_pass[phase] / n < 0.25, (r.result.config.seed, phase, two_pass[phase], n)


def test_steady_fr1_gives_at_most_one_goal_revision() -> None:
    experiment = _experiment()
    world = experiment.world.model_copy(
        update={"schedule": [PhaseSpec(kind=PhaseKind.FIXED_RATIO, steps=300, ratio=1)]}
    )
    configs = experiment.model_copy(update={"world": world}).runs()
    configs = [c for c in configs if c.agent.name == BASELINE]
    assert len(configs) == len(SEEDS)
    for r in _execute(configs):
        assert r.metrics.steps == 300
        assert r.metrics.goal_revisions <= 1, r.result.config.seed


# --- trait gate ---------------------------------------------------------------------------------


@pytest.mark.parametrize(("trait", "metric", "direction"), CONTRASTS)
def test_trait_contrast_exceeds_two_paired_standard_errors(
    runs: dict[str, list[Run]], trait: str, metric: str, direction: int
) -> None:
    low, high = runs[f"{trait}_low"], runs[f"{trait}_high"]
    assert [r.result.config.seed for r in low] == [r.result.config.seed for r in high] == SEEDS
    diffs = [
        direction * (getattr(h.metrics, metric) - getattr(lo.metrics, metric))
        for lo, h in zip(low, high, strict=True)
    ]
    mean = statistics.fmean(diffs)
    se = statistics.stdev(diffs) / math.sqrt(len(diffs))
    assert mean > 2 * se, (mean, se)
    assert mean > 0


@pytest.mark.parametrize("trait", [trait for trait, _, _ in CONTRASTS])
@pytest.mark.parametrize("level", ["low", "high"])
def test_trait_profiles_are_rarely_censored(
    runs: dict[str, list[Run]], trait: str, level: str
) -> None:
    profile = runs[f"{trait}_{level}"]
    assert sum(r.metrics.breakpoint_censored for r in profile) <= 4
    assert sum(r.metrics.extinction_censored for r in profile) <= 4
