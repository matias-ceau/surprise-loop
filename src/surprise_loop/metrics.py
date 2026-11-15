"""Behavioural measures of a run, computed from its records.

Everything here is pure: no I/O, no randomness, inputs are never mutated. `passes` is always the
flattened `PassRecord`s of a run in step order (see `flatten`). The result models are flat, so
each maps directly onto a CSV row.

The quit criterion, shared by `breakpoint` and `extinction_persistence`: within one phase's steps
in order, the first run of `quit_steps` consecutive steps with no active press. Steps outside the
phase never count toward a run. A measure is censored when the criterion is never met inside the
phase (the phase or the run ended first) or the phase was never reached.

The CSV writers at the end turn a list of `RunResult`s into the experiment's four tables. They
are pure and deterministic: rows follow the given results (the CLI passes them in config order,
all seeds of the first agent first), columns follow the model fields, and every cell is written
by `format_cell`, so the same results always give the same bytes.
"""

import csv
import io
from collections.abc import Iterable, Sequence
from enum import Enum

from pydantic import BaseModel, ConfigDict

from surprise_loop.loop import RunResult
from surprise_loop.state import PassRecord, TickRecord, Traits
from surprise_loop.world import Action, Event, PhaseKind


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


class PhaseMetrics(_Frozen):
    phase_index: int
    phase_kind: PhaseKind
    steps: int
    active_presses: int
    response_rate: float
    rewards: int
    mean_abs_pe: float


class Censored(_Frozen):
    value: int
    censored: bool


class PEPoint(_Frozen):
    step: int
    tick: int
    pass_index: int
    phase_index: int
    phase_kind: PhaseKind
    raw_signed: float
    signed: float
    unsigned: float


class RunMetrics(_Frozen):
    steps: int
    ticks: int
    truncated: bool
    rewards: int
    response_rate_fr: float
    response_rate_pr: float
    response_rate_ext: float
    inactive_rate: float
    breakpoint: int
    breakpoint_censored: bool
    extinction_persistence: int
    extinction_censored: bool
    mean_abs_pe: float
    reentries: int
    goal_revisions: int


def flatten(records: list[TickRecord]) -> list[PassRecord]:
    """All passes in tick order, then pass order: that is world-step order."""
    return [p for record in records for p in record.passes]


def _is_press(p: PassRecord) -> bool:
    return p.decision.action in (Action.PRESS_A, Action.PRESS_B)


def _is_active(p: PassRecord) -> bool:
    return p.outcome.info.active_press


def _is_reinforced(p: PassRecord) -> bool:
    return p.outcome.observation.event == Event.REWARD


def _rate(hits: int, steps: int) -> float:
    return hits / steps if steps else 0.0


def _active_rate(passes: list[PassRecord]) -> float:
    return _rate(sum(1 for p in passes if _is_active(p)), len(passes))


def _mean_abs_pe(passes: list[PassRecord]) -> float:
    total = sum(abs(p.prediction_error.raw_signed) for p in passes)
    return total / len(passes) if passes else 0.0


def _in_phase(passes: list[PassRecord], phase_index: int) -> list[PassRecord]:
    return [p for p in passes if p.outcome.info.phase_index == phase_index]


def phase_table(passes: list[PassRecord]) -> list[PhaseMetrics]:
    """One row per phase the passes reach, by ascending `phase_index`."""
    rows = []
    for index in sorted({p.outcome.info.phase_index for p in passes}):
        phase = _in_phase(passes, index)
        active = sum(1 for p in phase if _is_active(p))
        rows.append(
            PhaseMetrics(
                phase_index=index,
                phase_kind=phase[0].outcome.info.phase_kind,
                steps=len(phase),
                active_presses=active,
                response_rate=_rate(active, len(phase)),
                rewards=sum(1 for p in phase if _is_reinforced(p)),
                mean_abs_pe=_mean_abs_pe(phase),
            )
        )
    return rows


def response_rate(passes: list[PassRecord], phase_index: int | None = None) -> float:
    """Active presses per step, in one phase or over all passes; 0.0 when there are no steps."""
    selected = passes if phase_index is None else _in_phase(passes, phase_index)
    return _active_rate(selected)


def _first_phase(passes: list[PassRecord], kind: PhaseKind) -> list[PassRecord] | None:
    """The passes of the first phase of `kind`, or None when the passes reach no such phase."""
    for p in passes:
        if p.outcome.info.phase_kind == kind:
            return _in_phase(passes, p.outcome.info.phase_index)
    return None


def _check_quit_steps(quit_steps: int) -> None:
    if quit_steps < 1:
        raise ValueError(f"quit_steps must be >= 1, got {quit_steps}")


def _before_quit(phase: list[PassRecord], quit_steps: int) -> tuple[list[PassRecord], bool]:
    """The phase's passes before its quit run starts, and whether the quit criterion was never met.

    The quit run is the first run of `quit_steps` consecutive passes with no active press.
    """
    run = 0
    for i, p in enumerate(phase):
        run = 0 if _is_active(p) else run + 1
        if run == quit_steps:
            return phase[: i + 1 - quit_steps], False
    return phase, True


def breakpoint(passes: list[PassRecord], quit_steps: int) -> Censored:
    """The last ratio completed in the first PR phase before the quit criterion (0 if none)."""
    _check_quit_steps(quit_steps)
    phase = _first_phase(passes, PhaseKind.PROGRESSIVE_RATIO)
    if phase is None:
        return Censored(value=0, censored=True)
    before, censored = _before_quit(phase, quit_steps)
    completed = [
        p.outcome.info.ratio_completed for p in before if p.outcome.info.ratio_completed is not None
    ]
    return Censored(value=completed[-1] if completed else 0, censored=censored)


def extinction_persistence(passes: list[PassRecord], quit_steps: int) -> Censored:
    """Active presses in the first extinction phase before the quit criterion."""
    _check_quit_steps(quit_steps)
    phase = _first_phase(passes, PhaseKind.EXTINCTION)
    if phase is None:
        return Censored(value=0, censored=True)
    before, censored = _before_quit(phase, quit_steps)
    return Censored(value=sum(1 for p in before if _is_active(p)), censored=censored)


def pe_trace(passes: list[PassRecord]) -> list[PEPoint]:
    """One point per pass, in step order."""
    return [
        PEPoint(
            step=p.outcome.info.step,
            tick=p.tick,
            pass_index=p.pass_index,
            phase_index=p.outcome.info.phase_index,
            phase_kind=p.outcome.info.phase_kind,
            raw_signed=p.prediction_error.raw_signed,
            signed=p.prediction_error.signed,
            unsigned=p.prediction_error.unsigned,
        )
        for p in passes
    ]


def summarize(result: RunResult) -> RunMetrics:
    """The run's measures as one flat, CSV-ready row."""
    passes = flatten(result.records)
    quit_steps = result.config.params.metrics.quit_steps

    def of_kind(kind: PhaseKind) -> list[PassRecord]:
        return [p for p in passes if p.outcome.info.phase_kind == kind]

    pr_indices = [p.outcome.info.phase_index for p in of_kind(PhaseKind.PROGRESSIVE_RATIO)]
    first_pr = min(pr_indices, default=None)
    fr = [
        p
        for p in of_kind(PhaseKind.FIXED_RATIO)
        if first_pr is None or p.outcome.info.phase_index < first_pr
    ]
    bp = breakpoint(passes, quit_steps)
    persistence = extinction_persistence(passes, quit_steps)
    inactive = sum(1 for p in passes if _is_press(p) and not _is_active(p))
    return RunMetrics(
        steps=len(passes),
        ticks=len(result.records),
        truncated=result.world.phase_index < len(result.config.world.schedule),
        rewards=sum(1 for p in passes if _is_reinforced(p)),
        response_rate_fr=_active_rate(fr),
        response_rate_pr=_active_rate(of_kind(PhaseKind.PROGRESSIVE_RATIO)),
        response_rate_ext=_active_rate(of_kind(PhaseKind.EXTINCTION)),
        inactive_rate=_rate(inactive, len(passes)),
        breakpoint=bp.value,
        breakpoint_censored=bp.censored,
        extinction_persistence=persistence.value,
        extinction_censored=persistence.censored,
        mean_abs_pe=_mean_abs_pe(passes),
        reentries=sum(1 for record in result.records if len(record.passes) == 2),
        goal_revisions=result.agent.goals.revisions,
    )


def icc1(groups: list[list[float]]) -> float:
    """One-way random-effects ICC(1), raw and unclipped.

    ICC = (MSB - MSW) / (MSB + (k0 - 1) * MSW) with k0 = (N - sum(k_i^2) / N) / (n - 1), which is
    the common group size when groups are balanced. nan with fewer than two groups, any group of
    fewer than two members, or zero total variance.
    """
    n = len(groups)
    if n < 2 or any(len(g) < 2 for g in groups):
        return float("nan")
    values = [x for g in groups for x in g]
    if min(values) == max(values):
        return float("nan")
    total = len(values)
    grand = sum(values) / total
    means = [sum(g) / len(g) for g in groups]
    ss_between = sum(len(g) * (m - grand) ** 2 for g, m in zip(groups, means, strict=True))
    ss_within = sum((x - m) ** 2 for g, m in zip(groups, means, strict=True) for x in g)
    ms_between = ss_between / (n - 1)
    ms_within = ss_within / (total - n)
    k0 = (total - sum(len(g) ** 2 for g in groups) / total) / (n - 1)
    return (ms_between - ms_within) / (ms_between + (k0 - 1) * ms_within)


# --- CSV tables ---------------------------------------------------------------------------------

# The `RunMetrics` fields whose between-agent stability `stability_csv` reports: every numeric
# behavioural measure. Left out: `steps` and `ticks` (set by the schedule and the cap, and by
# re-entry for ticks, so not a disposition of the agent), `truncated` and the two `*_censored`
# flags (booleans that qualify another measure rather than measure behaviour). Censored
# breakpoint and persistence values are included as they are; results.csv carries the flags.
STABILITY_METRICS: tuple[str, ...] = (
    "rewards",
    "response_rate_fr",
    "response_rate_pr",
    "response_rate_ext",
    "inactive_rate",
    "breakpoint",
    "extinction_persistence",
    "mean_abs_pe",
    "reentries",
    "goal_revisions",
)

Cell = str | int | float | bool | Enum


def format_cell(value: Cell) -> str:
    """One CSV cell: floats by `repr` (shortest round-trip form; nan as "nan"), bools as
    True/False, enums by value, ints and strings as they are."""
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _csv(columns: Sequence[str], rows: Iterable[Sequence[Cell]]) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([format_cell(v) for v in row])
    return out.getvalue()


def _key(result: RunResult) -> list[Cell]:
    return [result.config.agent.name, result.config.seed]


def results_csv(results: Sequence[RunResult]) -> str:
    """One row per run: agent, the three traits, seed, then every `RunMetrics` field."""
    trait_names = list(Traits.model_fields)
    metric_names = list(RunMetrics.model_fields)
    rows = []
    for result in results:
        traits = result.config.agent.traits
        metrics = summarize(result)
        rows.append(
            [
                result.config.agent.name,
                *(getattr(traits, t) for t in trait_names),
                result.config.seed,
                *(getattr(metrics, m) for m in metric_names),
            ]
        )
    return _csv(["agent", *trait_names, "seed", *metric_names], rows)


def phases_csv(results: Sequence[RunResult]) -> str:
    """One row per phase reached per run: agent, seed, then the `PhaseMetrics` fields."""
    names = list(PhaseMetrics.model_fields)
    rows = [
        [*_key(result), *(getattr(phase, n) for n in names)]
        for result in results
        for phase in phase_table(flatten(result.records))
    ]
    return _csv(["agent", "seed", *names], rows)


def pe_trace_csv(results: Sequence[RunResult]) -> str:
    """One row per world step per run: agent, seed, then the `PEPoint` fields."""
    names = list(PEPoint.model_fields)
    rows = [
        [*_key(result), *(getattr(point, n) for n in names)]
        for result in results
        for point in pe_trace(flatten(result.records))
    ]
    return _csv(["agent", "seed", *names], rows)


def stability_csv(results: Sequence[RunResult]) -> str:
    """ICC(1) of each `STABILITY_METRICS` measure, with seeds grouped within agent.

    Agents are grouped by name in order of first appearance, each group's values in the order
    given. `n_agents` is the number of groups, `n_seeds` the number of distinct seeds.
    """
    groups: dict[str, list[RunMetrics]] = {}
    for result in results:
        groups.setdefault(result.config.agent.name, []).append(summarize(result))
    n_seeds = len({result.config.seed for result in results})
    rows = [
        [
            name,
            icc1([[float(getattr(m, name)) for m in group] for group in groups.values()]),
            len(groups),
            n_seeds,
        ]
        for name in STABILITY_METRICS
    ]
    return _csv(["metric", "icc1", "n_agents", "n_seeds"], rows)


TABLE_FILES: tuple[str, ...] = ("results.csv", "phases.csv", "pe_trace.csv", "stability.csv")


def experiment_tables(results: Sequence[RunResult]) -> dict[str, str]:
    """The experiment's four CSV files, by file name (the names in `TABLE_FILES`)."""
    writers = (results_csv, phases_csv, pe_trace_csv, stability_csv)
    return {name: write(results) for name, write in zip(TABLE_FILES, writers, strict=True)}
