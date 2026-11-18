"""Measures of a run, computed from its records.

Everything here is pure: no I/O, no randomness, inputs are never mutated. `passes` is always the
flattened `PassRecord`s of a run in step order (see `flatten`): each pass is one world step. The
result model is flat, so it maps directly onto a CSV row. Every measure is defined for a run with
no passes (counts and shares are 0) and for a body that is collapsed all day.

Comfort is read from the true body (`outcome.info.body`) under the run's temperament, never from
the belief: a channel is uncomfortable after a step when its need has magnitude at least
`params.metrics.comfort_felt` (heat either way; water and energy only as a deficit).

The CSV writers at the end turn a list of `RunResult`s into the experiment's tables. They are pure
and deterministic: rows follow the given results (the CLI passes them in config order, all seeds
of the first agent first), columns follow the model fields, and every cell is written by
`format_cell`, so the same results always give the same bytes.
"""

import csv
import io
from collections.abc import Iterable, Sequence
from enum import Enum

from surprise_loop.feeling import needs
from surprise_loop.loop import RunResult
from surprise_loop.phases.heuristics import reactive_choice
from surprise_loop.state import FeltScene, PassRecord, TickRecord
from surprise_loop.world import Action, Channel, ClearingWorld, Frozen, Levels


class RunMetrics(Frozen):
    steps: int
    ticks: int
    truncated: bool
    uncomfortable_heat: int
    uncomfortable_water: int
    uncomfortable_energy: int
    collapsed_steps: int
    stumbles: int
    n_go_meadow: int
    n_go_oak: int
    n_go_stream: int
    n_drink: int
    n_rest: int
    n_look: int
    mean_size: float
    band_hits_heat: float
    band_hits_water: float
    band_hits_energy: float
    world_expectations: int
    world_held: int
    overrides: int
    # among the passes offering at least two actions, the share whose action is the reactive
    # rule's choice (0.0 when there is none): with a single offer the rule cannot disagree
    rule_agreement: float
    final_valence: float
    mean_arousal: float


def flatten(records: list[TickRecord]) -> list[PassRecord]:
    """All passes in tick order, then pass order: that is world-step order."""
    return [p for record in records for p in record.passes]


def _share(hits: int, total: int) -> float:
    return hits / total if total else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _true_body(p: PassRecord) -> Levels:
    body = p.outcome.info.body
    return Levels(heat=body.heat, water=body.water, energy=body.energy)


def summarize(result: RunResult) -> RunMetrics:
    """The run's measures as one flat, CSV-ready row."""
    passes = flatten(result.records)
    config = result.config
    t, comfort = config.agent.temperament, config.params.metrics.comfort_felt
    n = len(passes)

    true_needs = [needs(_true_body(p), t) for p in passes]

    def uncomfortable(c: Channel) -> int:
        return sum(1 for need in true_needs if abs(need[c]) >= comfort)

    def band_hits(c: Channel) -> float:
        return _share(sum(1 for p in passes if p.comparison.body[c] == 0), n)

    def count(action: Action) -> int:
        return sum(1 for p in passes if p.decision.action is action)

    with_choice = [p for p in passes if len(p.offers) >= 2]
    agreeing = sum(
        1
        for p in with_choice
        if p.decision.action is reactive_choice(p.scene, p.offers, config.params.deliberate)
    )
    compared = [p.comparison.world for p in passes if p.comparison.world is not None]
    return RunMetrics(
        steps=n,
        ticks=len(result.records),
        truncated=not ClearingWorld.restore(config.world, result.world).done,
        uncomfortable_heat=uncomfortable(Channel.HEAT),
        uncomfortable_water=uncomfortable(Channel.WATER),
        uncomfortable_energy=uncomfortable(Channel.ENERGY),
        collapsed_steps=sum(1 for p in passes if p.outcome.info.collapsed),
        stumbles=sum(1 for p in passes if p.outcome.info.stumbled),
        n_go_meadow=count(Action.GO_MEADOW),
        n_go_oak=count(Action.GO_OAK),
        n_go_stream=count(Action.GO_STREAM),
        n_drink=count(Action.DRINK),
        n_rest=count(Action.REST),
        n_look=count(Action.LOOK),
        mean_size=_mean([p.comparison.size for p in passes]),
        band_hits_heat=band_hits(Channel.HEAT),
        band_hits_water=band_hits(Channel.WATER),
        band_hits_energy=band_hits(Channel.ENERGY),
        world_expectations=len(compared),
        world_held=sum(1 for mismatch in compared if not mismatch),
        overrides=sum(1 for p in passes if p.override is not None),
        rule_agreement=_share(agreeing, len(with_choice)),
        final_valence=result.agent.feeling.valence,
        mean_arousal=_mean([p.feeling_after.arousal for p in passes]),
    )


def _noticed_signs(scene: FeltScene) -> dict[str, str]:
    return {f.name: f.sign for f in scene.facts if f.noticed}


def opposite_value_steps(a: RunResult, b: RunResult) -> int:
    """The steps (passes paired by index, up to the shorter run) at which some fact is noticed in
    both scenes and valued "+" in one and "-" in the other."""
    count = 0
    for pa, pb in zip(flatten(a.records), flatten(b.records), strict=False):
        signs_a, signs_b = _noticed_signs(pa.scene), _noticed_signs(pb.scene)
        if any({signs_a[name], signs_b[name]} == {"+", "-"} for name in signs_a.keys() & signs_b):
            count += 1
    return count


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

# The `RunMetrics` fields whose between-agent stability `stability_csv` reports: every
# behavioural measure. Left out: `steps` and `ticks` (set by the length of the day and the cap,
# and by overrides for steps, so not a disposition of the agent) and `truncated` (a flag that
# qualifies the other measures rather than measuring behaviour).
STABILITY_METRICS: tuple[str, ...] = tuple(
    f for f in RunMetrics.model_fields if f not in {"steps", "ticks", "truncated"}
)

Cell = str | int | float | bool | Enum | None


def format_cell(value: Cell) -> str:
    """One CSV cell: floats by `repr` (shortest round-trip form; nan as "nan"), bools as
    True/False, enums by value, None as an empty cell, ints and strings as they are."""
    if value is None:
        return ""
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
    """One row per run: agent, seed, then every `RunMetrics` field."""
    names = list(RunMetrics.model_fields)
    rows = []
    for result in results:
        metrics = summarize(result)
        rows.append([*_key(result), *(getattr(metrics, m) for m in names)])
    return _csv(["agent", "seed", *names], rows)


# per channel: the comparison's body error, the belief after the pass, the true body after it
TRACE_COLUMNS: tuple[str, ...] = (
    "agent",
    "seed",
    "step",
    "minute",
    "place",
    "action",
    "event",
    *(f"error_{c.value}" for c in Channel),
    "world",
    "better",
    "size",
    *(f"belief_{c.value}" for c in Channel),
    *(f"body_{c.value}" for c in Channel),
    "valence",
    "arousal",
)


def _trace_row(result: RunResult, p: PassRecord) -> list[Cell]:
    info, c = p.outcome.info, p.comparison
    body = _true_body(p)
    return [
        *_key(result),
        info.step,
        info.minute,
        p.scene.place,  # where the decision was made
        p.decision.action,
        p.outcome.observation.event,
        *(c.body[ch] for ch in Channel),
        c.world,
        c.better,
        c.size,
        *(p.belief_after.of(ch) for ch in Channel),
        *(body.of(ch) for ch in Channel),
        p.feeling_after.valence,
        p.feeling_after.arousal,
    ]


def trace_csv(results: Sequence[RunResult]) -> str:
    """One row per world step per run. `step` and `minute` are the world's after the step; the
    comparison's `world` is empty when there was nothing to compare."""
    rows = [_trace_row(result, p) for result in results for p in flatten(result.records)]
    return _csv(TRACE_COLUMNS, rows)


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


def _agents(results: Sequence[RunResult]) -> list[str]:
    """The agents' names in order of first appearance."""
    return list(dict.fromkeys(result.config.agent.name for result in results))


def pairs_csv(results: Sequence[RunResult]) -> str:
    """For results of exactly two agents: one row per seed (in the first agent's order) with
    `opposite_value_steps` of the two runs of that seed, agents in order of first appearance."""
    agent_a, agent_b = _agents(results)
    run = {(result.config.agent.name, result.config.seed): result for result in results}
    seeds = [result.config.seed for result in results if result.config.agent.name == agent_a]
    rows = [
        [seed, agent_a, agent_b, opposite_value_steps(run[agent_a, seed], run[agent_b, seed])]
        for seed in seeds
    ]
    return _csv(["seed", "agent_a", "agent_b", "opposite_value_steps"], rows)


TABLE_FILES: tuple[str, ...] = ("results.csv", "trace.csv", "stability.csv", "pairs.csv")


def experiment_tables(results: Sequence[RunResult]) -> dict[str, str]:
    """The experiment's CSV files by file name: results.csv and trace.csv; stability.csv when the
    results hold at least two seeds (with one, no ICC is defined); pairs.csv when they hold
    exactly two agents."""
    tables = {"results.csv": results_csv(results), "trace.csv": trace_csv(results)}
    if len({result.config.seed for result in results}) >= 2:
        tables["stability.csv"] = stability_csv(results)
    if len(_agents(results)) == 2:
        tables["pairs.csv"] = pairs_csv(results)
    return tables
