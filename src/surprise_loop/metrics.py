"""Measures of a run, computed from its records.

Everything here is pure: no I/O, no randomness, inputs are never mutated. `summarize` turns a run
into one flat row, so it maps directly onto a CSV row; every measure is defined for a run with no
ticks (counts and shares are 0, the map errors are those of the starting state).

Comfort is read from the true body after each step (`outcome.signal.body`), never from what was
felt: a channel is uncomfortable after a step when its true level is at least
`params.metrics.comfort_at[channel]`. Pleasure is never a discomfort.

The CSV writers at the end turn a list of `RunResult`s into the experiment's tables. They are pure
and deterministic: rows follow the given results (the CLI passes them in config order, all seeds
of the first agent first), columns follow the model fields, and every cell is written by
`format_cell`, so the same results always give the same bytes.
"""

import csv
import io
from collections.abc import Iterable, Sequence
from enum import Enum
from itertools import combinations, pairwise
from typing import Literal

from surprise_loop.loop import RunResult
from surprise_loop.phases.heuristics import measures
from surprise_loop.reasoner import reasoner_beliefs
from surprise_loop.state import AgentState, Measures, TickRecord
from surprise_loop.world import (
    Belief,
    Channel,
    Direction,
    Frozen,
    Position,
    WorldConfig,
    key,
    layout_positions,
    neighbours,
)

DISCOMFORTS: tuple[Channel, ...] = (
    Channel.HUNGER,
    Channel.THIRST,
    Channel.ACHE,
    Channel.SICKNESS,
    Channel.UNEASE,
)


class RunMetrics(Frozen):
    steps: int
    uncomfortable_hunger: int
    uncomfortable_thirst: int
    uncomfortable_ache: int
    uncomfortable_sickness: int
    uncomfortable_unease: int
    uncomfortable_total: int
    surprised_share: float
    mean_surprise: float
    hit_moved: float
    hit_hunger: float
    hit_thirst: float
    hit_ache: float
    hit_sickness: float
    hit_unease: float
    hit_pleasure: float
    hit_smell: float
    hit_damp: float
    rule_agreement: float
    chance_agreement: float
    tiles_tried: int
    start_error: float
    own_error: float
    reasoner_error: float
    gap: float
    own_error_all: float
    reasoner_error_all: float
    gap_all: float
    false_harm: int
    false_safe: int
    reasoner_false_harm: int
    reasoner_false_safe: int
    wrong_tile_steps: int
    sickness_arrivals: int
    changes_per_step: float
    stray_changes: int


class Gate2(Frozen):
    completed: bool
    start_error: float
    own_error: float
    reasoner_error: float
    passed: bool


def _share(hits: int, total: int) -> float:
    return hits / total if total else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _last_measures(result: RunResult) -> Measures:
    """The last tick's measures; for a run with no tick, those of its starting state."""
    if result.records:
        return result.records[-1].measures
    config = result.config
    return measures(result.agent, config.world, config.params.metrics)


def _stray_changes(records: list[TickRecord], memory: int) -> int:
    """Changes to a spot that is not the tick's target, not a neighbour of the position after the
    step, and not the target of any of the previous `memory` ticks."""
    stray = 0
    for i, record in enumerate(records):
        near: set[Position] = {
            record.outcome.info.target,
            *neighbours(record.outcome.signal.position),
            *(r.outcome.info.target for r in records[max(0, i - memory) : i]),
        }
        stray += sum(1 for change in record.appraisal.changes if change.at not in near)
    return stray


def summarize(result: RunResult) -> RunMetrics:
    """The run's measures as one flat, CSV-ready row."""
    records = result.records
    config = result.config
    comfort = config.params.metrics.comfort_at
    n = len(records)

    uncomfortable = {
        c: sum(1 for r in records if r.outcome.signal.body[c] >= comfort[c]) for c in DISCOMFORTS
    }

    def hit_level(c: Channel) -> float:
        return _share(
            sum(1 for r in records if r.decision.expectation.levels[c] == r.felt.levels[c]), n
        )

    def hit(field: Literal["moved", "smell", "damp"]) -> float:
        hits = (getattr(r.decision.expectation, field) == getattr(r.felt, field) for r in records)
        return _share(sum(hits), n)

    last = _last_measures(result)
    return RunMetrics(
        steps=n,
        uncomfortable_hunger=uncomfortable[Channel.HUNGER],
        uncomfortable_thirst=uncomfortable[Channel.THIRST],
        uncomfortable_ache=uncomfortable[Channel.ACHE],
        uncomfortable_sickness=uncomfortable[Channel.SICKNESS],
        uncomfortable_unease=uncomfortable[Channel.UNEASE],
        uncomfortable_total=sum(uncomfortable.values()),
        surprised_share=_share(sum(1 for r in records if r.comparison.surprise > 0), n),
        mean_surprise=_mean([r.comparison.surprise for r in records]),
        hit_moved=hit("moved"),
        hit_hunger=hit_level(Channel.HUNGER),
        hit_thirst=hit_level(Channel.THIRST),
        hit_ache=hit_level(Channel.ACHE),
        hit_sickness=hit_level(Channel.SICKNESS),
        hit_unease=hit_level(Channel.UNEASE),
        hit_pleasure=hit_level(Channel.PLEASURE),
        hit_smell=hit("smell"),
        hit_damp=hit("damp"),
        rule_agreement=_share(sum(1 for r in records if r.decision.direction in r.rule_best), n),
        chance_agreement=_mean([len(r.rule_best) / len(Direction) for r in records]),
        tiles_tried=last.tiles_tried,
        start_error=last.start_error,
        own_error=last.own_error,
        reasoner_error=last.reasoner_error,
        gap=last.gap,
        own_error_all=last.own_error_all,
        reasoner_error_all=last.reasoner_error_all,
        gap_all=last.gap_all,
        false_harm=last.false_harm,
        false_safe=last.false_safe,
        reasoner_false_harm=last.reasoner_false_harm,
        reasoner_false_safe=last.reasoner_false_safe,
        wrong_tile_steps=sum(len(r.measures.wrong) for r in records),
        sickness_arrivals=sum(1 for r in records if r.outcome.info.sickness_arrived),
        changes_per_step=_mean([len(r.appraisal.changes) for r in records]),
        stray_changes=_stray_changes(records, max(config.world.body.sick_delays)),
    )


def _beliefs(
    agent: AgentState, at: Position, which: Literal["own", "reasoner"]
) -> dict[Belief, float]:
    if which == "own":
        return agent.own_map.at(at)
    return reasoner_beliefs(agent.reasoner, at)


def map_distance(
    a: AgentState,
    b: AgentState,
    world: WorldConfig,
    which: Literal["own", "reasoner"],
    tiles: Literal["shared", "all"] = "shared",
) -> float | None:
    """The mean absolute difference of the five beliefs, read from each creature's own map or
    from the reasoner's. With `shared`, over the layout tiles tried in both states, and None when
    there is none; with `all`, over every tile of the layout, where an untried tile has the map's
    untried beliefs or the reasoner's presumption."""
    over = layout_positions(world)
    if tiles == "shared":
        over = tuple(at for at in over if key(at) in a.tries and key(at) in b.tries)
        if not over:
            return None
    total = sum(
        abs(_beliefs(a, at, which)[belief] - _beliefs(b, at, which)[belief])
        for at in over
        for belief in Belief
    )
    return total / (len(over) * len(Belief))


def gate2(result: RunResult) -> Gate2:
    """Gate 2's marks: the run completed, and over the tiles it tried the creature's own map ended
    closer to the world than its start, closing at least half of what the reasoner closed."""
    config = result.config
    ticks = config.ticks if config.ticks is not None else config.world.steps
    completed = len(result.records) == ticks
    last = _last_measures(result)
    start, own, reasoner = last.start_error, last.own_error, last.reasoner_error
    passed = completed and own < start and (start - own) >= 0.5 * (start - reasoner)
    return Gate2(
        completed=completed,
        start_error=start,
        own_error=own,
        reasoner_error=reasoner,
        passed=passed,
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


TRACE_COLUMNS: tuple[str, ...] = (
    "agent",
    "seed",
    "tick",
    "direction",
    "moved",
    "surprise",
    "worse",
    "tiles_tried",
    "start_error",
    "own_error",
    "reasoner_error",
    "gap",
    "false_harm",
    "false_safe",
)


def _trace_row(result: RunResult, r: TickRecord) -> list[Cell]:
    m = r.measures
    return [
        *_key(result),
        r.tick,
        r.decision.direction,
        r.outcome.signal.moved,
        r.comparison.surprise,
        r.comparison.worse,
        m.tiles_tried,
        m.start_error,
        m.own_error,
        m.reasoner_error,
        m.gap,
        m.false_harm,
        m.false_safe,
    ]


def trace_csv(results: Sequence[RunResult]) -> str:
    """One row per tick per run."""
    return _csv(TRACE_COLUMNS, [_trace_row(res, r) for res in results for r in res.records])


def _agents(results: Sequence[RunResult]) -> list[str]:
    """The agents' names in order of first appearance."""
    return list(dict.fromkeys(result.config.agent.name for result in results))


def _seeds(results: Sequence[RunResult]) -> list[int]:
    """The seeds in order of first appearance."""
    return list(dict.fromkeys(result.config.seed for result in results))


DISTANCE_COLUMNS: tuple[str, ...] = (
    "kind",
    "agent_a",
    "agent_b",
    "seed_a",
    "seed_b",
    "own_distance",
    "reasoner_distance",
    "own_distance_all",
    "reasoner_distance_all",
)


def distances_csv(results: Sequence[RunResult]) -> str:
    """The distance between the final maps of two runs. `between` rows first: at each seed, every
    two agents. Then `within` rows: for each agent, each two consecutive seeds."""
    run = {(r.config.agent.name, r.config.seed): r for r in results}
    agents, seeds = _agents(results), _seeds(results)
    pairs = [
        ("between", (a, seed), (b, seed))
        for seed in seeds
        for a, b in combinations([a for a in agents if (a, seed) in run], 2)
    ] + [
        ("within", (agent, s), (agent, t))
        for agent in agents
        for s, t in pairwise([s for s in seeds if (agent, s) in run])
    ]
    rows = []
    for kind, first, second in pairs:
        a, b = run[first], run[second]
        world = a.config.world
        rows.append(
            [
                kind,
                first[0],
                second[0],
                first[1],
                second[1],
                map_distance(a.agent, b.agent, world, "own"),
                map_distance(a.agent, b.agent, world, "reasoner"),
                map_distance(a.agent, b.agent, world, "own", tiles="all"),
                map_distance(a.agent, b.agent, world, "reasoner", tiles="all"),
            ]
        )
    return _csv(DISTANCE_COLUMNS, rows)


def stability_csv(results: Sequence[RunResult]) -> str:
    """ICC(1) of each `RunMetrics` field, with seeds grouped within agent.

    Agents are grouped by name in order of first appearance, each group's values in the order
    given. `n_agents` is the number of groups, `n_seeds` the number of distinct seeds.
    """
    groups: dict[str, list[RunMetrics]] = {}
    for result in results:
        groups.setdefault(result.config.agent.name, []).append(summarize(result))
    n_seeds = len(_seeds(results))
    rows = [
        [
            name,
            icc1([[float(getattr(m, name)) for m in group] for group in groups.values()]),
            len(groups),
            n_seeds,
        ]
        for name in RunMetrics.model_fields
    ]
    return _csv(["metric", "icc1", "n_agents", "n_seeds"], rows)


TABLE_FILES: tuple[str, ...] = ("results.csv", "trace.csv", "distances.csv", "stability.csv")


def experiment_tables(results: Sequence[RunResult]) -> dict[str, str]:
    """The experiment's CSV files by file name: results.csv, trace.csv and distances.csv always;
    stability.csv when the results hold at least two seeds (with one, no ICC is defined)."""
    tables = {
        "results.csv": results_csv(results),
        "trace.csv": trace_csv(results),
        "distances.csv": distances_csv(results),
    }
    if len(_seeds(results)) >= 2:
        tables["stability.csv"] = stability_csv(results)
    return tables
