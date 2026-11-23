"""The measures of a run, the distance between two maps, gate 2, and the experiment's tables."""

import asyncio
import csv
import io
import math

import pytest
from creatures import run_config

from surprise_loop.config import Backends, initial_agent_state
from surprise_loop.loop import RunResult, run
from surprise_loop.metrics import (
    TABLE_FILES,
    RunMetrics,
    experiment_tables,
    format_cell,
    gate2,
    icc1,
    map_distance,
    summarize,
)
from surprise_loop.reasoner import reasoner_beliefs
from surprise_loop.state import AgentState, Change
from surprise_loop.world import Belief, Channel, GridWorld, key, layout_positions

DISCOMFORTS = (Channel.HUNGER, Channel.THIRST, Channel.ACHE, Channel.SICKNESS, Channel.UNEASE)


def stand_in(ticks: int, agent: str = "plain", seed: int = 1, **changes: object) -> RunResult:
    return asyncio.run(run(run_config(agent, seed, ticks=ticks, **changes)))


@pytest.fixture(scope="module")
def forty() -> RunResult:
    return stand_in(40)


# --- summarize ----------------------------------------------------------------------------------


def test_summarize_counts_what_the_records_hold(forty: RunResult) -> None:
    records = forty.records
    comfort = forty.config.params.metrics.comfort_at
    m = summarize(forty)

    assert m.steps == len(records) == 40
    uncomfortable = {
        c: sum(1 for r in records if r.outcome.signal.body[c] >= comfort[c]) for c in DISCOMFORTS
    }
    assert m.uncomfortable_hunger == uncomfortable[Channel.HUNGER]
    assert m.uncomfortable_unease == uncomfortable[Channel.UNEASE]
    assert m.uncomfortable_total == sum(uncomfortable.values())
    assert m.surprised_share == sum(1 for r in records if r.comparison.surprise > 0) / 40
    assert m.mean_surprise == pytest.approx(sum(r.comparison.surprise for r in records) / 40)
    moved_hits = sum(1 for r in records if r.decision.expectation.moved == r.felt.moved)
    assert m.hit_moved == moved_hits / 40
    assert m.hit_hunger == (
        sum(
            1
            for r in records
            if r.decision.expectation.levels[Channel.HUNGER] == r.felt.levels[Channel.HUNGER]
        )
        / 40
    )
    assert m.rule_agreement == 1.0
    last = records[-1].measures
    assert m.tiles_tried == last.tiles_tried
    assert (m.start_error, m.own_error, m.reasoner_error, m.gap) == (
        last.start_error,
        last.own_error,
        last.reasoner_error,
        last.gap,
    )
    assert (m.own_error_all, m.false_harm, m.reasoner_false_safe) == (
        last.own_error_all,
        last.false_harm,
        last.reasoner_false_safe,
    )
    assert m.wrong_tile_steps == sum(len(r.measures.wrong) for r in records)
    assert m.sickness_arrivals == sum(1 for r in records if r.outcome.info.sickness_arrived)
    assert m.changes_per_step == sum(len(r.appraisal.changes) for r in records) / 40


def test_a_stand_in_run_is_alive_enough_to_measure(forty: RunResult) -> None:
    m = summarize(forty)
    assert 0 < m.surprised_share < 1
    assert m.tiles_tried > 0
    assert m.own_error < m.start_error


def test_a_blind_run_strays_from_the_rule_and_chance_is_the_mean_share() -> None:
    result = stand_in(40, backends=Backends(deliberate="blind"))
    m = summarize(result)
    assert m.rule_agreement < 1
    chance = sum(len(r.rule_best) / 4 for r in result.records) / len(result.records)
    assert m.chance_agreement == pytest.approx(chance)


def test_the_reasoner_changes_only_tiles_it_has_reason_to() -> None:
    result = stand_in(60)
    assert summarize(result).stray_changes == 0
    assert summarize(result).changes_per_step > 0


def test_a_change_far_from_the_step_is_stray(forty: RunResult) -> None:
    records = list(forty.records)
    last = records[-1]
    far = (100, 100)
    beliefs = dict.fromkeys(Belief, 0.0)
    appraisal = last.appraisal.model_copy(
        update={"changes": (*last.appraisal.changes, Change(at=far, beliefs=beliefs))}
    )
    records[-1] = last.model_copy(update={"appraisal": appraisal})
    assert summarize(forty.model_copy(update={"records": records})).stray_changes == 1


def test_a_run_with_no_ticks_has_shares_of_zero() -> None:
    config = run_config(ticks=1)
    world = GridWorld(config.world, config.seed)
    empty = RunResult(
        config=config,
        records=[],
        agent=initial_agent_state(config, world.observe()),
        world=world.snapshot(),
    )
    m = summarize(empty)
    assert (m.steps, m.surprised_share, m.rule_agreement, m.changes_per_step) == (0, 0.0, 0.0, 0.0)
    assert m.own_error == m.start_error


# --- map_distance -------------------------------------------------------------------------------


def blank_agent() -> AgentState:
    config = run_config()
    return initial_agent_state(config, GridWorld(config.world, config.seed).observe())


def with_tiles(agent: AgentState, tiles: dict[tuple[int, int], dict[Belief, float]]) -> AgentState:
    changes = [Change(at=at, beliefs=b) for at, b in tiles.items()]
    return agent.model_copy(
        update={
            "own_map": agent.own_map.changed(changes),
            "tries": {key(at): 1 for at in tiles},
        }
    )


def test_map_distance_of_a_state_with_itself_is_zero(forty: RunResult) -> None:
    world = forty.config.world
    assert map_distance(forty.agent, forty.agent, world, "own") == 0.0
    assert map_distance(forty.agent, forty.agent, world, "reasoner") == 0.0


def test_map_distance_is_none_without_a_tile_tried_by_both() -> None:
    world = run_config().world
    a = with_tiles(blank_agent(), {(1, 0): dict.fromkeys(Belief, 0.0)})
    b = with_tiles(blank_agent(), {(0, 1): dict.fromkeys(Belief, 0.0)})
    assert map_distance(a, b, world, "own") is None
    assert map_distance(blank_agent(), blank_agent(), world, "reasoner") is None


def test_map_distance_is_the_mean_difference_over_tiles_both_tried() -> None:
    world = run_config().world
    zero, one = dict.fromkeys(Belief, 0.0), dict.fromkeys(Belief, 1.0)
    half = dict.fromkeys(Belief, 0.0) | {Belief.BLOCKS: 0.5, Belief.HURTS: 1.0}
    a = with_tiles(blank_agent(), {(1, 0): zero, (0, 1): zero, (-1, 0): zero})
    # (-1, 0) is tried by a only, (0, -1) by b only: both are ignored
    b = with_tiles(blank_agent(), {(1, 0): one, (0, 1): half, (0, -1): one})
    expected = (5 * 1.0 + 0.5 + 1.0) / (2 * len(Belief))
    assert map_distance(a, b, world, "own") == pytest.approx(expected)


def test_map_distance_reads_the_reasoners_map(forty: RunResult) -> None:
    other = stand_in(40, seed=2)
    world = forty.config.world
    layout = {key(p) for p in layout_positions(world)}
    both = sorted(forty.agent.tries.keys() & other.agent.tries.keys() & layout)
    assert both
    total = 0.0
    for k in both:
        at = tuple(int(n) for n in k.split(","))
        ra = reasoner_beliefs(forty.agent.reasoner, at)
        rb = reasoner_beliefs(other.agent.reasoner, at)
        total += sum(abs(ra[b] - rb[b]) for b in Belief)
    expected = total / (len(both) * len(Belief))
    assert map_distance(forty.agent, other.agent, world, "reasoner") == pytest.approx(expected)


def test_map_distance_over_all_tiles_reads_untried_beliefs_where_a_tile_is_untried() -> None:
    world = run_config().world
    zero, one = dict.fromkeys(Belief, 0.0), dict.fromkeys(Belief, 1.0)
    a = with_tiles(blank_agent(), {(1, 0): zero, (-1, 0): zero})
    b = with_tiles(blank_agent(), {(1, 0): one, (0, -1): one})
    untried = blank_agent().own_map.untried
    layout = layout_positions(world)
    assert {(1, 0), (-1, 0), (0, -1)} <= set(layout)
    # (1, 0) is tried by both; (-1, 0) and (0, -1) by one, against the other's untried beliefs;
    # every other tile is untried in both and adds nothing
    total = (
        len(Belief) * 1.0
        + sum(abs(0.0 - untried[x]) for x in Belief)
        + sum(abs(1.0 - untried[x]) for x in Belief)
    )
    shared, everywhere = 1.0, total / (len(layout) * len(Belief))
    assert map_distance(a, b, world, "own") == pytest.approx(shared)
    assert map_distance(a, b, world, "own", tiles="shared") == pytest.approx(shared)
    assert map_distance(a, b, world, "own", tiles="all") == pytest.approx(everywhere)
    assert everywhere != pytest.approx(shared)


def test_map_distance_over_all_tiles_is_never_none(forty: RunResult) -> None:
    world = forty.config.world
    blank = blank_agent()
    assert map_distance(blank, blank, world, "own", tiles="all") == 0.0
    assert map_distance(blank, blank, world, "reasoner", tiles="all") == 0.0
    layout = layout_positions(world)
    expected = sum(
        abs(reasoner_beliefs(forty.agent.reasoner, at)[x] - reasoner_beliefs(blank.reasoner, at)[x])
        for at in layout
        for x in Belief
    ) / (len(layout) * len(Belief))
    assert expected > 0
    got = map_distance(forty.agent, blank, world, "reasoner", tiles="all")
    assert got == pytest.approx(expected)


# --- gate 2 -------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def short() -> RunResult:
    return stand_in(5)


def with_errors(result: RunResult, start: float, own: float, reasoner: float) -> RunResult:
    records = list(result.records)
    last = records[-1]
    measures = last.measures.model_copy(
        update={"start_error": start, "own_error": own, "reasoner_error": reasoner}
    )
    records[-1] = last.model_copy(update={"measures": measures})
    return result.model_copy(update={"records": records})


def test_gate2_passes_when_the_map_closed_half_of_what_the_reasoner_closed(
    short: RunResult,
) -> None:
    g = gate2(with_errors(short, start=0.3, own=0.1, reasoner=0.05))
    assert g.completed and g.passed
    assert (g.start_error, g.own_error, g.reasoner_error) == (0.3, 0.1, 0.05)


def test_gate2_fails_when_the_map_closed_less_than_half(short: RunResult) -> None:
    g = gate2(with_errors(short, start=0.3, own=0.2, reasoner=0.05))
    assert g.completed and not g.passed


def test_gate2_fails_when_the_map_got_no_closer(short: RunResult) -> None:
    assert not gate2(with_errors(short, start=0.3, own=0.3, reasoner=0.05)).passed


def test_gate2_fails_when_the_run_stopped_short(short: RunResult) -> None:
    longer = short.config.model_copy(update={"ticks": 6})
    cut = with_errors(short, start=0.3, own=0.1, reasoner=0.05).model_copy(
        update={"config": longer}
    )
    g = gate2(cut)
    assert not g.completed and not g.passed


def test_gate2_counts_a_day_shorter_than_its_tick_cap_as_completed() -> None:
    base = run_config("plain")
    world = base.world.model_copy(update={"steps": 5})
    result = stand_in(9, world=world)
    assert len(result.records) == 5
    g = gate2(with_errors(result, start=0.3, own=0.1, reasoner=0.05))
    assert g.completed and g.passed


# --- tables -------------------------------------------------------------------------------------

AGENTS, SEEDS = ("plain", "wary"), (1, 2)


@pytest.fixture(scope="module")
def four() -> list[RunResult]:
    return [stand_in(10, agent, seed) for agent in AGENTS for seed in SEEDS]


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def test_experiment_tables_writes_the_four_files(four: list[RunResult]) -> None:
    assert set(experiment_tables(four)) == set(TABLE_FILES)


def test_results_csv_has_one_row_per_run(four: list[RunResult]) -> None:
    text = experiment_tables(four)["results.csv"]
    assert text.splitlines()[0].split(",") == ["agent", "seed", *RunMetrics.model_fields]
    table = rows(text)
    assert [(r["agent"], r["seed"]) for r in table] == [(a, str(s)) for a in AGENTS for s in SEEDS]
    for row, result in zip(table, four, strict=True):
        for field, value in summarize(result).model_dump().items():
            assert row[field] == format_cell(value)


def test_trace_csv_has_one_row_per_tick(four: list[RunResult]) -> None:
    text = experiment_tables(four)["trace.csv"]
    assert text.splitlines()[0].split(",") == [
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
    ]
    table = rows(text)
    assert len(table) == sum(len(r.records) for r in four) == 40
    first, record = table[0], four[0].records[0]
    assert (first["agent"], first["seed"], first["tick"]) == ("plain", "1", "1")
    assert first["direction"] == record.decision.direction.value
    assert first["moved"] == str(record.outcome.signal.moved)
    assert first["surprise"] == repr(record.comparison.surprise)
    assert first["own_error"] == repr(record.measures.own_error)


def test_distances_csv_pairs_agents_at_a_seed_and_seeds_of_an_agent(
    four: list[RunResult],
) -> None:
    text = experiment_tables(four)["distances.csv"]
    assert text.splitlines()[0] == (
        "kind,agent_a,agent_b,seed_a,seed_b,own_distance,reasoner_distance,"
        "own_distance_all,reasoner_distance_all"
    )
    table = rows(text)
    assert [(r["kind"], r["agent_a"], r["agent_b"], r["seed_a"], r["seed_b"]) for r in table] == [
        ("between", "plain", "wary", "1", "1"),
        ("between", "plain", "wary", "2", "2"),
        ("within", "plain", "plain", "1", "2"),
        ("within", "wary", "wary", "1", "2"),
    ]
    run = {(r.config.agent.name, r.config.seed): r for r in four}
    world = four[0].config.world
    expected = map_distance(run["plain", 1].agent, run["wary", 1].agent, world, "reasoner")
    assert table[0]["reasoner_distance"] == format_cell(expected)
    plain, wary = run["plain", 1].agent, run["wary", 1].agent
    expected_all = map_distance(plain, wary, world, "own", tiles="all")
    assert table[0]["own_distance_all"] == format_cell(expected_all)
    expected_all = map_distance(plain, wary, world, "reasoner", tiles="all")
    assert table[0]["reasoner_distance_all"] == format_cell(expected_all)


def test_a_distance_without_shared_tiles_is_an_empty_cell(short: RunResult) -> None:
    far = short.model_copy(
        update={
            "agent": short.agent.model_copy(update={"tries": {}}),
            "config": short.config.model_copy(update={"seed": 2}),
        }
    )
    table = rows(experiment_tables([short, far])["distances.csv"])
    assert len(table) == 1 and table[0]["kind"] == "within"
    assert table[0]["own_distance"] == table[0]["reasoner_distance"] == ""
    assert table[0]["own_distance_all"] != "" and table[0]["reasoner_distance_all"] != ""


def test_stability_csv_has_one_row_per_measure(four: list[RunResult]) -> None:
    table = rows(experiment_tables(four)["stability.csv"])
    assert [r["metric"] for r in table] == list(RunMetrics.model_fields)
    assert all((r["n_agents"], r["n_seeds"]) == ("2", "2") for r in table)


def test_one_seed_writes_no_stability_csv_but_a_distances_csv(four: list[RunResult]) -> None:
    tables = experiment_tables([r for r in four if r.config.seed == 1])
    assert set(tables) == {"results.csv", "trace.csv", "distances.csv"}
    assert len(rows(tables["distances.csv"])) == 1


# --- icc1 and cells -----------------------------------------------------------------------------


def test_icc1_is_one_without_variance_inside_groups() -> None:
    assert icc1([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]) == pytest.approx(1.0)


def test_icc1_is_nan_for_one_group_or_no_variance() -> None:
    assert math.isnan(icc1([[1.0, 2.0, 3.0]]))
    assert math.isnan(icc1([[1.0, 1.0], [1.0, 1.0]]))
    assert math.isnan(icc1([[1.0], [2.0]]))


def test_format_cell() -> None:
    assert format_cell(None) == ""
    assert format_cell(0.1) == "0.1"
    assert format_cell(float("nan")) == "nan"
    assert format_cell(True) == "True"
    assert format_cell(Belief.HURTS) == "hurts"
    assert format_cell(3) == "3"
