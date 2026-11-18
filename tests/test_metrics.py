"""The measures of a run: hand-built pass lists, a real stub afternoon, a body collapsed all day,
`opposite_value_steps` and `icc1`."""

import asyncio
import math
from pathlib import Path

import pytest

from surprise_loop.config import MetricsParams, RunConfig, load_config
from surprise_loop.loop import RunResult, run
from surprise_loop.metrics import RunMetrics, flatten, icc1, opposite_value_steps, summarize
from surprise_loop.state import Comparison, FeltFact, Override, PassRecord, TickRecord
from surprise_loop.world import Action, BodyState, Channel, FactName, Levels

ROOT = Path(__file__).resolve().parents[1]
AFTERNOON = ROOT / "experiments" / "clearing_afternoon.yaml"
DAY = ROOT / "experiments" / "clearing_day.yaml"

ZERO_FELT = {c: 0.0 for c in Channel}


def runs_of(path: Path) -> list[RunConfig]:
    return load_config(path).runs()


@pytest.fixture(scope="module")
def ilo_afternoon() -> RunResult:
    return asyncio.run(run(runs_of(AFTERNOON)[0]))


@pytest.fixture(scope="module")
def tavi_afternoon() -> RunResult:
    return asyncio.run(run(runs_of(AFTERNOON)[1]))


# --- hand-built pass lists ------------------------------------------------------------------


def comparison(
    heat: float = 0.0,
    water: float = 0.0,
    energy: float = 0.0,
    *,
    world: bool | None = None,
    size: float = 0.0,
) -> Comparison:
    body = {Channel.HEAT: heat, Channel.WATER: water, Channel.ENERGY: energy}
    return Comparison(body=body, body_felt=dict(body), world=world, better=0.0, size=size)


def make_pass(
    base: PassRecord,
    *,
    action: Action,
    body: tuple[float, float, float],
    comparison: Comparison,
    arousal: float,
    collapsed: bool = False,
    stumbled: bool = False,
    override: Override | None = None,
) -> PassRecord:
    """`base` with the fields the metrics read replaced. Its scene feels nothing, so the
    reactive rule on it chooses LOOK (offered at every place)."""
    heat, water, energy = body
    info = base.outcome.info.model_copy(
        update={
            "body": BodyState(heat=heat, water=water, energy=energy),
            "collapsed": collapsed,
            "stumbled": stumbled,
        }
    )
    assert Action.LOOK in base.offers
    return base.model_copy(
        update={
            "scene": base.scene.model_copy(update={"felt": ZERO_FELT}),
            "decision": base.decision.model_copy(update={"action": action}),
            "outcome": base.outcome.model_copy(update={"info": info}),
            "comparison": comparison,
            "feeling_after": base.feeling_after.model_copy(update={"arousal": arousal}),
            "override": override,
        }
    )


def with_ticks(base: RunResult, ticks: list[list[PassRecord]], valence: float) -> RunResult:
    """`base` holding the given passes, one inner list per tick, and a final valence."""
    template = base.records[0]
    records = [
        TickRecord(
            tick=n,
            observation=template.observation,
            appraisal=template.appraisal,
            passes=[p.model_copy(update={"tick": n, "pass_index": i}) for i, p in enumerate(tick)],
        )
        for n, tick in enumerate(ticks, start=1)
    ]
    feeling = base.agent.feeling.model_copy(update={"valence": valence})
    return base.model_copy(
        update={"records": records, "agent": base.agent.model_copy(update={"feeling": feeling})}
    )


INJECTED = Override(channel=Channel.WATER, actions=[Action.GO_STREAM], detail="injected")


@pytest.fixture(scope="module")
def hand_built(ilo_afternoon: RunResult) -> RunResult:
    """Four passes in three ticks (the third tick has two passes), under Ilo's temperament:
    setpoint heat -0.1, water 0.8, energy 0.7; tightness heat 3, water 6, energy 6.
    Felt per pass (heat, water, energy):
      p0 ( 1.2, -0.9, -1.2)   p1 (0.0, 1.2, 1.2)   p2 (0.9, 0.6, -1.8)   p3 (-1.2, -1.2, 3.0)
    """
    b = ilo_afternoon.records[0].passes[0]
    p0 = make_pass(
        b,
        action=Action.LOOK,
        body=(0.3, 0.95, 0.9),
        comparison=comparison(size=0.5),
        arousal=0.2,
    )
    p1 = make_pass(
        b,
        action=Action.REST,
        body=(-0.1, 0.6, 0.5),
        comparison=comparison(heat=0.1, world=True, size=1.5),
        arousal=0.4,
        collapsed=True,
    )
    p2 = make_pass(
        b,
        action=Action.LOOK,
        body=(0.2, 0.7, 1.0),
        comparison=comparison(water=-0.05, world=False, size=0.0),
        arousal=0.6,
        stumbled=True,
        override=INJECTED,
    )
    p3 = make_pass(
        b,
        action=Action.GO_OAK,
        body=(-0.5, 1.0, 0.2),
        comparison=comparison(water=0.01, energy=0.02, world=False, size=1.0),
        arousal=0.8,
    )
    return with_ticks(ilo_afternoon, [[p0], [p1], [p2, p3]], valence=-0.3)


def test_flatten_is_tick_order_then_pass_order(hand_built: RunResult) -> None:
    passes = flatten(hand_built.records)
    assert [(p.tick, p.pass_index) for p in passes] == [(1, 0), (2, 0), (3, 0), (3, 1)]


def test_every_metric_of_a_hand_built_run(hand_built: RunResult) -> None:
    assert summarize(hand_built) == RunMetrics(
        steps=4,
        ticks=3,
        truncated=False,
        uncomfortable_heat=2,  # p0 1.2 and p3 -1.2; p2's 0.9 is below 1
        uncomfortable_water=1,  # p1 1.2; p0's and p3's surpluses do not count
        uncomfortable_energy=2,  # p1 1.2 and p3 3.0; p0's -1.2 and p2's -1.8 are surpluses
        collapsed_steps=1,
        stumbles=1,
        n_go_meadow=0,
        n_go_oak=1,
        n_go_stream=0,
        n_drink=0,
        n_rest=1,
        n_look=2,
        mean_size=0.75,  # (0.5 + 1.5 + 0 + 1) / 4
        band_hits_heat=0.75,  # p1 misses
        band_hits_water=0.5,  # p2 and p3 miss
        band_hits_energy=0.75,  # p3 misses
        world_expectations=3,
        world_held=2,
        overrides=1,
        rule_agreement=0.5,  # the rule says LOOK on a scene that feels nothing: p0 and p2
        final_valence=-0.3,
        mean_arousal=0.5,  # (0.2 + 0.4 + 0.6 + 0.8) / 4
    )


def test_rule_agreement_leaves_out_passes_with_a_single_offer(hand_built: RunResult) -> None:
    # one more tick whose only offer is REST: chooser and rule both take it, so it says nothing
    b = flatten(hand_built.records)[0]
    one = make_pass(b, action=Action.REST, body=(0.0, 0.8, 0.7), comparison=comparison(),
                    arousal=0.0).model_copy(update={"offers": [Action.REST]})  # fmt: skip
    ticks = [record.passes for record in hand_built.records]
    metrics = summarize(with_ticks(hand_built, [*ticks, [one]], valence=-0.3))
    assert metrics.steps == 5
    assert metrics.rule_agreement == 0.5  # still p0 and p2 of the four passes with a choice


def test_rule_agreement_is_zero_when_no_pass_had_a_choice(hand_built: RunResult) -> None:
    b = flatten(hand_built.records)[0]
    one = make_pass(b, action=Action.LOOK, body=(0.0, 0.8, 0.7), comparison=comparison(),
                    arousal=0.0).model_copy(update={"offers": [Action.LOOK]})  # fmt: skip
    assert summarize(with_ticks(hand_built, [[one]], valence=0.0)).rule_agreement == 0.0


def test_one_action_count_per_action() -> None:
    counts = [f for f in RunMetrics.model_fields if f.startswith("n_")]
    assert counts == [f"n_{a.value}" for a in Action]


def test_comfort_counts_at_exactly_comfort_felt(hand_built: RunResult) -> None:
    # p0's heat felt is 3 * (0.3 + 0.1) = 1.2 (to float precision), p3's is -1.2
    felt_p0 = 3 * (0.3 + 0.1)
    params = hand_built.config.params.model_copy(
        update={"metrics": MetricsParams(comfort_felt=felt_p0)}
    )
    config = hand_built.config.model_copy(update={"params": params})
    assert summarize(hand_built.model_copy(update={"config": config})).uncomfortable_heat >= 1
    above = params.model_copy(update={"metrics": MetricsParams(comfort_felt=felt_p0 + 0.01)})
    config = hand_built.config.model_copy(update={"params": above})
    assert summarize(hand_built.model_copy(update={"config": config})).uncomfortable_heat == 0


def test_comfort_follows_the_true_body_not_the_belief_or_the_scene(
    ilo_afternoon: RunResult,
) -> None:
    """Under Ilo's temperament the body (0.3, 0.6, 0.5) feels 1.2 on every channel and the
    setpoint (-0.1, 0.8, 0.7) feels nothing. Pass 0's true body strains while its belief and its
    scene feel nothing; pass 1 is the reverse."""
    b = ilo_afternoon.records[0].passes[0]
    strained, setpoint = (
        Levels(heat=0.3, water=0.6, energy=0.5),
        Levels(heat=-0.1, water=0.8, energy=0.7),
    )
    loud = {c: 1.2 for c in Channel}
    hidden = make_pass(
        b, action=Action.LOOK, body=(0.3, 0.6, 0.5), comparison=comparison(), arousal=0.2
    ).model_copy(update={"belief_after": setpoint})
    imagined = make_pass(
        b, action=Action.LOOK, body=(-0.1, 0.8, 0.7), comparison=comparison(), arousal=0.2
    )
    imagined = imagined.model_copy(
        update={"belief_after": strained, "scene": imagined.scene.model_copy(update={"felt": loud})}
    )
    metrics = summarize(with_ticks(ilo_afternoon, [[hidden], [imagined]], valence=0.0))
    assert (
        metrics.uncomfortable_heat,
        metrics.uncomfortable_water,
        metrics.uncomfortable_energy,
    ) == (1, 1, 1)
    only_hidden = summarize(with_ticks(ilo_afternoon, [[hidden]], valence=0.0))
    assert only_hidden.uncomfortable_heat == only_hidden.uncomfortable_water == 1
    assert only_hidden.uncomfortable_energy == 1
    only_imagined = summarize(with_ticks(ilo_afternoon, [[imagined]], valence=0.0))
    assert only_imagined.uncomfortable_heat == only_imagined.uncomfortable_water == 0
    assert only_imagined.uncomfortable_energy == 0


def test_a_run_with_no_passes_is_all_zeros(ilo_afternoon: RunResult) -> None:
    empty = ilo_afternoon.model_copy(update={"records": []})
    metrics = summarize(empty)
    assert metrics.steps == metrics.ticks == 0
    assert metrics.mean_size == metrics.mean_arousal == metrics.rule_agreement == 0.0
    assert metrics.band_hits_heat == metrics.band_hits_water == metrics.band_hits_energy == 0.0
    assert metrics.final_valence == ilo_afternoon.agent.feeling.valence
    for name, value in metrics.model_dump().items():
        if name not in {"truncated", "final_valence"}:
            assert value == 0, name


def test_truncated_when_the_tick_cap_ends_the_run_before_the_world() -> None:
    capped = runs_of(AFTERNOON)[0].model_copy(update={"ticks": 5})
    metrics = summarize(asyncio.run(run(capped)))
    assert metrics.truncated is True
    assert metrics.ticks == 5


# --- a real stub afternoon ------------------------------------------------------------------


@pytest.mark.parametrize("agent", [0, 1])
def test_a_stub_afternoon_has_no_world_expectation_and_follows_the_rule(
    agent: int, ilo_afternoon: RunResult, tavi_afternoon: RunResult
) -> None:
    result = (ilo_afternoon, tavi_afternoon)[agent]
    passes = flatten(result.records)
    metrics = summarize(result)
    assert metrics.steps == len(passes) == 42 and metrics.ticks == len(result.records)
    assert metrics.truncated is False
    # the stubs never commit to an answer about the world: nothing to compare, nothing divided
    assert metrics.world_expectations == 0 and metrics.world_held == 0
    # the reactive stub is the rule
    assert metrics.rule_agreement == 1.0
    assert sum(getattr(metrics, f"n_{a.value}") for a in Action) == metrics.steps
    assert metrics.mean_size == pytest.approx(sum(p.comparison.size for p in passes) / len(passes))
    assert metrics.final_valence == result.agent.feeling.valence
    for value in metrics.model_dump().values():
        assert math.isfinite(value)


def test_metrics_are_pure(ilo_afternoon: RunResult) -> None:
    before = ilo_afternoon.model_dump_json()
    assert summarize(ilo_afternoon) == summarize(ilo_afternoon)
    assert ilo_afternoon.model_dump_json() == before


# --- a body stuck at a limit ----------------------------------------------------------------


@pytest.mark.parametrize("agent", [0, 1])
def test_a_body_collapsed_all_day_gives_defined_metrics(agent: int) -> None:
    config = runs_of(DAY)[agent]
    # no water and nothing in the gut: the body is at the water floor on every step
    body = config.agent.body.model_copy(update={"water": 0.0, "energy": 0.0})
    config = config.model_copy(update={"agent": config.agent.model_copy(update={"body": body})})
    result = asyncio.run(run(config))
    metrics = summarize(result)
    assert metrics.truncated is False  # the day still ends
    assert metrics.steps == len(flatten(result.records)) > 0
    # a body that starts at a limit collapses after its first step, then on every step
    assert metrics.collapsed_steps == metrics.steps - 1
    assert metrics.uncomfortable_water == metrics.steps
    for value in metrics.model_dump().values():
        assert math.isfinite(value)


# --- opposite_value_steps -------------------------------------------------------------------


def fact(name: FactName, sign: str, noticed: bool = True) -> FeltFact:
    value = {"+": 0.5, "-": -0.5, "0": 0.0}[sign]
    return FeltFact(
        name=name,
        about=None,
        weight=0.2,
        value=value,
        sign=sign,  # type: ignore[arg-type]
        grade=0 if sign == "0" else 1,
        noticed=noticed,
        phrase=None,
    )


def with_scenes(base: RunResult, scenes: list[list[FeltFact]]) -> RunResult:
    p = base.records[0].passes[0]
    scenes_of = [p.scene.model_copy(update={"facts": facts}) for facts in scenes]
    return with_ticks(base, [[p.model_copy(update={"scene": s})] for s in scenes_of], valence=0.0)


def test_opposite_value_steps_counts_facts_noticed_by_both_with_opposite_signs(
    ilo_afternoon: RunResult,
) -> None:
    a = with_scenes(
        ilo_afternoon,
        [
            [fact(FactName.AIR, "+"), fact(FactName.SUN, "+")],  # AIR opposite: counts
            [fact(FactName.AIR, "+")],  # b does not notice AIR
            [fact(FactName.SUN, "+")],  # same sign
            [fact(FactName.AIR, "0")],  # neutral is not a sign
            [fact(FactName.AIR, "-")],  # b has another fact
            [fact(FactName.BREEZE, "-"), fact(FactName.SHADE, "+")],  # two opposite: counts once
            [fact(FactName.AIR, "+")],  # beyond b's length
        ],
    )
    b = with_scenes(
        ilo_afternoon,
        [
            [fact(FactName.SUN, "+"), fact(FactName.AIR, "-")],
            [fact(FactName.AIR, "-", noticed=False)],
            [fact(FactName.SUN, "+")],
            [fact(FactName.AIR, "-")],
            [fact(FactName.SUN, "+")],
            [fact(FactName.BREEZE, "+"), fact(FactName.SHADE, "-")],
        ],
    )
    assert opposite_value_steps(a, b) == 2
    assert opposite_value_steps(b, a) == 2


def test_opposite_value_steps_of_a_run_with_itself_or_an_empty_run_is_zero(
    ilo_afternoon: RunResult,
) -> None:
    assert opposite_value_steps(ilo_afternoon, ilo_afternoon) == 0
    empty = ilo_afternoon.model_copy(update={"records": []})
    assert opposite_value_steps(ilo_afternoon, empty) == 0


def test_the_afternoon_pair_values_some_fact_oppositely(
    ilo_afternoon: RunResult, tavi_afternoon: RunResult
) -> None:
    # at the first step Ilo finds the warm air a burden and Tavi finds it pleasant
    assert opposite_value_steps(ilo_afternoon, tavi_afternoon) >= 1


# --- icc1 (as v0) ---------------------------------------------------------------------------


def test_icc1_is_one_for_distinct_means_without_within_group_variance() -> None:
    assert icc1([[1.0, 1.0], [2.0, 2.0], [5.0, 5.0]]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("groups", "expected"),
    [
        ([[0.0, 2.0], [0.0, 2.0]], -1.0),  # k = 2: -1/(k-1)
        ([[-1.0, 0.0, 1.0], [1.0, -1.0, 0.0], [0.0, 1.0, -1.0]], -0.5),  # k = 3
        ([[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]], -1 / 3),  # k = 4
    ],
)
def test_icc1_for_identical_group_means_is_minus_one_over_k_minus_one(
    groups: list[list[float]], expected: float
) -> None:
    assert icc1(groups) == pytest.approx(expected)


def test_icc1_is_zero_when_between_equals_within_mean_square() -> None:
    # means 0, 1, 2 (MSB = 2) and within-group deviations of +-1 (MSW = 2)
    assert icc1([[-1.0, 1.0], [0.0, 2.0], [1.0, 3.0]]) == pytest.approx(0.0, abs=1e-12)


def test_icc1_unbalanced_groups_use_k0() -> None:
    # N = 5, means 1.5 and 5, MSB = 14.7, MSW = 14.5 / 3, k0 = (5 - 13/5) / 1 = 2.4
    msb, msw, k0 = 14.7, 14.5 / 3, 2.4
    expected = (msb - msw) / (msb + (k0 - 1) * msw)
    assert icc1([[1.0, 2.0], [3.0, 4.0, 8.0]]) == pytest.approx(expected)


@pytest.mark.parametrize(
    "groups",
    [
        [],
        [[1.0, 2.0, 3.0]],  # a single group
        [[1.0], [2.0], [3.0]],  # single-member groups
        [[1.0, 2.0], [3.0]],  # one single-member group
        [[0.1, 0.1], [0.1, 0.1, 0.1]],  # zero total variance
    ],
)
def test_icc1_is_nan_when_undefined(groups: list[list[float]]) -> None:
    assert math.isnan(icc1(groups))


def test_icc1_does_not_mutate_its_input() -> None:
    groups = [[3.0, 1.0], [2.0, 5.0]]
    icc1(groups)
    assert groups == [[3.0, 1.0], [2.0, 5.0]]
