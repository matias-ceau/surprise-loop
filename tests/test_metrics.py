import asyncio
import math

import pytest

from surprise_loop.config import (
    AgentSpec,
    Backends,
    ConflictParams,
    MetricsParams,
    Params,
    RunConfig,
    initial_agent_state,
)
from surprise_loop.loop import RunResult, run
from surprise_loop.metrics import (
    Censored,
    PEPoint,
    PhaseMetrics,
    RunMetrics,
    breakpoint,
    extinction_persistence,
    flatten,
    icc1,
    pe_trace,
    phase_table,
    response_rate,
    summarize,
)
from surprise_loop.state import (
    Appraisal,
    Decision,
    Feedback,
    GoalName,
    Mood,
    PassRecord,
    Prediction,
    PredictionError,
    TickRecord,
    Traits,
)
from surprise_loop.world import (
    Action,
    Event,
    Observation,
    Outcome,
    PhaseKind,
    PhaseSpec,
    SelfAdminWorld,
    StepInfo,
    WorldConfig,
)

FR, PR, EXT = PhaseKind.FIXED_RATIO, PhaseKind.PROGRESSIVE_RATIO, PhaseKind.EXTINCTION
ACTIVE, INACTIVE = Action.PRESS_A, Action.PRESS_B

# --- helpers ------------------------------------------------------------------------------------


def make_pass(
    step: int,
    phase_index: int,
    phase_kind: PhaseKind,
    action: Action = Action.REST,
    *,
    completed: int | None = None,
    pe: float = 0.0,
    tick: int | None = None,
    pass_index: int = 0,
) -> PassRecord:
    """One pass with the lever A active: pressing A is an active press, B an inactive one.

    `completed` marks a reinforced press that completed a ratio of that size.
    """
    active = action == ACTIVE
    reinforced = completed is not None
    if reinforced:
        assert active, "only an active press can be reinforced"
        event, reward = Event.REWARD, 1.0
    else:
        event = {
            Action.PRESS_A: Event.NO_REWARD,
            Action.PRESS_B: Event.NO_REWARD,
            Action.EXPLORE: Event.EXPLORED,
            Action.REST: Event.RESTED,
        }[action]
        reward = 0.0
    prediction = Prediction(expected_event=Event.NO_REWARD, expected_reward=0.0, confidence=0.5)
    return PassRecord(
        tick=step if tick is None else tick,
        pass_index=pass_index,
        decision=Decision(action=action, prediction=prediction),
        outcome=Outcome(
            observation=Observation(step=step, last_action=action, event=event, reward=reward),
            info=StepInfo(
                step=step,
                phase_index=phase_index,
                phase_kind=phase_kind,
                ratio_required=None if phase_kind == EXT else (completed or 1),
                active_press=active,
                ratio_completed=completed,
            ),
        ),
        prediction_error=PredictionError(
            raw_signed=pe, signed=pe / 2, unsigned=abs(pe) / 2, event_mismatch=False
        ),
        feedback=Feedback(alignment=0.0),
        goal_revised=None,
        conflict=None,
    )


_CODES = {"a": ACTIVE, "i": INACTIVE, ".": Action.REST, "e": Action.EXPLORE}


def passes(*phases: tuple[PhaseKind, str]) -> list[PassRecord]:
    """Consecutive phases, one pass per tick, steps numbered from 1.

    Each phase is (kind, pattern); per step: `a` active press, `i` inactive press, `.` rest,
    `e` explore, a digit d a reinforced active press completing a ratio of d.
    """
    out: list[PassRecord] = []
    for phase_index, (kind, pattern) in enumerate(phases):
        for code in pattern:
            step = len(out) + 1
            if code.isdigit():
                out.append(make_pass(step, phase_index, kind, ACTIVE, completed=int(code)))
            else:
                out.append(make_pass(step, phase_index, kind, _CODES[code]))
    return out


# --- flatten ------------------------------------------------------------------------------------


def _record(tick: int, *ps: PassRecord) -> TickRecord:
    return TickRecord(
        tick=tick,
        observation=Observation(step=0, last_action=None, event=Event.START, reward=0.0),
        appraisal=Appraisal(valence=0.0, arousal=0.0),
        passes=list(ps),
    )


def test_flatten_orders_passes_by_tick_then_pass() -> None:
    p1 = make_pass(1, 0, FR, tick=1)
    p2 = make_pass(2, 0, FR, tick=2, pass_index=0)
    p3 = make_pass(3, 0, FR, tick=2, pass_index=1)
    p4 = make_pass(4, 0, FR, tick=3)
    records = [_record(1, p1), _record(2, p2, p3), _record(3, p4)]
    assert flatten(records) == [p1, p2, p3, p4]
    assert flatten([]) == []


# --- phase_table and response_rate --------------------------------------------------------------


def test_phase_table_keeps_two_fr_phases_separate() -> None:
    ps = passes((FR, "a1.."), (FR, "iaa2aa"), (PR, "e."))
    ps[0] = make_pass(1, 0, FR, ACTIVE, pe=-0.4)
    ps[1] = make_pass(2, 0, FR, ACTIVE, completed=1, pe=0.8)
    ps[4] = make_pass(5, 1, FR, INACTIVE, pe=0.6)
    rows = [row.model_dump() for row in phase_table(ps)]
    assert rows == [
        {
            "phase_index": 0,
            "phase_kind": FR,
            "steps": 4,
            "active_presses": 2,
            "response_rate": 0.5,
            "rewards": 1,
            "mean_abs_pe": pytest.approx(0.3),
        },
        {
            "phase_index": 1,
            "phase_kind": FR,
            "steps": 6,
            "active_presses": 5,
            "response_rate": pytest.approx(5 / 6),
            "rewards": 1,
            "mean_abs_pe": pytest.approx(0.1),
        },
        {
            "phase_index": 2,
            "phase_kind": PR,
            "steps": 2,
            "active_presses": 0,
            "response_rate": 0.0,
            "rewards": 0,
            "mean_abs_pe": 0.0,
        },
    ]


def test_phase_table_has_no_row_for_unreached_phases() -> None:
    ps = passes((FR, "aa"))
    assert [row.phase_index for row in phase_table(ps)] == [0]
    assert phase_table([]) == []


@pytest.mark.parametrize(
    ("phase_index", "expected"),
    [(None, 6 / 10), (0, 2 / 4), (1, 4 / 6), (2, 0.0)],
)
def test_response_rate_per_phase_and_overall(phase_index: int | None, expected: float) -> None:
    ps = passes((FR, "a1i."), (FR, "aai2ae"))
    assert response_rate(ps, phase_index) == pytest.approx(expected)


def test_response_rate_is_zero_without_steps() -> None:
    assert response_rate([]) == 0.0
    assert response_rate([], 0) == 0.0


# --- breakpoint and extinction persistence ------------------------------------------------------


@pytest.mark.parametrize(
    ("pr_pattern", "quit_steps", "expected"),
    [
        # last completion before a run of quit_steps non-press steps; later completions ignored
        ("1a2a3...a4", 3, Censored(value=3, censored=False)),
        # exactly quit_steps non-press steps: the criterion is met
        ("12...", 3, Censored(value=2, censored=False)),
        # quit_steps - 1 non-press steps: not met; the phase ends first
        ("12..a3..", 3, Censored(value=3, censored=True)),
        # a quit run cut short by the end of the phase
        ("12a..", 3, Censored(value=2, censored=True)),
        # active presses (unreinforced) reset the run; inactive presses do not
        ("1.i.a3i.i.5", 4, Censored(value=3, censored=False)),
        # no completion before quitting
        ("aa....1", 4, Censored(value=0, censored=False)),
        # the quit run starts at the first step
        ("...12", 3, Censored(value=0, censored=False)),
    ],
)
def test_breakpoint(pr_pattern: str, quit_steps: int, expected: Censored) -> None:
    ps = passes((FR, "a1"), (PR, pr_pattern), (EXT, "...."))
    assert breakpoint(ps, quit_steps) == expected


def test_breakpoint_quit_run_does_not_straddle_a_phase_boundary() -> None:
    # Two non-press steps end FR and two begin PR: four in a row, but only two inside PR.
    ps = passes((FR, "1a.."), (PR, "..12..a"), (EXT, ".."))
    assert breakpoint(ps, 3) == Censored(value=2, censored=True)
    # And the PR phase's trailing non-presses do not run on into extinction.
    ps = passes((FR, "1"), (PR, "12.."), (EXT, "...."))
    assert breakpoint(ps, 3) == Censored(value=2, censored=True)


def test_breakpoint_uses_the_first_pr_phase() -> None:
    ps = passes((PR, "12..."), (PR, "123456"))
    assert breakpoint(ps, 3) == Censored(value=2, censored=False)


@pytest.mark.parametrize(
    ("ext_pattern", "quit_steps", "expected"),
    [
        ("aaa.a...aa", 3, Censored(value=4, censored=False)),  # later presses ignored
        ("aa...", 3, Censored(value=2, censored=False)),  # exactly quit_steps
        ("aa..a..", 3, Censored(value=3, censored=True)),  # quit_steps - 1: not met
        ("a..", 3, Censored(value=1, censored=True)),  # cut short by the phase end
        ("aia.i...", 3, Censored(value=2, censored=False)),  # inactive presses do not count
        ("...aaa", 3, Censored(value=0, censored=False)),
    ],
)
def test_extinction_persistence(ext_pattern: str, quit_steps: int, expected: Censored) -> None:
    ps = passes((FR, "a1"), (EXT, ext_pattern), (FR, "...."))
    assert extinction_persistence(ps, quit_steps) == expected


def test_extinction_quit_run_does_not_straddle_a_phase_boundary() -> None:
    ps = passes((PR, "1a.."), (EXT, ".aa..a"), (FR, "...."))
    assert extinction_persistence(ps, 3) == Censored(value=3, censored=True)


def test_extinction_uses_the_first_extinction_phase() -> None:
    ps = passes((EXT, "aa..."), (EXT, "aaaaa"))
    assert extinction_persistence(ps, 3) == Censored(value=2, censored=False)


def test_extinction_cut_short_by_the_end_of_the_run() -> None:
    ps = passes((FR, "1"), (EXT, "aaa.."))
    assert extinction_persistence(ps, 3) == Censored(value=3, censored=True)


def test_agent_that_never_presses() -> None:
    ps = passes((FR, "....."), (PR, "e.e.e."), (EXT, "....."))
    assert response_rate(ps) == 0.0
    assert all(row.response_rate == 0.0 for row in phase_table(ps))
    assert breakpoint(ps, 3) == Censored(value=0, censored=False)
    assert extinction_persistence(ps, 3) == Censored(value=0, censored=False)


@pytest.mark.parametrize(
    "ps",
    [
        passes((FR, "a1aa1")),
        passes((FR, "a1"), (FR, "a1")),
        [],
    ],
)
def test_missing_pr_or_extinction_phase_is_censored_zero(ps: list[PassRecord]) -> None:
    assert breakpoint(ps, 3) == Censored(value=0, censored=True)
    assert extinction_persistence(ps, 3) == Censored(value=0, censored=True)


def test_missing_only_one_of_the_phases() -> None:
    ps = passes((FR, "1"), (EXT, "a...."))
    assert breakpoint(ps, 3) == Censored(value=0, censored=True)
    assert extinction_persistence(ps, 3) == Censored(value=1, censored=False)
    ps = passes((FR, "1"), (PR, "1...."))
    assert breakpoint(ps, 3) == Censored(value=1, censored=False)
    assert extinction_persistence(ps, 3) == Censored(value=0, censored=True)


@pytest.mark.parametrize("quit_steps", [0, -1])
def test_quit_steps_must_be_positive(quit_steps: int) -> None:
    ps = passes((PR, "1.."), (EXT, "a.."))
    with pytest.raises(ValueError, match="quit_steps"):
        breakpoint(ps, quit_steps)
    with pytest.raises(ValueError, match="quit_steps"):
        extinction_persistence(ps, quit_steps)


def test_metric_functions_do_not_mutate_their_input() -> None:
    ps = passes((FR, "a1."), (PR, "12..."), (EXT, "a..."))
    before = list(ps)
    phase_table(ps)
    response_rate(ps)
    breakpoint(ps, 2)
    extinction_persistence(ps, 2)
    pe_trace(ps)
    assert ps == before


# --- pe_trace -----------------------------------------------------------------------------------


def test_pe_trace_has_one_point_per_pass_in_step_order() -> None:
    ps = [
        make_pass(1, 0, FR, ACTIVE, pe=0.5, tick=1),
        make_pass(2, 0, FR, ACTIVE, completed=2, pe=-0.25, tick=2, pass_index=0),
        make_pass(3, 1, PR, INACTIVE, pe=1.0, tick=2, pass_index=1),
        make_pass(4, 2, EXT, Action.REST, pe=-1.5, tick=3),
    ]
    assert pe_trace(ps) == [
        PEPoint(
            step=1,
            tick=1,
            pass_index=0,
            phase_index=0,
            phase_kind=FR,
            action=ACTIVE,
            active_press=True,
            raw_signed=0.5,
            signed=0.25,
            unsigned=0.25,
        ),
        PEPoint(
            step=2,
            tick=2,
            pass_index=0,
            phase_index=0,
            phase_kind=FR,
            action=ACTIVE,
            active_press=True,
            raw_signed=-0.25,
            signed=-0.125,
            unsigned=0.125,
        ),
        PEPoint(
            step=3,
            tick=2,
            pass_index=1,
            phase_index=1,
            phase_kind=PR,
            action=INACTIVE,
            active_press=False,
            raw_signed=1.0,
            signed=0.5,
            unsigned=0.5,
        ),
        PEPoint(
            step=4,
            tick=3,
            pass_index=0,
            phase_index=2,
            phase_kind=EXT,
            action=Action.REST,
            active_press=False,
            raw_signed=-1.5,
            signed=-0.75,
            unsigned=0.75,
        ),
    ]
    assert pe_trace([]) == []


# --- models -------------------------------------------------------------------------------------


def test_metric_models_are_frozen_flat_and_strict() -> None:
    c = Censored(value=1, censored=False)
    with pytest.raises(ValueError):
        c.value = 2  # type: ignore[misc]
    with pytest.raises(ValueError):
        Censored(value=1, censored=False, extra=1)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        PEPoint(
            step=1,
            tick=1,
            pass_index=0,
            phase_index=0,
            phase_kind=FR,
            action=ACTIVE,
            active_press=True,
            raw_signed=math.nan,
            signed=0.0,
            unsigned=0.0,
        )
    for model in (PhaseMetrics, Censored, PEPoint, RunMetrics):
        for name, field in model.model_fields.items():
            assert field.annotation in (int, float, bool, PhaseKind, Action), (model, name)


# --- summarize ----------------------------------------------------------------------------------

TRAITS = Traits(reward_sensitivity=1.0, surprise_learning_rate=0.3, mood_inertia=0.5)
PRIORITIES = {GoalName.SEEK_REWARD: 1.0, GoalName.EXPLORE: 0.5, GoalName.CONSERVE: 0.5}
QUIT_STEPS = 4
PARAMS = Params(
    conflict=ConflictParams(enabled=False), metrics=MetricsParams(quit_steps=QUIT_STEPS)
)


def make_config(schedule: list[PhaseSpec], ticks: int | None, seed: int = 5) -> RunConfig:
    return RunConfig(
        run_id="metrics-test",
        seed=seed,
        ticks=ticks,
        world=WorldConfig(active_lever="random", reward_magnitude=1.0, schedule=schedule),
        agent=AgentSpec(
            name="a", traits=TRAITS, mood=Mood(valence=0.0, arousal=0.0), goal_priorities=PRIORITIES
        ),
        params=PARAMS,
        backends=Backends(),
        model=None,
    )


SCHEDULES = {
    # exactly 100 steps: two FR phases, PR, extinction; the run finishes its schedule
    "complete": [
        PhaseSpec(kind=FR, steps=15, ratio=1),
        PhaseSpec(kind=FR, steps=15, ratio=2),
        PhaseSpec(kind=PR, steps=40, pr_start=1, pr_increment=2),
        PhaseSpec(kind=EXT, steps=30),
    ],
    # 150 steps: 100 ticks stop inside extinction; a trailing FR phase follows the PR phase
    "truncated": [
        PhaseSpec(kind=FR, steps=20, ratio=1),
        PhaseSpec(kind=PR, steps=30, pr_start=1, pr_increment=1),
        PhaseSpec(kind=FR, steps=10, ratio=1),
        PhaseSpec(kind=EXT, steps=90),
    ],
}


def _quit_start(phase_passes: list[PassRecord], quit_steps: int) -> int | None:
    """Index in `phase_passes` where the first run of `quit_steps` non-press steps starts."""
    for start in range(len(phase_passes) - quit_steps + 1):
        window = phase_passes[start : start + quit_steps]
        if not any(p.outcome.info.active_press for p in window):
            return start
    return None


@pytest.mark.parametrize("name", list(SCHEDULES))
def test_summarize_a_100_tick_stub_run_matches_hand_computation(name: str) -> None:
    config = make_config(SCHEDULES[name], ticks=100)
    result = asyncio.run(run(config))
    m = summarize(result)

    ps = [p for record in result.records for p in record.passes]
    kinds = {i: spec.kind for i, spec in enumerate(config.world.schedule)}
    assert len(result.records) == 100

    first_pr = min(i for i, k in kinds.items() if k == PR)
    first_ext = min(i for i, k in kinds.items() if k == EXT)

    def rate(selected: list[PassRecord]) -> float:
        hits = 0
        for p in selected:
            if p.outcome.info.active_press:
                hits += 1
        return hits / len(selected) if selected else 0.0

    fr_before_pr = [
        p for p in ps if p.outcome.info.phase_kind == FR and p.outcome.info.phase_index < first_pr
    ]
    pr_steps = [p for p in ps if p.outcome.info.phase_kind == PR]
    ext_steps = [p for p in ps if p.outcome.info.phase_kind == EXT]

    inactive = 0
    rewards = 0
    abs_pe = 0.0
    for p in ps:
        if (
            p.decision.action in (Action.PRESS_A, Action.PRESS_B)
            and not p.outcome.info.active_press
        ):
            inactive += 1
        if p.outcome.observation.event == Event.REWARD:
            rewards += 1
        abs_pe += abs(p.prediction_error.raw_signed)

    pr_phase = [p for p in ps if p.outcome.info.phase_index == first_pr]
    start = _quit_start(pr_phase, QUIT_STEPS)
    bp = 0
    for p in pr_phase[: len(pr_phase) if start is None else start]:
        if p.outcome.info.ratio_completed is not None:
            bp = p.outcome.info.ratio_completed

    ext_phase = [p for p in ps if p.outcome.info.phase_index == first_ext]
    start_ext = _quit_start(ext_phase, QUIT_STEPS)
    persistence = 0
    for p in ext_phase[: len(ext_phase) if start_ext is None else start_ext]:
        if p.outcome.info.active_press:
            persistence += 1

    expected = RunMetrics(
        steps=len(ps),
        ticks=100,
        truncated=name == "truncated",
        rewards=rewards,
        response_rate_fr=rate(fr_before_pr),
        response_rate_pr=rate(pr_steps),
        response_rate_ext=rate(ext_steps),
        inactive_rate=inactive / len(ps),
        breakpoint=bp,
        breakpoint_censored=start is None,
        extinction_persistence=persistence,
        extinction_censored=start_ext is None,
        mean_abs_pe=abs_pe / len(ps),
        reentries=sum(1 for r in result.records if len(r.passes) == 2),
        goal_revisions=result.agent.goals.revisions,
    )
    for field, value in expected.model_dump().items():
        got = getattr(m, field)
        if isinstance(value, float):
            assert got == pytest.approx(value), field
        else:
            assert got == value, field
    # the run exercised the measures
    assert m.steps == 100 and m.rewards > 0 and m.response_rate_fr > 0


def test_summarize_counts_reentries_revisions_and_truncation() -> None:
    schedule = [
        PhaseSpec(kind=FR, steps=2, ratio=1),
        PhaseSpec(kind=PR, steps=3, pr_start=1, pr_increment=1),
        PhaseSpec(kind=EXT, steps=4),
    ]
    config = make_config(schedule, ticks=None)
    ps = passes((FR, "1i"), (PR, "12"))
    re = [
        p.model_copy(update={"tick": t, "pass_index": i})
        for p, (t, i) in zip(ps, [(1, 0), (1, 1), (2, 0), (3, 0)], strict=True)
    ]
    records = [_record(1, re[0], re[1]), _record(2, re[2]), _record(3, re[3])]
    agent = initial_agent_state(config)
    agent = agent.model_copy(
        update={"goals": agent.goals.model_copy(update={"revisions": 7}), "tick": 3}
    )
    world = (
        SelfAdminWorld(config.world, config.seed)
        .snapshot()
        .model_copy(update={"phase_index": 1, "step": 4})
    )
    m = summarize(RunResult(config=config, records=records, agent=agent, world=world))
    assert (m.steps, m.ticks, m.reentries, m.goal_revisions) == (4, 3, 1, 7)
    assert m.truncated is True
    assert m.rewards == 3
    assert m.response_rate_fr == 0.5
    assert m.response_rate_pr == 1.0
    assert m.response_rate_ext == 0.0
    assert m.inactive_rate == 0.25
    assert (m.breakpoint, m.breakpoint_censored) == (2, True)
    assert (m.extinction_persistence, m.extinction_censored) == (0, True)

    done = world.model_copy(update={"phase_index": 3})
    assert (
        summarize(RunResult(config=config, records=records, agent=agent, world=done)).truncated
        is False
    )


def test_summarize_with_no_records_has_zero_rates() -> None:
    config = make_config(SCHEDULES["complete"], ticks=None)
    agent = initial_agent_state(config)
    world = SelfAdminWorld(config.world, config.seed).snapshot()
    m = summarize(RunResult(config=config, records=[], agent=agent, world=world))
    assert m == RunMetrics(
        steps=0,
        ticks=0,
        truncated=True,
        rewards=0,
        response_rate_fr=0.0,
        response_rate_pr=0.0,
        response_rate_ext=0.0,
        inactive_rate=0.0,
        breakpoint=0,
        breakpoint_censored=True,
        extinction_persistence=0,
        extinction_censored=True,
        mean_abs_pe=0.0,
        reentries=0,
        goal_revisions=0,
    )


# --- icc1 ---------------------------------------------------------------------------------------


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
