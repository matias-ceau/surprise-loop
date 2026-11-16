"""Reconcile: the four channels and the habit-vs-goal conflict rule, on hand-built workspaces."""

import asyncio
import math

import pytest

from surprise_loop.config import (
    AgentSpec,
    Backends,
    ConflictParams,
    DispositionParams,
    GoalParams,
    MemoryParams,
    MoodParams,
    Params,
    RunConfig,
    Runtime,
    initial_agent_state,
)
from surprise_loop.phases import reconcile
from surprise_loop.state import (
    AgentState,
    Conflict,
    Decision,
    Dispositions,
    Feedback,
    GoalName,
    Goals,
    Mood,
    Prediction,
    PredictionError,
    Traits,
    Workspace,
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

WORLD = WorldConfig(
    active_lever="press_a",
    reward_magnitude=1.0,
    schedule=[PhaseSpec(kind=PhaseKind.FIXED_RATIO, steps=50, ratio=1)],
)
RS, ALPHA, M = 1.5, 0.5, 0.4
TRAITS = Traits(reward_sensitivity=RS, surprise_learning_rate=ALPHA, mood_inertia=M)
PRIORITIES = {GoalName.SEEK_REWARD: 1.0, GoalName.EXPLORE: 0.5, GoalName.CONSERVE: 0.5}
SEEK = GoalName.SEEK_REWARD


def goal_params(**overrides: object) -> GoalParams:
    fields: dict[str, object] = {
        "ema_rate": 1.0,  # beta = ema_rate * alpha = 0.5
        "error_threshold": 0.1,
        "patience": 3,
        "revision_step": 0.25,
        "min_priority": 0.1,
        "max_priority": 2.0,
    }
    return GoalParams(**(fields | overrides))


def disp_params(**overrides: object) -> DispositionParams:
    fields: dict[str, object] = {"rate": 0.2, "bound": 1.0, "alignment_weight": 0.5}
    return DispositionParams(**(fields | overrides))


def params(
    *,
    goals: GoalParams | None = None,
    dispositions: DispositionParams | None = None,
    conflict: ConflictParams | None = None,
    arousal_gain: float = 0.8,
    base_salience: float = 0.1,
) -> Params:
    """Every group Reconcile reads, set explicitly."""
    return Params(
        mood=MoodParams(arousal_gain=arousal_gain),
        memory=MemoryParams(retrieval_k=10, base_salience=base_salience),
        goals=goals or goal_params(),
        dispositions=dispositions or disp_params(),
        conflict=conflict or ConflictParams(enabled=False, habit_threshold=0.0),
    )


def make_rt(p: Params, traits: Traits = TRAITS) -> Runtime:
    config = RunConfig(
        run_id="reconcile",
        seed=1,
        ticks=50,
        world=WORLD,
        agent=AgentSpec(
            name="a", traits=traits, mood=Mood(valence=0, arousal=0), goal_priorities=PRIORITIES
        ),
        params=p,
        backends=Backends(),
        model=None,
    )
    return Runtime(config=config, world=SelfAdminWorld(WORLD, 1))


def make_agent(
    rt: Runtime,
    *,
    mood: Mood | None = None,
    priorities: dict[GoalName, float] | None = None,
    biases: dict[Action, float] | None = None,
    tick: int = 3,
) -> AgentState:
    agent = initial_agent_state(rt.config)
    update: dict[str, object] = {"tick": tick}
    if mood is not None:
        update["mood"] = mood
    if priorities is not None:
        update["goals"] = agent.goals.model_copy(update={"priorities": priorities})
    if biases is not None:
        update["dispositions"] = Dispositions(biases=biases)
    return agent.model_copy(update=update)


def pe(signed: float, unsigned: float | None = None) -> PredictionError:
    return PredictionError(
        raw_signed=signed,
        signed=signed,
        unsigned=abs(signed) if unsigned is None else unsigned,
        event_mismatch=False,
    )


def make_ws(
    agent: AgentState,
    *,
    error: PredictionError,
    action: Action = Action.PRESS_A,
    reward: float = 0.0,
    alignment: float = 0.0,
    pass_index: int = 0,
    step: int = 4,
) -> Workspace:
    event = Event.REWARD if reward > 0 else Event.NO_REWARD
    observation = Observation(step=step, last_action=action, event=event, reward=reward)
    info = StepInfo(
        step=step,
        phase_index=0,
        phase_kind=PhaseKind.FIXED_RATIO,
        ratio_required=1,
        active_press=action == Action.PRESS_A,
        ratio_completed=None,
    )
    prediction = Prediction(expected_event=Event.NO_REWARD, expected_reward=0.0, confidence=1.0)
    return Workspace(
        agent=agent,
        observation=Observation(step=step - 1, last_action=None, event=Event.START, reward=0.0),
        pass_index=pass_index,
        decision=Decision(action=action, prediction=prediction),
        outcome=Outcome(observation=observation, info=info),
        prediction_error=error,
        feedback=Feedback(alignment=alignment, critique="c"),
    )


def run(ws: Workspace, rt: Runtime) -> Workspace:
    return asyncio.run(reconcile(ws, rt))


def run_passes(
    agent: AgentState, rt: Runtime, signeds: list[float], **kw: object
) -> list[Workspace]:
    """Reconcile one pass per signed error in turn, feeding each result's agent to the next."""
    results = []
    for s in signeds:
        result = run(make_ws(agent, error=pe(s), **kw), rt)  # type: ignore[arg-type]
        results.append(result)
        agent = result.agent
    return results


# --- purity -------------------------------------------------------------------------------------


def test_reconcile_leaves_its_input_untouched_and_never_changes_tick() -> None:
    rt = make_rt(params())
    ws = make_ws(make_agent(rt), error=pe(0.7), reward=1.0, alignment=0.3)
    before = ws.model_copy(deep=True)

    result = run(ws, rt)

    assert ws == before
    assert result is not ws
    assert result.agent.tick == ws.agent.tick


# --- mood ---------------------------------------------------------------------------------------


def test_one_pass_moves_mood_by_the_formula() -> None:
    rt = make_rt(params(arousal_gain=0.8))
    agent = make_agent(rt, mood=Mood(valence=0.2, arousal=0.3))

    mood = run(make_ws(agent, error=pe(-0.6, 0.9)), rt).agent.mood

    assert mood.valence == pytest.approx(M * 0.2 + (1 - M) * math.tanh(RS * -0.6))
    assert mood.arousal == pytest.approx(M * 0.3 + (1 - M) * min(1, 0.8 * 0.9))


def test_arousal_drive_saturates_at_one() -> None:
    rt = make_rt(params(arousal_gain=5.0))
    agent = make_agent(rt, mood=Mood(valence=0.0, arousal=0.1))

    mood = run(make_ws(agent, error=pe(0.0, 0.5)), rt).agent.mood

    assert mood.arousal == pytest.approx(M * 0.1 + (1 - M) * 1)


def test_higher_mood_inertia_moves_mood_less() -> None:
    def delta(m: float) -> tuple[float, float]:
        traits = Traits(reward_sensitivity=RS, surprise_learning_rate=ALPHA, mood_inertia=m)
        rt = make_rt(params(), traits)
        agent = make_agent(rt, mood=Mood(valence=0.0, arousal=0.0))
        mood = run(make_ws(agent, error=pe(0.5, 0.5)), rt).agent.mood
        return abs(mood.valence), abs(mood.arousal)

    low, high = delta(0.2), delta(0.8)
    assert high[0] < low[0]
    assert high[1] < low[1]


@pytest.mark.parametrize("signed", [1e12, -1e12])
def test_mood_stays_in_range_for_extreme_error(signed: float) -> None:
    traits = Traits(reward_sensitivity=1e6, surprise_learning_rate=1.0, mood_inertia=0.0)
    rt = make_rt(params(arousal_gain=1e6), traits)
    agent = make_agent(rt, mood=Mood(valence=-signed / abs(signed), arousal=1.0))

    mood = run(make_ws(agent, error=pe(signed, abs(signed))), rt).agent.mood

    assert -1 <= mood.valence <= 1
    assert 0 <= mood.arousal <= 1
    assert mood.valence == (1.0 if signed > 0 else -1.0)
    assert mood.arousal == 1.0


# --- memory -------------------------------------------------------------------------------------


def test_one_episode_per_pass_with_salience_from_unsigned_error() -> None:
    rt = make_rt(params(base_salience=0.15))
    agent = make_agent(rt, tick=6)

    results = run_passes(agent, rt, [0.2, -0.4])

    episodes = results[-1].agent.memory.episodes
    assert len(results[0].agent.memory.episodes) == 1
    assert len(episodes) == 2
    assert [e.salience for e in episodes] == pytest.approx([0.15 + 0.2, 0.15 + 0.4])
    assert all(e.tick == 7 for e in episodes)
    assert episodes[-1].critique == "c"
    assert episodes[-1].prediction_error == pe(-0.4)


def test_salience_is_clamped_at_one() -> None:
    rt = make_rt(params(base_salience=0.5))
    episode = run(make_ws(make_agent(rt), error=pe(3.0)), rt).agent.memory.episodes[-1]
    assert episode.salience == 1.0


# --- goals --------------------------------------------------------------------------------------


@pytest.mark.parametrize("sign", [1, -1])
def test_priority_is_revised_on_the_patience_th_same_sign_pass(sign: int) -> None:
    rt = make_rt(params(goals=goal_params(patience=3, revision_step=0.25)))
    agent = make_agent(rt)

    results = run_passes(agent, rt, [sign * 1.0] * 4)

    priorities = [r.agent.goals.priorities[SEEK] for r in results]
    streaks = [r.agent.goals.error_streak[SEEK] for r in results]
    # beta = 0.5: ema = 0.5, 0.75, then revision resets to 0, then 0.5 again
    emas = [r.agent.goals.error_ema[SEEK] for r in results]
    assert priorities[:2] == [1.0, 1.0]
    assert priorities[2] == pytest.approx(1.0 + sign * 0.25)
    assert priorities[3] == priorities[2]
    assert streaks == [sign, 2 * sign, 0, sign]
    assert emas == pytest.approx([sign * 0.5, sign * 0.75, 0.0, sign * 0.5])
    assert [r.agent.goals.revisions for r in results] == [0, 0, 1, 1]
    assert [r.passes[-1].goal_revised for r in results] == [None, None, SEEK, None]


def test_a_sub_threshold_pass_restarts_the_streak() -> None:
    rt = make_rt(params(goals=goal_params(patience=3, error_threshold=0.1)))
    # ema: 0.5, 0.75, then 0.5·0.75 + 0.5·(−0.6) = 0.075 (below 0.1), then 0.5375, 0.76875
    results = run_passes(make_agent(rt), rt, [1.0, 1.0, -0.6, 1.0, 1.0])

    streaks = [r.agent.goals.error_streak[SEEK] for r in results]
    assert streaks == [1, 2, 0, 1, 2]
    assert all(r.agent.goals.priorities[SEEK] == 1.0 for r in results)
    assert results[-1].agent.goals.revisions == 0


def test_an_ema_exactly_at_the_threshold_does_not_count() -> None:
    rt = make_rt(params(goals=goal_params(patience=1, error_threshold=0.1)))
    # ema = 0.5 · 0.2 = 0.1, not above the threshold (a strict comparison); 0.5 · 0.21 is
    at = run_passes(make_agent(rt), rt, [0.2])[-1].agent.goals
    above = run_passes(make_agent(rt), rt, [0.21])[-1].agent.goals
    assert at.error_ema[SEEK] == 0.1
    assert at.error_streak[SEEK] == 0 and at.revisions == 0
    assert above.revisions == 1


def test_a_sign_flip_restarts_the_streak() -> None:
    rt = make_rt(params(goals=goal_params(patience=3, error_threshold=0.1)))
    # ema: 0.5, 0.75, then 0.375 − 1.0 = −0.625, then −0.8125
    results = run_passes(make_agent(rt), rt, [1.0, 1.0, -2.0, -1.0])

    streaks = [r.agent.goals.error_streak[SEEK] for r in results]
    assert streaks == [1, 2, -1, -2]
    assert all(r.agent.goals.priorities[SEEK] == 1.0 for r in results)


def test_passes_serving_another_goal_leave_this_goal_untouched() -> None:
    rt = make_rt(params(goals=goal_params(patience=1)))
    first = run(make_ws(make_agent(rt), error=pe(1.0)), rt).agent  # seek_reward revised once
    before = first.goals

    after = run(make_ws(first, error=pe(-1.0), action=Action.EXPLORE), rt).agent.goals

    assert after.priorities[SEEK] == before.priorities[SEEK]
    assert after.error_ema[SEEK] == before.error_ema[SEEK]
    assert after.error_streak[SEEK] == before.error_streak[SEEK]
    assert after.priorities[GoalName.EXPLORE] == pytest.approx(0.5 - 0.25)
    assert after.priorities[GoalName.CONSERVE] == before.priorities[GoalName.CONSERVE]


@pytest.mark.parametrize(("start", "signed", "bound"), [(1.9, 1.0, 2.0), (0.2, -1.0, 0.1)])
def test_revision_is_clamped_at_both_bounds(start: float, signed: float, bound: float) -> None:
    rt = make_rt(
        params(goals=goal_params(patience=1, revision_step=0.25, min_priority=0.1, max_priority=2))
    )
    agent = make_agent(rt, priorities={**PRIORITIES, SEEK: start})

    result = run(make_ws(agent, error=pe(signed)), rt)

    assert result.agent.goals.priorities[SEEK] == bound
    assert result.agent.goals.revisions == 1
    assert result.passes[-1].goal_revised == SEEK


# --- dispositions -------------------------------------------------------------------------------


def test_only_the_taken_actions_bias_moves_by_the_formula() -> None:
    rt = make_rt(params(dispositions=disp_params(rate=0.2, bound=1.0, alignment_weight=0.5)))
    biases = {Action.PRESS_A: 0.1, Action.PRESS_B: -0.2, Action.EXPLORE: 0.3, Action.REST: 0.0}
    agent = make_agent(rt, biases=biases)

    result = run(make_ws(agent, error=pe(0.3), reward=1.0, alignment=-0.4), rt)

    v = RS * 1.0 + 0.5 * -0.4
    new = result.agent.dispositions.biases
    assert new[Action.PRESS_A] == pytest.approx(0.1 + 0.2 * ALPHA * (v - 0.1))
    assert {a: b for a, b in new.items() if a != Action.PRESS_A} == {
        a: b for a, b in biases.items() if a != Action.PRESS_A
    }


@pytest.mark.parametrize(("reward", "expected"), [(1e6, 0.7), (-1e6, -0.7)])
def test_bias_is_clamped_at_the_bound(reward: float, expected: float) -> None:
    alpha_one = Traits(reward_sensitivity=RS, surprise_learning_rate=1.0, mood_inertia=M)
    rt = make_rt(params(dispositions=disp_params(rate=1.0, bound=0.7)), alpha_one)

    result = run(make_ws(make_agent(rt), error=pe(0.0), reward=reward), rt)

    assert result.agent.dispositions.biases[Action.PRESS_A] == expected


# --- timescales ---------------------------------------------------------------------------------


def test_mood_moves_faster_than_goals_and_dispositions_with_default_params() -> None:
    # The one test that uses the default Params, by spec. "Relative change" is the change as a
    # fraction of the channel's range: valence 2, priority max − min, bias 2·bound.
    p = Params()
    rt = make_rt(p)
    agent = make_agent(rt)
    ws = make_ws(agent, error=pe(1.0), reward=1.0, alignment=1.0)

    new = run(ws, rt).agent

    mood_change = abs(new.mood.valence - agent.mood.valence) / 2
    goal_change = abs(new.goals.priorities[SEEK] - agent.goals.priorities[SEEK]) / (
        p.goals.max_priority - p.goals.min_priority
    )
    bias_change = abs(
        new.dispositions.biases[Action.PRESS_A] - agent.dispositions.biases[Action.PRESS_A]
    ) / (2 * p.dispositions.bound)
    assert mood_change > goal_change
    assert mood_change > bias_change
    assert mood_change > 0


# --- conflict -----------------------------------------------------------------------------------


def conflict_rt(enabled: bool = True, threshold: float = 0.3) -> Runtime:
    """patience 1 so one pass revises; rate 0 so the bias stays where the test puts it."""
    return make_rt(
        params(
            goals=goal_params(patience=1),
            dispositions=disp_params(rate=0.0),
            conflict=ConflictParams(enabled=enabled, habit_threshold=threshold),
        )
    )


def biased(rt: Runtime, bias: float) -> AgentState:
    return make_agent(rt, biases={a: (bias if a == Action.PRESS_A else 0.0) for a in Action})


def test_conflict_fires_on_downward_revision_with_bias_at_threshold() -> None:
    rt = conflict_rt(threshold=0.3)

    result = run(make_ws(biased(rt, 0.3), error=pe(-1.0)), rt)

    assert result.passes[-1].goal_revised == SEEK
    conflict = result.conflict
    assert conflict is not None
    assert conflict == result.passes[-1].conflict
    assert conflict.kind == "habit_vs_goal"
    assert conflict.action == Action.PRESS_A
    assert conflict.detail


def test_no_conflict_on_an_upward_revision() -> None:
    rt = conflict_rt()
    result = run(make_ws(biased(rt, 0.9), error=pe(1.0)), rt)
    assert result.passes[-1].goal_revised == SEEK
    assert result.conflict is None
    assert result.passes[-1].conflict is None


def test_no_conflict_below_the_habit_threshold() -> None:
    rt = conflict_rt(threshold=0.3)
    result = run(make_ws(biased(rt, 0.29), error=pe(-1.0)), rt)
    assert result.passes[-1].goal_revised == SEEK
    assert result.conflict is None


def test_no_conflict_without_a_revision() -> None:
    rt = make_rt(
        params(
            goals=goal_params(patience=2),
            dispositions=disp_params(rate=0.0),
            conflict=ConflictParams(enabled=True, habit_threshold=0.3),
        )
    )
    result = run(make_ws(biased(rt, 0.9), error=pe(-1.0)), rt)
    assert result.passes[-1].goal_revised is None
    assert result.conflict is None


def test_no_conflict_when_disabled() -> None:
    rt = conflict_rt(enabled=False)
    result = run(make_ws(biased(rt, 0.9), error=pe(-1.0)), rt)
    assert result.passes[-1].goal_revised == SEEK
    assert result.conflict is None


def test_never_a_conflict_on_pass_one() -> None:
    rt = conflict_rt()
    earlier = Conflict(kind="habit_vs_goal", action=Action.PRESS_A, detail="pass 0")
    ws = make_ws(biased(rt, 0.9), error=pe(-1.0), pass_index=1)
    ws = ws.model_copy(update={"conflict": earlier})

    result = run(ws, rt)

    assert result.passes[-1].goal_revised == SEEK
    assert result.passes[-1].pass_index == 1
    assert result.passes[-1].conflict is None
    # the pass-0 conflict Deliberate saw stays in the workspace
    assert result.conflict == earlier


def test_conflict_uses_the_bias_after_this_pass_update() -> None:
    # Decision recorded in the report: the rule compares the bias as updated by this pass.
    rt = make_rt(
        params(
            goals=goal_params(patience=1),
            dispositions=disp_params(rate=1.0, bound=1.0, alignment_weight=0.0),
            conflict=ConflictParams(enabled=True, habit_threshold=0.3),
        )
    )
    # bias 0.2 < 0.3 before; after: 0.2 + 1·α·(RS·1 − 0.2) = 0.2 + 0.5·1.3 = 0.85
    result = run(make_ws(biased(rt, 0.2), error=pe(-1.0), reward=1.0), rt)
    assert result.agent.dispositions.biases[Action.PRESS_A] == pytest.approx(0.85)
    assert result.conflict is not None


def test_goal_state_types_round_trip() -> None:
    rt = make_rt(params(goals=goal_params(patience=1)))
    goals = run(make_ws(make_agent(rt), error=pe(-1.0)), rt).agent.goals
    assert Goals.model_validate_json(goals.model_dump_json()) == goals
