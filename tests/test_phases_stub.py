import asyncio
import math
from collections.abc import Awaitable, Callable

import pytest

from surprise_loop.config import (
    AgentSpec,
    AppraiseParams,
    Backends,
    DeliberateParams,
    EvaluateParams,
    MemoryParams,
    Params,
    RunConfig,
    Runtime,
    derive_rng,
    initial_agent_state,
)
from surprise_loop.phases import act, appraise, deliberate, evaluate, monitor, perceive, reconcile
from surprise_loop.phases.appraise import appraise_stub
from surprise_loop.phases.evaluate import evaluate_stub
from surprise_loop.phases.heuristics import expected_reward, goal_utilities, softmax
from surprise_loop.phases.perceive import build_percept
from surprise_loop.state import (
    AgentState,
    Conflict,
    Decision,
    Dispositions,
    Episode,
    Feedback,
    GoalName,
    Memory,
    Mood,
    Percept,
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
    WorldState,
)

Phase = Callable[[Workspace, Runtime], Awaitable[Workspace]]

WORLD_CONFIG = WorldConfig(
    active_lever="press_a",
    reward_magnitude=1.0,
    schedule=[PhaseSpec(kind=PhaseKind.FIXED_RATIO, steps=20, ratio=1)],
)
TRAITS = Traits(reward_sensitivity=1.0, surprise_learning_rate=0.3, mood_inertia=0.5)


def make_rt(
    params: Params | None = None,
    *,
    seed: int = 7,
    traits: Traits = TRAITS,
    backends: Backends | None = None,
    world: object | None = None,
) -> Runtime:
    config = RunConfig(
        run_id="test",
        seed=seed,
        ticks=10,
        world=WORLD_CONFIG,
        agent=AgentSpec(name="a", traits=traits),
        params=params or Params(),
        backends=backends or Backends(),
        model=None,
    )
    return Runtime(config=config, world=world or SelfAdminWorld(WORLD_CONFIG, seed))


def run(phase: Phase, ws: Workspace, rt: Runtime) -> Workspace:
    return asyncio.run(phase(ws, rt))


def prediction(
    reward: float = 0.3, event: Event = Event.NO_REWARD, confidence: float = 0.0
) -> Prediction:
    return Prediction(expected_event=event, expected_reward=reward, confidence=confidence)


def episode(action: Action, reward: float, event: Event | None = None, tick: int = 1) -> Episode:
    if event is None:
        event = Event.REWARD if reward > 0 else Event.NO_REWARD
    return Episode(
        tick=tick,
        step=tick,
        action=action,
        prediction=prediction(),
        event=event,
        reward=reward,
        prediction_error=PredictionError(
            raw_signed=0.0, signed=0.0, unsigned=0.0, event_mismatch=False
        ),
        alignment=0.0,
        salience=0.1,
    )


def agent_with(
    rt: Runtime,
    *,
    episodes: list[Episode] | None = None,
    biases: dict[Action, float] | None = None,
    mood: Mood | None = None,
    tick: int = 4,
) -> AgentState:
    agent = initial_agent_state(rt.config)
    update: dict[str, object] = {"tick": tick}
    if episodes is not None:
        update["memory"] = Memory(episodes=episodes)
    if biases is not None:
        update["dispositions"] = Dispositions(biases=biases)
    if mood is not None:
        update["mood"] = mood
    return agent.model_copy(update=update)


START = Observation(step=3, last_action=Action.PRESS_A, event=Event.REWARD, reward=1.0)


def base_ws(rt: Runtime, agent: AgentState | None = None, obs: Observation = START) -> Workspace:
    return Workspace(agent=agent or agent_with(rt), observation=obs)


def full_ws(rt: Runtime, agent: AgentState | None = None) -> Workspace:
    """A workspace with every slot filled up to (but excluding) Reconcile."""
    ws = base_ws(rt, agent)
    for phase in (perceive, appraise, deliberate, act, monitor, evaluate):
        ws = run(phase, ws, rt)
    return ws


# --- purity: every phase returns a new Workspace and leaves its input untouched ---------------

INPUT_SLOTS = {
    "perceive": (perceive, 0),
    "appraise": (appraise, 1),
    "deliberate": (deliberate, 2),
    "act": (act, 3),
    "monitor": (monitor, 4),
    "evaluate": (evaluate, 5),
    "reconcile": (reconcile, 6),
}


@pytest.mark.parametrize("name", list(INPUT_SLOTS))
def test_phase_returns_new_workspace_and_leaves_input_equal(name: str) -> None:
    phase, position = INPUT_SLOTS[name]
    rt = make_rt()
    episodes = [episode(Action.PRESS_A, 1.0, tick=t) for t in (1, 2, 3)]
    ws = base_ws(rt, agent_with(rt, episodes=episodes))
    for earlier in (perceive, appraise, deliberate, act, monitor, evaluate)[:position]:
        ws = run(earlier, ws, rt)
    before = ws.model_copy(deep=True)

    result = run(phase, ws, rt)

    assert isinstance(result, Workspace)
    assert result is not ws
    assert result != ws
    assert ws == before


# --- Perceive -----------------------------------------------------------------------------------


def test_percept_has_no_step_and_no_step_info_field() -> None:
    assert "step" not in Percept.model_fields
    for field in Percept.model_fields.values():
        assert StepInfo not in (field.annotation, *getattr(field.annotation, "__args__", ()))


def test_build_percept_copies_observation_and_recalls_last_k() -> None:
    rt = make_rt()
    episodes = [episode(Action.PRESS_A, float(t % 2), tick=t) for t in range(1, 6)]
    agent = agent_with(rt, episodes=episodes)
    params = Params(memory=MemoryParams(retrieval_k=3))
    obs = Observation(step=9, last_action=Action.EXPLORE, event=Event.EXPLORED, reward=0.0)

    percept = build_percept(obs, agent, params)

    assert percept == Percept(
        event=Event.EXPLORED, reward=0.0, last_action=Action.EXPLORE, recalled=episodes[2:]
    )


def test_perceive_sets_percept_from_observation() -> None:
    params = Params(memory=MemoryParams(retrieval_k=2))
    rt = make_rt(params)
    episodes = [episode(Action.PRESS_B, 0.0, tick=t) for t in (1, 2, 3)]
    ws = base_ws(rt, agent_with(rt, episodes=episodes))

    result = run(perceive, ws, rt)

    assert result.percept == build_percept(ws.observation, ws.agent, params)
    assert result.percept.recalled == episodes[1:]


# --- heuristics: E[x] -----------------------------------------------------------------------------


def test_expected_reward_with_nothing_recalled_is_the_prior() -> None:
    params = DeliberateParams(prior_expected_reward=0.37, prior_weight=2.0)
    estimate = expected_reward(Action.PRESS_A, [], 0.3, params)
    assert estimate.value == pytest.approx(0.37)
    assert estimate.n == 0


def test_expected_reward_matches_formula_by_hand() -> None:
    params = DeliberateParams(prior_expected_reward=0.5, prior_weight=2.0)
    # oldest first; i = 0 is the most recent press_a (reward 1), i = 1 the older one (reward 0)
    recalled = [
        episode(Action.PRESS_A, 0.0, tick=1),
        episode(Action.PRESS_B, 1.0, tick=2),
        episode(Action.PRESS_A, 1.0, tick=3),
    ]
    alpha = 0.5
    p = 2.0 * 0.5**2
    expected = (p * 0.5 + 1.0 * 1.0 + 0.5 * 0.0) / (p + 1.0 + 0.5)
    estimate = expected_reward(Action.PRESS_A, recalled, alpha, params)
    assert estimate.value == pytest.approx(expected)
    assert estimate.n == 2


def _estimates_through_memory(reward: float, alpha: float, k: int) -> list[float]:
    params = Params(
        memory=MemoryParams(retrieval_k=k),
        deliberate=DeliberateParams(prior_expected_reward=0.3, prior_weight=1.0),
    )
    traits = Traits(reward_sensitivity=1.0, surprise_learning_rate=alpha, mood_inertia=0.5)
    rt = make_rt(params, traits=traits)
    values = []
    for n in range(k + 1):
        episodes = [episode(Action.PRESS_A, reward, tick=t) for t in range(1, n + 1)]
        agent = agent_with(rt, episodes=episodes)
        percept = build_percept(START, agent, params)
        values.append(expected_reward(Action.PRESS_A, percept.recalled, alpha, params.deliberate))
    return [v.value for v in values]


@pytest.mark.parametrize("alpha", [0.3, 0.9])
def test_repeated_reward_one_rises_monotonically_above_095_within_k(alpha: float) -> None:
    values = _estimates_through_memory(1.0, alpha, k=10)
    assert values[0] == pytest.approx(0.3)
    assert all(b > a for a, b in zip(values, values[1:], strict=False))
    assert all(v <= 1 for v in values)
    assert values[-1] > 0.95


@pytest.mark.parametrize("alpha", [0.3, 0.9])
def test_repeated_reward_zero_falls_monotonically_toward_zero(alpha: float) -> None:
    values = _estimates_through_memory(0.0, alpha, k=10)
    assert values[0] == pytest.approx(0.3)
    assert all(b < a for a, b in zip(values, values[1:], strict=False))
    assert all(v >= 0 for v in values)
    assert values[-1] < 0.05


# --- heuristics: utilities and softmax --------------------------------------------------------


def test_goal_utilities_match_formula_without_bias_or_mood_terms() -> None:
    params = DeliberateParams(
        prior_expected_reward=0.4,
        action_costs={
            Action.PRESS_A: 0.1,
            Action.PRESS_B: 0.2,
            Action.EXPLORE: 0.03,
            Action.REST: 0.01,
        },
        explore_value=0.6,
        rest_value=0.7,
        approach_weight=5.0,
        vigor_weight=5.0,
    )
    rt = make_rt(Params(deliberate=params))
    agent = agent_with(rt, biases={a: 0.9 for a in Action}, mood=Mood(valence=1.0, arousal=1.0))
    agent = agent.model_copy(
        update={
            "traits": Traits(reward_sensitivity=2.0, surprise_learning_rate=0.3, mood_inertia=0.5),
            "goals": agent.goals.model_copy(
                update={
                    "priorities": {
                        GoalName.SEEK_REWARD: 1.5,
                        GoalName.EXPLORE: 0.25,
                        GoalName.CONSERVE: 0.5,
                    }
                }
            ),
        }
    )

    u = goal_utilities(agent, [], params)

    assert list(u) == list(Action)
    assert u[Action.PRESS_A] == pytest.approx(1.5 * 2.0 * 0.4 - 0.1)
    assert u[Action.PRESS_B] == pytest.approx(1.5 * 2.0 * 0.4 - 0.2)
    assert u[Action.EXPLORE] == pytest.approx(0.25 * 0.6 - 0.03)
    assert u[Action.REST] == pytest.approx(0.5 * 0.7 - 0.01)


def test_softmax_is_stable_and_normalised() -> None:
    probs = softmax([1000.0, 1001.0, 999.0], temperature=1.0)
    assert sum(probs) == pytest.approx(1.0)
    expected = [math.exp(-1) / (1 + math.exp(-1) + math.exp(-2))]
    assert probs[0] == pytest.approx(expected[0])
    assert probs[1] > probs[0] > probs[2]
    cold = softmax([0.0, 1.0], temperature=1e-6)
    assert cold == [0.0, 1.0]
    hot = softmax([0.0, 1.0], temperature=1e6)
    assert hot[0] == pytest.approx(0.5, abs=1e-5)


# --- Deliberate -----------------------------------------------------------------------------------


def _deliberate_ready(rt: Runtime, agent: AgentState, pass_index: int = 0) -> Workspace:
    ws = base_ws(rt, agent).model_copy(update={"pass_index": pass_index})
    return run(appraise, run(perceive, ws, rt), rt)


def test_deliberate_with_nothing_recalled_predicts_prior_with_zero_confidence() -> None:
    params = Params(deliberate=DeliberateParams(prior_expected_reward=0.42))
    rt = make_rt(params)
    for seed in range(30):
        rt = make_rt(params, seed=seed)
        result = run(deliberate, _deliberate_ready(rt, agent_with(rt, episodes=[])), rt)
        d = result.decision
        assert d.prediction.confidence == 0
        if d.action in (Action.PRESS_A, Action.PRESS_B):
            assert d.prediction.expected_reward == pytest.approx(0.42)
            assert d.prediction.expected_event == Event.NO_REWARD
        else:
            assert d.prediction.expected_reward == 0
            expected = Event.EXPLORED if d.action == Action.EXPLORE else Event.RESTED
            assert d.prediction.expected_event == expected
        assert d.rationale


def _spread_params() -> Params:
    return Params(deliberate=DeliberateParams(temperature=5.0))


def test_deliberate_draw_matches_derived_rng_and_is_reproducible() -> None:
    params = _spread_params()
    seen = set()
    for seed in range(40):
        for pass_index in (0, 1):
            rt = make_rt(params, seed=seed)
            episodes = [
                episode(Action.PRESS_A, 1.0, tick=1),
                episode(Action.REST, 0.0, Event.RESTED),
            ]
            ws = _deliberate_ready(rt, agent_with(rt, episodes=episodes, tick=6), pass_index)

            first = run(deliberate, ws, rt).decision
            second = run(deliberate, ws, rt).decision
            assert first == second

            appraisal = ws.appraisal
            u = goal_utilities(ws.agent, ws.percept.recalled, params.deliberate)
            dp = params.deliberate
            for a in (Action.PRESS_A, Action.PRESS_B):
                u[a] += dp.approach_weight * appraisal.valence + dp.vigor_weight * appraisal.arousal
            u = {a: u[a] + ws.agent.dispositions.biases[a] for a in Action}
            probs = softmax(list(u.values()), dp.temperature)
            draw = derive_rng(seed, 7, pass_index).random()
            cumulative, expected = 0.0, list(Action)[-1]
            for a, p in zip(Action, probs, strict=True):
                cumulative += p
                if draw < cumulative:
                    expected = a
                    break
            assert first.action == expected
            seen.add(first.action)
    assert len(seen) == len(Action)


def test_deliberate_omits_bias_when_conflict_is_set() -> None:
    costs = {Action.PRESS_A: 0.05, Action.PRESS_B: 0.1, Action.EXPLORE: 0.02, Action.REST: 0.0}
    params = Params(
        deliberate=DeliberateParams(
            temperature=1e-3, prior_expected_reward=0.5, rest_value=0.0, action_costs=costs
        )
    )
    rt = make_rt(params)
    biases = {Action.PRESS_A: 0.0, Action.PRESS_B: 0.0, Action.EXPLORE: 0.0, Action.REST: 1.0}
    ws = _deliberate_ready(rt, agent_with(rt, biases=biases))

    without_conflict = run(deliberate, ws, rt).decision
    conflict = Conflict(kind="habit_vs_goal", action=Action.REST, detail="test")
    with_conflict = run(deliberate, ws.model_copy(update={"conflict": conflict}), rt).decision

    assert without_conflict.action == Action.REST
    assert with_conflict.action == Action.PRESS_A


def test_deliberate_predicts_most_frequent_recalled_event_and_confidence() -> None:
    params = Params(
        deliberate=DeliberateParams(temperature=1e-3, confidence_halfsat=2.0, rest_value=10.0)
    )
    rt = make_rt(params)
    episodes = [
        episode(Action.REST, 0.0, Event.RESTED, tick=1),
        episode(Action.REST, 1.0, Event.REWARD, tick=2),
        episode(Action.REST, 0.0, Event.RESTED, tick=3),
        episode(Action.PRESS_A, 1.0, Event.REWARD, tick=4),
    ]
    ws = _deliberate_ready(rt, agent_with(rt, episodes=episodes))

    d = run(deliberate, ws, rt).decision

    assert d.action == Action.REST
    assert d.prediction.expected_event == Event.RESTED
    assert d.prediction.expected_reward == 0
    assert d.prediction.confidence == pytest.approx(3 / (3 + 2.0))


def test_deliberate_uses_recalled_episodes_from_the_percept() -> None:
    costs = {Action.PRESS_A: 0.05, Action.PRESS_B: 0.06, Action.EXPLORE: 0.02, Action.REST: 0.0}
    params = Params(
        deliberate=DeliberateParams(temperature=1e-3, prior_expected_reward=0.0, action_costs=costs)
    )
    rt = make_rt(params)
    episodes = [episode(Action.PRESS_B, 1.0, tick=t) for t in (1, 2, 3)]
    ws = _deliberate_ready(rt, agent_with(rt, episodes=episodes))
    assert run(deliberate, ws, rt).decision.action == Action.PRESS_B
    forgetful = ws.model_copy(update={"percept": ws.percept.model_copy(update={"recalled": []})})
    assert run(deliberate, forgetful, rt).decision.action != Action.PRESS_B


@pytest.mark.parametrize("missing", ["percept", "appraisal"])
def test_deliberate_names_missing_input(missing: str) -> None:
    rt = make_rt()
    ws = _deliberate_ready(rt, agent_with(rt)).model_copy(update={missing: None})
    with pytest.raises(ValueError, match=missing):
        run(deliberate, ws, rt)


# --- Appraise -------------------------------------------------------------------------------------


def test_appraise_matches_formula() -> None:
    params = Params(appraise=AppraiseParams(reward_gain=0.25))
    traits = Traits(reward_sensitivity=2.0, surprise_learning_rate=0.3, mood_inertia=0.5)
    rt = make_rt(params, traits=traits)
    agent = agent_with(rt, mood=Mood(valence=-0.2, arousal=0.6))
    obs = Observation(step=1, last_action=Action.PRESS_A, event=Event.REWARD, reward=0.8)
    ws = run(perceive, base_ws(rt, agent, obs), rt)

    a = run(appraise, ws, rt).appraisal

    assert a.valence == pytest.approx(-0.2 + 0.25 * 2.0 * 1.0 * 0.8)
    assert a.arousal == pytest.approx(0.6)


@pytest.mark.parametrize("reward", [1e9, -1e9])
def test_appraise_stays_in_range_for_extreme_inputs(reward: float) -> None:
    params = Params(appraise=AppraiseParams(reward_gain=1e6))
    traits = Traits(reward_sensitivity=1e6, surprise_learning_rate=1.0, mood_inertia=0.0)
    rt = make_rt(params, traits=traits)
    valence = 1.0 if reward > 0 else -1.0
    agent = agent_with(rt, mood=Mood(valence=valence, arousal=1.0))
    obs = Observation(step=1, last_action=Action.PRESS_A, event=Event.REWARD, reward=reward)
    ws = run(perceive, base_ws(rt, agent, obs), rt)

    a = run(appraise_stub, ws, rt).appraisal

    assert a.valence == valence
    assert a.arousal == 1.0


def test_appraise_requires_percept() -> None:
    rt = make_rt()
    with pytest.raises(ValueError, match="percept"):
        run(appraise, base_ws(rt), rt)


# --- Act ------------------------------------------------------------------------------------------


class CountingWorld:
    def __init__(self, world: SelfAdminWorld) -> None:
        self.inner = world
        self.steps = 0

    @property
    def done(self) -> bool:
        return self.inner.done

    def observe(self) -> Observation:
        return self.inner.observe()

    def step(self, action: Action) -> Outcome:
        self.steps += 1
        return self.inner.step(action)

    def snapshot(self) -> WorldState:
        return self.inner.snapshot()


def test_act_steps_the_world_with_the_decided_action() -> None:
    rt = make_rt()
    ws = run(deliberate, _deliberate_ready(rt, agent_with(rt)), rt)
    ws = ws.model_copy(
        update={"decision": Decision(action=Action.PRESS_A, prediction=prediction())}
    )
    reference = SelfAdminWorld(WORLD_CONFIG, 7).step(Action.PRESS_A)

    result = run(act, ws, rt)

    assert result.outcome == reference
    assert rt.world.snapshot().step == 1


def test_act_is_the_only_phase_that_steps_the_world() -> None:
    counting = CountingWorld(SelfAdminWorld(WORLD_CONFIG, 7))
    rt = make_rt(world=counting)
    ws = base_ws(rt)
    for phase in (perceive, appraise, deliberate, act, monitor, evaluate, reconcile):
        before = counting.steps
        ws = run(phase, ws, rt)
        assert counting.steps - before == (1 if phase is act else 0), phase.__name__


def test_act_requires_decision() -> None:
    rt = make_rt()
    with pytest.raises(ValueError, match="decision"):
        run(act, base_ws(rt), rt)


# --- Monitor --------------------------------------------------------------------------------------


def _outcome(action: Action, event: Event, reward: float) -> Outcome:
    obs = Observation(step=5, last_action=action, event=event, reward=reward)
    info = StepInfo(
        step=5,
        phase_index=0,
        phase_kind=PhaseKind.FIXED_RATIO,
        ratio_required=1,
        active_press=action == Action.PRESS_A,
        ratio_completed=None,
    )
    return Outcome(observation=obs, info=info)


@pytest.mark.parametrize(
    ("action", "pred", "event", "reward", "raw", "mismatch"),
    [
        (Action.PRESS_A, prediction(0.3, Event.NO_REWARD), Event.REWARD, 1.0, 0.7, True),
        (Action.PRESS_B, prediction(0.6, Event.NO_REWARD), Event.NO_REWARD, 0.0, -0.6, False),
        (Action.EXPLORE, prediction(0.0, Event.EXPLORED), Event.EXPLORED, 0.0, 0.0, False),
    ],
)
def test_monitor_hand_cases(
    action: Action, pred: Prediction, event: Event, reward: float, raw: float, mismatch: bool
) -> None:
    rt = make_rt()
    ws = base_ws(rt).model_copy(
        update={
            "decision": Decision(action=action, prediction=pred),
            "outcome": _outcome(action, event, reward),
        }
    )

    pe = run(monitor, ws, rt).prediction_error

    assert pe.raw_signed == pytest.approx(raw)
    assert pe.signed == pytest.approx(raw)
    assert pe.unsigned == pytest.approx(abs(raw))
    assert pe.event_mismatch is mismatch


@pytest.mark.parametrize("missing", ["decision", "outcome"])
def test_monitor_names_missing_input(missing: str) -> None:
    rt = make_rt()
    ws = full_ws(rt).model_copy(update={missing: None})
    with pytest.raises(ValueError, match=missing):
        run(monitor, ws, rt)


# --- Evaluate -------------------------------------------------------------------------------------


def test_evaluate_leaves_agent_equal_and_only_sets_feedback() -> None:
    rt = make_rt()
    ws = full_ws(rt).model_copy(update={"feedback": None})

    result = run(evaluate, ws, rt)

    assert result.agent == ws.agent
    assert result.feedback is not None
    assert result.model_copy(update={"feedback": None}) == ws


def test_evaluate_alignment_matches_formula() -> None:
    params = Params(evaluate=EvaluateParams(scale=0.3))
    rt = make_rt(params)
    episodes = [episode(Action.PRESS_A, 1.0, tick=1), episode(Action.PRESS_B, 0.0, tick=2)]
    ws = full_ws(rt, agent_with(rt, episodes=episodes))
    for action in Action:
        decided = ws.model_copy(
            update={"decision": Decision(action=action, prediction=prediction())}
        )
        u = goal_utilities(ws.agent, ws.percept.recalled, params.deliberate)
        mean = sum(u.values()) / len(u)
        feedback = run(evaluate, decided, rt).feedback
        assert feedback.alignment == pytest.approx(math.tanh((u[action] - mean) / 0.3))


@pytest.mark.parametrize("action", [Action.PRESS_A, Action.REST])
def test_evaluate_stays_in_range_for_extreme_inputs(action: Action) -> None:
    params = Params(
        evaluate=EvaluateParams(scale=1e-12),
        deliberate=DeliberateParams(prior_expected_reward=1e6),
    )
    traits = Traits(reward_sensitivity=1e6, surprise_learning_rate=1.0, mood_inertia=0.0)
    rt = make_rt(params, traits=traits)
    ws = full_ws(rt).model_copy(
        update={"decision": Decision(action=action, prediction=prediction())}
    )

    alignment = run(evaluate_stub, ws, rt).feedback.alignment

    assert -1 <= alignment <= 1
    assert alignment == (1.0 if action == Action.PRESS_A else -1.0)


@pytest.mark.parametrize("missing", ["percept", "decision"])
def test_evaluate_names_missing_input(missing: str) -> None:
    rt = make_rt()
    ws = full_ws(rt).model_copy(update={missing: None})
    with pytest.raises(ValueError, match=missing):
        run(evaluate, ws, rt)


# --- Reconcile ------------------------------------------------------------------------------------


def test_reconcile_appends_episode_and_pass_record_only() -> None:
    params = Params(memory=MemoryParams(base_salience=0.15))
    rt = make_rt(params)
    old = [episode(Action.PRESS_A, 1.0, tick=1)]
    ws = full_ws(rt, agent_with(rt, episodes=old, tick=4))
    pe = PredictionError(raw_signed=-0.4, signed=-0.4, unsigned=0.4, event_mismatch=True)
    feedback = Feedback(alignment=-0.25, critique="meh")
    ws = ws.model_copy(update={"prediction_error": pe, "feedback": feedback, "pass_index": 1})
    earlier = ws.passes

    result = run(reconcile, ws, rt)

    assert result.agent.memory.episodes[:-1] == old
    new = result.agent.memory.episodes[-1]
    assert new.salience == pytest.approx(0.15 + 0.4)
    assert new == Episode(
        tick=5,
        step=ws.outcome.info.step,
        action=ws.decision.action,
        prediction=ws.decision.prediction,
        event=ws.outcome.observation.event,
        reward=ws.outcome.observation.reward,
        prediction_error=pe,
        alignment=-0.25,
        critique="meh",
        salience=new.salience,
    )
    assert result.passes[:-1] == earlier
    record = result.passes[-1]
    assert (record.tick, record.pass_index) == (5, 1)
    assert record.decision == ws.decision
    assert record.outcome == ws.outcome
    assert record.prediction_error == pe
    assert record.feedback == feedback
    assert record.goal_revised is None
    assert record.conflict is None
    assert result.conflict is None
    agent_rest = result.agent.model_copy(update={"memory": ws.agent.memory})
    assert agent_rest == ws.agent


def test_reconcile_clamps_salience_to_one() -> None:
    params = Params(memory=MemoryParams(base_salience=0.9))
    rt = make_rt(params)
    pe = PredictionError(raw_signed=5.0, signed=5.0, unsigned=5.0, event_mismatch=False)
    ws = full_ws(rt).model_copy(update={"prediction_error": pe})
    assert run(reconcile, ws, rt).agent.memory.episodes[-1].salience == 1.0


@pytest.mark.parametrize("missing", ["decision", "outcome", "prediction_error", "feedback"])
def test_reconcile_names_missing_input(missing: str) -> None:
    rt = make_rt()
    ws = full_ws(rt).model_copy(update={missing: None})
    with pytest.raises(ValueError, match=missing):
        run(reconcile, ws, rt)


# --- dispatch -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "phase", "position"),
    [("appraise", appraise, 1), ("deliberate", deliberate, 2), ("evaluate", evaluate, 5)],
)
def test_llm_backend_is_not_implemented_yet(name: str, phase: Phase, position: int) -> None:
    stub_rt = make_rt()
    ws = base_ws(stub_rt)
    for earlier in (perceive, appraise, deliberate, act, monitor, evaluate)[:position]:
        ws = run(earlier, ws, stub_rt)
    llm_rt = make_rt(backends=Backends(**{name: "llm"}))
    with pytest.raises(NotImplementedError, match=name):
        run(phase, ws, llm_rt)


def test_phases_package_exports_only_the_seven_phases() -> None:
    import surprise_loop.phases as phases

    assert sorted(phases.__all__) == sorted(
        ["perceive", "appraise", "deliberate", "act", "monitor", "evaluate", "reconcile"]
    )
