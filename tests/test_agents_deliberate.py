"""Deliberate on pydantic-ai: prompt, output mapping, validation and retry, and whole runs.

Only TestModel and FunctionModel are used; conftest.py forbids requests to any real model.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import ModelResponse, ToolCallPart, UnexpectedModelBehavior, models
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from surprise_loop.agents import (
    DELIBERATE_INSTRUCTIONS,
    DeliberateOutput,
    deliberate_agent,
    deliberate_prompt,
)
from surprise_loop.config import (
    MODEL_ENV_VAR,
    AgentSpec,
    Backends,
    ConfigError,
    ConflictParams,
    LLMParams,
    Params,
    RunConfig,
    Runtime,
    initial_agent_state,
)
from surprise_loop.loop import load_dumps, load_model, replay, run
from surprise_loop.phases import deliberate
from surprise_loop.phases.deliberate import deliberate_llm
from surprise_loop.state import (
    Appraisal,
    Conflict,
    Decision,
    Episode,
    GoalName,
    Mood,
    Percept,
    Prediction,
    PredictionError,
    TickDump,
    Traits,
    Workspace,
)
from surprise_loop.world import (
    Action,
    Event,
    Observation,
    PhaseKind,
    PhaseSpec,
    SelfAdminWorld,
    WorldConfig,
)

HIDDEN = ["phase_kind", "ratio_required", "active_press", "ratio_completed", "active_lever"]
HIDDEN += [kind.value for kind in PhaseKind] + ["step"]

TRAITS = Traits(reward_sensitivity=1.37, surprise_learning_rate=0.41, mood_inertia=0.63)
PRIORITIES = {GoalName.SEEK_REWARD: 1.23, GoalName.EXPLORE: 0.47, GoalName.CONSERVE: 0.58}
VALID = {
    "reasoning": "lever A paid twice in a row",
    "action": "press_a",
    "expected_event": "reward",
    "expected_reward": 0.8,
    "confidence": 0.7,
}


def make_config(
    *,
    backends: Backends | None = None,
    model: str | None = "test",
    llm: LLMParams | None = None,
    ticks: int | None = 20,
) -> RunConfig:
    schedule = [
        PhaseSpec(kind=PhaseKind.FIXED_RATIO, steps=10, ratio=1),
        PhaseSpec(kind=PhaseKind.PROGRESSIVE_RATIO, steps=10, pr_start=1, pr_increment=1),
        PhaseSpec(kind=PhaseKind.EXTINCTION, steps=20),
    ]
    return RunConfig(
        run_id="llm-deliberate",
        seed=5,
        ticks=ticks,
        world=WorldConfig(active_lever="random", reward_magnitude=1.0, schedule=schedule),
        agent=AgentSpec(name="a", traits=TRAITS, goal_priorities=PRIORITIES),
        params=Params(llm=llm or LLMParams()),
        backends=backends or Backends(deliberate="llm"),
        model=model,
    )


def episode(tick: int, action: Action, event: Event, reward: float, critique: str) -> Episode:
    return Episode(
        tick=tick,
        step=tick,
        action=action,
        prediction=Prediction(expected_event=Event.REWARD, expected_reward=0.9, confidence=0.35),
        event=event,
        reward=reward,
        prediction_error=PredictionError(
            raw_signed=reward - 0.9, signed=-0.53, unsigned=0.74, event_mismatch=True
        ),
        alignment=-0.27,
        critique=critique,
        salience=0.84 if reward == 0 else 0.19,
    )


def make_ws(config: RunConfig, *, conflict: bool = False) -> Workspace:
    agent = initial_agent_state(config)
    agent = agent.model_copy(
        update={
            "mood": Mood(valence=-0.31, arousal=0.72),
            "dispositions": agent.dispositions.model_copy(
                update={
                    "biases": {
                        Action.PRESS_A: 0.44,
                        Action.PRESS_B: -0.12,
                        Action.EXPLORE: 0.06,
                        Action.REST: -0.09,
                    }
                }
            ),
            "tick": 7,
        }
    )
    recalled = [
        episode(5, Action.PRESS_A, Event.REWARD, 1.0, "good, that served seek_reward"),
        episode(6, Action.PRESS_B, Event.NO_REWARD, 0.0, "wasted effort on lever B"),
    ]
    observation = Observation(step=8, last_action=Action.PRESS_B, event=Event.NO_REWARD, reward=0)
    return Workspace(
        agent=agent,
        observation=observation,
        percept=Percept(
            event=Event.NO_REWARD, reward=0.0, last_action=Action.PRESS_B, recalled=recalled
        ),
        appraisal=Appraisal(valence=-0.42, arousal=0.66, note="disappointed by lever B"),
        conflict=(
            Conflict(kind="habit_vs_goal", action=Action.PRESS_A, detail="habit detail xyz")
            if conflict
            else None
        ),
    )


def runtime(config: RunConfig, model: object) -> Runtime:
    return Runtime(config=config, world=SelfAdminWorld(config.world, config.seed), model=model)


def respond(*outputs: dict[str, Any]) -> tuple[FunctionModel, list[list[ModelMessage]]]:
    """A FunctionModel answering with `outputs` in turn (the last one repeats); records calls."""
    calls: list[list[ModelMessage]] = []

    def function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(list(messages))
        args = outputs[min(len(calls), len(outputs)) - 1]
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])

    return FunctionModel(function), calls


def user_prompt(messages: list[ModelMessage]) -> str:
    first = messages[0]
    assert isinstance(first, ModelRequest)
    parts = [p for p in first.parts if isinstance(p, UserPromptPart)]
    assert len(parts) == 1
    assert isinstance(parts[0].content, str)
    return parts[0].content


# --- suite guard ----------------------------------------------------------------------------------


def test_the_suite_forbids_real_model_requests() -> None:
    assert models.ALLOW_MODEL_REQUESTS is False


# --- prompt ---------------------------------------------------------------------------------------


def test_prompt_contains_every_state_element() -> None:
    ws = make_ws(make_config(), conflict=True)
    prompt = deliberate_prompt(ws)

    # observation (as perceived)
    assert "Last action: press_b. Event: no_reward. Reward: 0.00." in prompt
    # mood
    assert "-0.31" in prompt and "0.72" in prompt
    # appraisal
    assert "-0.42" in prompt and "0.66" in prompt and "disappointed by lever B" in prompt
    # goal priorities
    for goal, value in [("seek_reward", "1.23"), ("explore", "0.47"), ("conserve", "0.58")]:
        assert f"- {goal}: {value}" in prompt
    # dispositions
    for action, value in [("press_a", "+0.44"), ("press_b", "-0.12"), ("explore", "+0.06")]:
        assert f"- {action}: {value}" in prompt
    assert "- rest: -0.09" in prompt
    # traits
    assert "reward_sensitivity: 1.37" in prompt
    assert "surprise_learning_rate: 0.41" in prompt
    assert "mood_inertia: 0.63" in prompt
    # recalled episodes, with salience and critique, oldest first
    first = prompt.index("good, that served seek_reward")
    second = prompt.index("wasted effort on lever B")
    assert first < second
    assert "salience 0.19" in prompt and "salience 0.84" in prompt
    assert "surprise -0.53" in prompt and "alignment -0.27" in prompt
    assert "predicted reward, reward 0.90 (confidence 0.35)" in prompt
    # conflict note
    assert "## Conflict" in prompt and "habit of press_a" in prompt
    assert "habit detail xyz" in prompt


def test_prompt_has_no_conflict_note_without_a_conflict() -> None:
    prompt = deliberate_prompt(make_ws(make_config()))
    assert "Conflict" not in prompt
    assert "habit detail" not in prompt


def test_prompt_with_no_memory_says_so() -> None:
    ws = make_ws(make_config())
    ws = ws.model_copy(update={"percept": ws.percept.model_copy(update={"recalled": []})})
    assert "None yet" in deliberate_prompt(ws)


@pytest.mark.parametrize("conflict", [False, True])
def test_prompt_and_instructions_hold_no_hidden_world_information(conflict: bool) -> None:
    ws = make_ws(make_config(), conflict=conflict)
    for text in (deliberate_prompt(ws), DELIBERATE_INSTRUCTIONS):
        for hidden in HIDDEN:
            assert hidden not in text.lower()


def test_prompts_sent_during_a_run_hold_no_hidden_world_information() -> None:
    model, calls = respond(VALID)
    config = make_config(ticks=25)
    with deliberate_agent.override(model=model):
        asyncio.run(run(config))

    assert len(calls) >= 25
    for messages in calls:
        prompt = user_prompt(messages)
        for hidden in HIDDEN:
            assert hidden not in prompt.lower()


def test_prompt_requires_percept_and_appraisal() -> None:
    ws = make_ws(make_config())
    with pytest.raises(ValueError, match="percept"):
        deliberate_prompt(ws.model_copy(update={"percept": None}))
    with pytest.raises(ValueError, match="appraisal"):
        deliberate_prompt(ws.model_copy(update={"appraisal": None}))


# --- output model ---------------------------------------------------------------------------------


def test_output_schema_carries_descriptions_and_ranges() -> None:
    schema = DeliberateOutput.model_json_schema()
    props = schema["properties"]
    assert list(props) == [
        "reasoning",
        "action",
        "expected_event",
        "expected_reward",
        "confidence",
    ]
    assert all(props[name].get("description") for name in props)
    assert props["confidence"]["minimum"] == 0 and props["confidence"]["maximum"] == 1
    assert props["expected_reward"]["minimum"] == 0
    assert sorted(props["expected_event"]["enum"]) == sorted(
        ["reward", "no_reward", "explored", "rested"]
    )
    assert schema["additionalProperties"] is False


# --- mapping --------------------------------------------------------------------------------------


def test_valid_response_maps_to_the_decision() -> None:
    config = make_config(llm=LLMParams(output_retries=2, temperature=0.4, timeout_s=12.5))
    ws = make_ws(config)
    seen: list[AgentInfo] = []

    def function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info)
        assert user_prompt(messages) == deliberate_prompt(ws)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, VALID)])

    out = asyncio.run(deliberate_llm(ws, runtime(config, FunctionModel(function))))

    assert out.decision == Decision(
        action=Action.PRESS_A,
        prediction=Prediction(expected_event=Event.REWARD, expected_reward=0.8, confidence=0.7),
        rationale="lever A paid twice in a row",
    )
    assert out == ws.model_copy(update={"decision": out.decision})
    [info] = seen
    assert info.instructions == DELIBERATE_INSTRUCTIONS
    assert info.output_tools[0].name == "final_result"
    assert info.model_settings is not None
    assert info.model_settings.get("temperature") == 0.4
    assert info.model_settings.get("timeout") == 12.5


def test_temperature_null_sends_no_temperature() -> None:
    config = make_config(llm=LLMParams(temperature=None))
    seen: list[AgentInfo] = []

    def function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, VALID)])

    asyncio.run(deliberate_llm(make_ws(config), runtime(config, FunctionModel(function))))
    assert seen[0].model_settings is not None
    assert "temperature" not in seen[0].model_settings


def test_dispatcher_selects_the_llm_backend() -> None:
    config = make_config()
    model, calls = respond(VALID)
    out = asyncio.run(deliberate(make_ws(config), runtime(config, model)))
    assert len(calls) == 1
    assert out.decision is not None and out.decision.rationale == VALID["reasoning"]


def test_llm_backend_without_a_runtime_model_raises_config_error() -> None:
    config = make_config()
    with pytest.raises(ConfigError, match="no resolved model"):
        asyncio.run(deliberate_llm(make_ws(config), runtime(config, None)))


# --- validation and retry -------------------------------------------------------------------------

INVALID = [
    pytest.param({**VALID, "confidence": 1.5}, "confidence", id="confidence-above-1"),
    pytest.param({**VALID, "confidence": -0.1}, "confidence", id="confidence-below-0"),
    pytest.param({**VALID, "action": "pull_lever_c"}, "action", id="unknown-action"),
    pytest.param({**VALID, "expected_event": "start"}, "expected_event", id="start-event"),
    pytest.param({**VALID, "expected_reward": -1.0}, "expected_reward", id="negative-reward"),
    pytest.param({**VALID, "mood": "fine"}, "mood", id="extra-field"),
    pytest.param({k: v for k, v in VALID.items() if k != "action"}, "action", id="missing-field"),
    # action/prediction consistency (the model validator)
    pytest.param({**VALID, "expected_event": "explored"}, "can only produce", id="press-explored"),
    pytest.param(
        {**VALID, "action": "press_b", "expected_event": "rested"},
        "can only produce",
        id="press-rested",
    ),
    pytest.param(
        {**VALID, "action": "explore", "expected_event": "reward", "expected_reward": 0},
        "always produces explored",
        id="explore-reward-event",
    ),
    pytest.param(
        {**VALID, "action": "explore", "expected_event": "explored", "expected_reward": 0.5},
        "never delivers a reward",
        id="explore-positive-reward",
    ),
    pytest.param(
        {**VALID, "action": "rest", "expected_event": "no_reward", "expected_reward": 0},
        "always produces rested",
        id="rest-no-reward-event",
    ),
    pytest.param(
        {**VALID, "action": "rest", "expected_event": "rested", "expected_reward": 0.2},
        "never delivers a reward",
        id="rest-positive-reward",
    ),
]


@pytest.mark.parametrize(
    ("action", "event", "reward"),
    [
        ("press_a", "reward", 1.0),
        ("press_a", "no_reward", 0.0),
        ("press_b", "no_reward", 0.3),
        ("explore", "explored", 0.0),
        ("rest", "rested", 0.0),
    ],
)
def test_consistent_predictions_are_accepted(action: str, event: str, reward: float) -> None:
    out = DeliberateOutput.model_validate(
        {**VALID, "action": action, "expected_event": event, "expected_reward": reward}
    )
    assert out.action == action and out.expected_event == event


def test_params_llm_defaults() -> None:
    assert LLMParams() == LLMParams(output_retries=3, temperature=1.0, timeout_s=60.0)


def retry_feedback(messages: list[ModelMessage]) -> str:
    last = messages[-1]
    assert isinstance(last, ModelRequest)
    returns = [p for p in last.parts if isinstance(p, ToolReturnPart) and p.outcome == "retried"]
    assert returns, "expected a retry prompt for the output tool"
    return json.dumps([str(p.content) for p in returns])


@pytest.mark.parametrize(("bad", "field"), INVALID)
def test_invalid_output_is_retried_then_accepted(bad: dict[str, Any], field: str) -> None:
    config = make_config(llm=LLMParams(output_retries=2))
    model, calls = respond(bad, VALID)

    out = asyncio.run(deliberate_llm(make_ws(config), runtime(config, model)))

    assert len(calls) == 2
    assert field in retry_feedback(calls[1])
    assert out.decision is not None
    assert out.decision.prediction.confidence == VALID["confidence"]
    assert out.decision.action == Action.PRESS_A
    assert out.decision.prediction.expected_event == Event.REWARD


@pytest.mark.parametrize("retries", [0, 1, 3])
@pytest.mark.parametrize(("bad", "field"), INVALID[:3])
def test_persistently_invalid_output_raises_after_the_retry_budget(
    bad: dict[str, Any], field: str, retries: int
) -> None:
    config = make_config(llm=LLMParams(output_retries=retries))
    model, calls = respond(bad)
    ws = make_ws(config)

    with pytest.raises(UnexpectedModelBehavior, match="output retries") as info:
        asyncio.run(deliberate_llm(ws, runtime(config, model)))

    assert len(calls) == retries + 1
    assert any("tick 8, pass 0" in note for note in info.value.__notes__)


def test_invalid_output_stops_the_run_without_stub_fallback(tmp_path: Path) -> None:
    model, calls = respond(VALID, VALID, VALID, {**VALID, "confidence": 7.0})
    config = make_config(llm=LLMParams(output_retries=1))
    run_dir = tmp_path / "run"

    with deliberate_agent.override(model=model), pytest.raises(UnexpectedModelBehavior):
        asyncio.run(run(config, run_dir))

    # The ticks before the failure are on disk, the failing one is not: resume picks up there.
    dumps = load_dumps(run_dir)
    assert [d.tick for d in dumps] == [1, 2, 3]
    assert all(p.decision.rationale == VALID["reasoning"] for d in dumps for p in d.record.passes)


# --- whole runs -----------------------------------------------------------------------------------


def test_the_string_test_resolves_to_test_model() -> None:
    assert isinstance(load_model("test"), TestModel)
    assert load_model(None) is None


def test_twenty_tick_llm_run_writes_valid_dumps_and_replays(tmp_path: Path) -> None:
    config = make_config(model="test", ticks=20)
    run_dir = tmp_path / "run"

    result = asyncio.run(run(config, run_dir))

    assert len(result.records) == 20
    dumps = load_dumps(run_dir)
    assert [d.tick for d in dumps] == list(range(1, 21))
    for path in sorted(run_dir.glob("tick_*.json")):
        dump = TickDump.model_validate_json(path.read_bytes())
        for record in dump.record.passes:
            assert record.decision.rationale  # came from the model, not the stub
            assert record.decision.rationale != "stub"
            assert 0 <= record.decision.prediction.confidence <= 1
    assert json.loads((run_dir / "config.json").read_text())["model"] == "test"

    # Replay uses the recorded decisions: a model that fails on any call proves none is made.
    def boom(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise AssertionError("replay must not call the model")

    with deliberate_agent.override(model=FunctionModel(boom)):
        assert asyncio.run(replay(run_dir)) == 20


def test_model_is_resolved_once_per_run(monkeypatch: pytest.MonkeyPatch) -> None:
    inferred: list[str] = []
    real_infer = models.infer_model

    def counting_infer(name: str) -> models.Model:
        inferred.append(name)
        return real_infer(name)

    monkeypatch.setattr(models, "infer_model", counting_infer)
    asyncio.run(run(make_config(model="test", ticks=10)))
    assert inferred == ["test"]


def test_model_from_the_environment_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MODEL_ENV_VAR, "test")
    result = asyncio.run(run(make_config(model=None, ticks=3)))
    assert result.config.model == "test"
    assert len(result.records) == 3


def test_llm_backend_with_no_model_fails_before_tick_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
    model, calls = respond(VALID)
    run_dir = tmp_path / "run"

    with deliberate_agent.override(model=model), pytest.raises(ConfigError, match=MODEL_ENV_VAR):
        asyncio.run(run(make_config(model=None), run_dir))

    assert calls == []
    assert not run_dir.exists()


def test_stub_backend_needs_no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
    result = asyncio.run(run(make_config(backends=Backends(), model=None, ticks=5)))
    assert result.config.model is None
    assert all(p.decision.rationale == "stub" for r in result.records for p in r.passes)


def test_a_real_reconcile_conflict_reaches_the_prompt_without_hidden_information() -> None:
    """Conflicts come from the real Reconcile: patience 1 and a habit threshold every bias meets,
    with a model that always over-predicts lever A, so seek_reward is soon revised down."""
    over_predict = {**VALID, "expected_reward": 5.0, "confidence": 1.0}
    model, calls = respond(over_predict)
    base = make_config(ticks=30)
    config = base.model_copy(
        update={
            "world": base.world.model_copy(update={"active_lever": "press_a"}),
            "params": base.params.model_copy(
                update={
                    "goals": base.params.goals.model_copy(update={"patience": 1}),
                    "conflict": ConflictParams(enabled=True, habit_threshold=-1.0),
                }
            ),
        }
    )
    with deliberate_agent.override(model=model):
        result = asyncio.run(run(config))

    conflicts = [p.conflict for r in result.records for p in r.passes if p.conflict]
    assert conflicts, "the setup should produce at least one real conflict"
    prompts = [user_prompt(messages) for messages in calls]
    with_note = [p for p in prompts if "## Conflict" in p]
    assert len(with_note) == len(conflicts)
    for prompt in with_note:
        assert "the priority of goal seek_reward was just lowered to" in prompt
        for hidden in HIDDEN:
            assert hidden not in prompt.lower()
