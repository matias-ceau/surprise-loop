"""Evaluate on pydantic-ai: prompt, output mapping, validation and retry, and whole runs with all
three LLM phases. Only TestModel and FunctionModel are used.
"""

import asyncio
from pathlib import Path
from typing import Any

import pytest
from llm_helpers import (
    assert_no_hidden,
    failing_model,
    make_config,
    make_ws,
    respond,
    retry_feedback,
    runtime,
    user_prompt,
)
from pydantic_ai import ModelResponse, ToolCallPart, UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel

from surprise_loop.agents import (
    EVALUATE_INSTRUCTIONS,
    EvaluateOutput,
    appraise_agent,
    deliberate_agent,
    evaluate_agent,
    evaluate_prompt,
)
from surprise_loop.config import (
    MODEL_ENV_VAR,
    Backends,
    ConfigError,
    LLMParams,
    RunConfig,
    load_config,
)
from surprise_loop.loop import load_dumps, replay, run
from surprise_loop.phases import evaluate
from surprise_loop.phases.evaluate import evaluate_llm
from surprise_loop.state import (
    Decision,
    Feedback,
    Prediction,
    PredictionError,
    TickDump,
    Workspace,
)
from surprise_loop.world import Action, Event, Observation, Outcome, PhaseKind, StepInfo

VALID = {"critique": "Pressing B again ignored that B never paid.", "alignment": -0.6}
ALL_LLM = Backends(appraise="llm", deliberate="llm", evaluate="llm")
SMOKE = Path(__file__).resolve().parents[1] / "experiments" / "llm_smoke.yaml"


def config(**kwargs: Any) -> RunConfig:
    kwargs.setdefault("backends", Backends(evaluate="llm"))
    return make_config(run_id="llm-evaluate", **kwargs)


def info(**update: Any) -> StepInfo:
    base = StepInfo(
        step=9,
        phase_index=1,
        phase_kind=PhaseKind.PROGRESSIVE_RATIO,
        ratio_required=3,
        active_press=False,
        ratio_completed=None,
    )
    return base.model_copy(update=update)


def eval_ws(cfg: RunConfig, *, step_info: StepInfo | None = None, step: int = 9) -> Workspace:
    """A workspace as Evaluate sees it: after Deliberate, Act and Monitor."""
    ws = make_ws(cfg)
    observation = Observation(
        step=step, last_action=Action.PRESS_B, event=Event.NO_REWARD, reward=0
    )
    return ws.model_copy(
        update={
            "decision": Decision(
                action=Action.PRESS_B,
                prediction=Prediction(
                    expected_event=Event.REWARD, expected_reward=0.8, confidence=0.7
                ),
                rationale="maybe B pays\nthis time",
            ),
            "outcome": Outcome(observation=observation, info=step_info or info()),
            "prediction_error": PredictionError(
                raw_signed=-0.8, signed=-0.62, unsigned=0.9, event_mismatch=True
            ),
        }
    )


# --- prompt ---------------------------------------------------------------------------------------


def test_prompt_covers_goals_action_prediction_and_outcome() -> None:
    prompt = evaluate_prompt(eval_ws(config()))

    # goals
    for goal, value in [("seek_reward", "1.23"), ("explore", "0.47"), ("conserve", "0.58")]:
        assert f"- {goal}: {value}" in prompt
    # the action and its reasoning, on one line
    assert "Action: press_b." in prompt
    assert "Its reasoning: maybe B pays this time" in prompt
    # the prediction
    assert "Event: reward. Reward: 0.80. Confidence: 0.70." in prompt
    # the actual outcome
    assert "Event: no_reward. Reward: 0.00. Surprise: -0.62" in prompt
    # recent experience, oldest first, without earlier critiques (no echo of the critic)
    assert "1. press_a: got reward, reward 1.00" in prompt
    assert "2. press_b: got no_reward, reward 0.00" in prompt
    assert "wasted effort" not in prompt


def test_prompt_with_no_experience_says_so() -> None:
    ws = eval_ws(config())
    ws = ws.model_copy(update={"percept": ws.percept.model_copy(update={"recalled": []})})
    assert "None: this was its first action." in evaluate_prompt(ws)


def test_prompt_and_instructions_hold_no_hidden_world_information() -> None:
    assert_no_hidden(evaluate_prompt(eval_ws(config())))
    assert_no_hidden(EVALUATE_INSTRUCTIONS)


def test_prompt_does_not_depend_on_step_info_or_the_step_counter() -> None:
    """The outcome's StepInfo and the observation's step never reach the prompt."""
    cfg = config()
    hidden = info(
        step=321,
        phase_index=3,
        phase_kind=PhaseKind.EXTINCTION,
        ratio_required=None,
        active_press=True,
        ratio_completed=17,
    )
    assert evaluate_prompt(eval_ws(cfg)) == evaluate_prompt(
        eval_ws(cfg, step_info=hidden, step=321)
    )


@pytest.mark.parametrize("missing", ["percept", "decision", "outcome", "prediction_error"])
def test_prompt_names_a_missing_input(missing: str) -> None:
    with pytest.raises(ValueError, match=missing):
        evaluate_prompt(eval_ws(config()).model_copy(update={missing: None}))


# --- output model ---------------------------------------------------------------------------------


def test_output_schema_carries_descriptions_and_ranges() -> None:
    schema = EvaluateOutput.model_json_schema()
    props = schema["properties"]
    assert list(props) == ["critique", "alignment"]
    assert all(props[name].get("description") for name in props)
    assert props["alignment"]["minimum"] == -1 and props["alignment"]["maximum"] == 1
    assert props["critique"]["minLength"] == 1
    assert schema["additionalProperties"] is False


def test_critique_is_collapsed_to_one_line() -> None:
    out = EvaluateOutput.model_validate({**VALID, "critique": " Good.\n\nKeep  going. "})
    assert out.critique == "Good. Keep going."


# --- mapping and purity ---------------------------------------------------------------------------


def test_valid_response_maps_to_feedback_and_nothing_else_changes() -> None:
    cfg = config(llm=LLMParams(output_retries=2, temperature=0.2, timeout_s=7.0))
    ws = eval_ws(cfg)
    seen: list[AgentInfo] = []

    def function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info)
        assert user_prompt(messages) == evaluate_prompt(ws)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, VALID)])

    out = asyncio.run(evaluate_llm(ws, runtime(cfg, FunctionModel(function))))

    assert out.feedback == Feedback(alignment=-0.6, critique=VALID["critique"])
    assert out.agent == ws.agent
    assert out == ws.model_copy(update={"feedback": out.feedback})
    [info_] = seen
    assert info_.instructions == EVALUATE_INSTRUCTIONS
    assert info_.output_tools[0].name == "final_result"
    assert info_.model_settings is not None
    assert info_.model_settings.get("temperature") == 0.2
    assert info_.model_settings.get("timeout") == 7.0


def test_dispatcher_selects_the_llm_backend_and_leaves_the_agent_alone() -> None:
    cfg = config()
    ws = eval_ws(cfg)
    before = ws.agent.model_dump()
    model, calls = respond(VALID)

    out = asyncio.run(evaluate(ws, runtime(cfg, model)))

    assert len(calls) == 1
    assert out.feedback is not None and out.feedback.critique == VALID["critique"]
    assert out.agent == ws.agent
    assert ws.agent.model_dump() == before  # the input was not mutated either


def test_llm_backend_without_a_runtime_model_raises_config_error() -> None:
    cfg = config()
    with pytest.raises(ConfigError, match="no resolved model"):
        asyncio.run(evaluate_llm(eval_ws(cfg), runtime(cfg, None)))


# --- validation and retry -------------------------------------------------------------------------

INVALID = [
    pytest.param({**VALID, "alignment": 1.5}, "alignment", id="alignment-above-1"),
    pytest.param({**VALID, "alignment": -1.01}, "alignment", id="alignment-below-minus-1"),
    pytest.param({**VALID, "alignment": "good"}, "alignment", id="not-a-number"),
    pytest.param({**VALID, "critique": " \n"}, "critique", id="blank-critique"),
    pytest.param({**VALID, "score": 3}, "score", id="extra-field"),
    pytest.param({"alignment": 0.2}, "critique", id="missing-critique"),
]


@pytest.mark.parametrize(("bad", "field"), INVALID)
def test_invalid_output_is_retried_then_accepted(bad: dict[str, Any], field: str) -> None:
    cfg = config(llm=LLMParams(output_retries=2))
    model, calls = respond(bad, VALID)

    out = asyncio.run(evaluate_llm(eval_ws(cfg), runtime(cfg, model)))

    assert len(calls) == 2
    assert field in retry_feedback(calls[1])
    assert out.feedback == Feedback(**VALID)


@pytest.mark.parametrize("retries", [0, 1, 3])
@pytest.mark.parametrize(("bad", "field"), INVALID[:3])
def test_persistently_invalid_output_raises_after_the_retry_budget(
    bad: dict[str, Any], field: str, retries: int
) -> None:
    cfg = config(llm=LLMParams(output_retries=retries))
    model, calls = respond(bad)

    with pytest.raises(UnexpectedModelBehavior, match="output retries") as err:
        asyncio.run(evaluate_llm(eval_ws(cfg), runtime(cfg, model)))

    assert len(calls) == retries + 1
    assert any("evaluate failed at tick 8, pass 0" in n for n in err.value.__notes__)


def test_invalid_output_stops_the_run_without_stub_fallback(tmp_path: Path) -> None:
    model, _ = respond(VALID, VALID, VALID, {**VALID, "alignment": 9.0})
    run_dir = tmp_path / "run"

    with evaluate_agent.override(model=model), pytest.raises(UnexpectedModelBehavior):
        asyncio.run(run(config(llm=LLMParams(output_retries=0), ticks=10), run_dir))

    dumps = load_dumps(run_dir)
    assert dumps, "the ticks before the failure are on disk"
    feedback = [p.feedback for d in dumps for p in d.record.passes]
    assert all(f == Feedback(**VALID) for f in feedback)


# --- whole runs -----------------------------------------------------------------------------------


def test_twenty_tick_run_with_all_three_llm_phases_writes_valid_dumps_and_replays(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    result = asyncio.run(run(config(backends=ALL_LLM, model="test", ticks=20), run_dir))

    assert len(result.records) == 20
    assert [d.tick for d in load_dumps(run_dir)] == list(range(1, 21))
    for path in sorted(run_dir.glob("tick_*.json")):
        record = TickDump.model_validate_json(path.read_bytes()).record
        assert record.appraisal.note  # the stubs leave note, rationale and critique empty/"stub"
        for p in record.passes:
            assert p.decision.rationale and p.decision.rationale != "stub"
            assert p.feedback.critique
            assert -1 <= p.feedback.alignment <= 1

    # Replay uses the recorded outputs: models that fail on any call prove none is made.
    with (
        appraise_agent.override(model=failing_model()),
        deliberate_agent.override(model=failing_model()),
        evaluate_agent.override(model=failing_model()),
    ):
        assert asyncio.run(replay(run_dir)) == 20


def test_model_text_flows_into_later_prompts_but_hidden_information_never_does() -> None:
    """The critique and the appraisal note are stored and shown to later phases; every prompt
    actually sent, by any of the three agents, is free of hidden world information."""
    appraise_model, appraise_calls = respond({"note": "NOTE-MARK", "valence": 0, "arousal": 0.5})
    deliberate_model, deliberate_calls = respond(
        {
            "reasoning": "REASON-MARK",
            "action": "press_a",
            "expected_event": "reward",
            "expected_reward": 1.0,
            "confidence": 0.5,
        }
    )
    evaluate_model, evaluate_calls = respond({"critique": "CRITIQUE-MARK", "alignment": 0.3})
    with (
        appraise_agent.override(model=appraise_model),
        deliberate_agent.override(model=deliberate_model),
        evaluate_agent.override(model=evaluate_model),
    ):
        result = asyncio.run(run(config(backends=ALL_LLM, ticks=25)))

    passes = sum(len(r.passes) for r in result.records)
    assert len(appraise_calls) == 25
    assert len(deliberate_calls) == len(evaluate_calls) == passes
    prompts = {
        "appraise": [user_prompt(m) for m in appraise_calls],
        "deliberate": [user_prompt(m) for m in deliberate_calls],
        "evaluate": [user_prompt(m) for m in evaluate_calls],
    }
    for texts in prompts.values():
        for text in texts:
            assert_no_hidden(text)
    # the critique reaches later Deliberate and Appraise prompts through memory
    assert any("CRITIQUE-MARK" in p for p in prompts["deliberate"])
    assert any("CRITIQUE-MARK" in p for p in prompts["appraise"])
    # the note reaches Deliberate; Deliberate's reasoning reaches Evaluate
    assert all("NOTE-MARK" in p for p in prompts["deliberate"])
    assert all("REASON-MARK" in p for p in prompts["evaluate"])


# --- the real-model smoke config ------------------------------------------------------------------


def test_llm_smoke_config_runs_every_phase_on_the_llm_and_crosses_fr_and_pr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = load_config(SMOKE)
    assert experiment.ticks == 40
    assert experiment.model is None
    assert experiment.backends == ALL_LLM
    [cfg] = experiment.runs()

    monkeypatch.setenv(MODEL_ENV_VAR, "test")
    result = asyncio.run(run(cfg))

    assert result.config.model == "test"
    assert len(result.records) == 40
    kinds = [p.outcome.info.phase_kind for r in result.records for p in r.passes]
    assert PhaseKind.FIXED_RATIO in kinds and PhaseKind.PROGRESSIVE_RATIO in kinds
    assert kinds.index(PhaseKind.PROGRESSIVE_RATIO) < 40 // 2  # PR is reached well before the end
