"""Appraise on pydantic-ai: prompt, output mapping, validation and retry, and whole runs.

Only TestModel and FunctionModel are used; conftest.py forbids requests to any real model.
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
    APPRAISE_INSTRUCTIONS,
    AppraiseOutput,
    appraise_agent,
    appraise_prompt,
    deliberate_agent,
)
from surprise_loop.config import Backends, ConfigError, ConflictParams, LLMParams, RunConfig
from surprise_loop.loop import load_dumps, replay, run
from surprise_loop.phases import appraise
from surprise_loop.phases.appraise import appraise_llm
from surprise_loop.state import Appraisal, Memory, TickDump
from surprise_loop.world import Action, Event, Observation

VALID = {"note": "stung by the empty lever", "valence": -0.45, "arousal": 0.61}
APPRAISE = Backends(appraise="llm")


def config(**kwargs: Any) -> RunConfig:
    kwargs.setdefault("backends", APPRAISE)
    return make_config(run_id="llm-appraise", **kwargs)


# --- prompt ---------------------------------------------------------------------------------------


def test_prompt_contains_every_state_element() -> None:
    prompt = appraise_prompt(make_ws(config()))

    # observation (as perceived) and the prediction it is compared with
    assert "Last action: press_b. Event: no_reward. Reward: 0.00." in prompt
    assert (
        "You had predicted event 'reward' with reward 0.90 (confidence 0.35). Surprise: -0.53 "
        "(actual minus expected reward, weighted by your confidence; negative = worse than "
        "expected)." in prompt
    )
    # mood
    assert "Valence: -0.31" in prompt and "Arousal: 0.72" in prompt
    # goals
    for goal, value in [("seek_reward", "1.23"), ("explore", "0.47"), ("conserve", "0.58")]:
        assert f"- {goal}: {value}" in prompt
    # traits
    assert "reward_sensitivity: 1.37" in prompt
    assert "surprise_learning_rate: 0.41" in prompt
    assert "mood_inertia: 0.63" in prompt
    # recalled episodes, oldest first, with critique and salience
    assert prompt.index("good, that served seek_reward") < prompt.index("wasted effort on lever B")
    assert "salience 0.84" in prompt
    # not Deliberate's business: the appraisal it is producing, habits, conflicts
    assert "appraisal" not in prompt.lower()
    assert "habit" not in prompt.lower()


def test_prompt_at_the_start_says_nothing_has_happened() -> None:
    ws = make_ws(config())
    agent = ws.agent.model_copy(update={"memory": Memory(episodes=[]), "tick": 0})
    percept = ws.percept.model_copy(
        update={"recalled": [], "last_action": None, "event": Event.START}
    )
    prompt = appraise_prompt(ws.model_copy(update={"agent": agent, "percept": percept}))
    assert "Last action: none yet. Event: start." in prompt
    assert "first moment" in prompt
    assert "None yet" in prompt
    assert "You had predicted" not in prompt


def test_prompt_and_instructions_hold_no_hidden_world_information() -> None:
    assert_no_hidden(appraise_prompt(make_ws(config())))
    assert_no_hidden(APPRAISE_INSTRUCTIONS)


def test_prompt_does_not_depend_on_the_world_observation() -> None:
    """Only the percept crosses into the prompt: the observation's step counter does not."""
    ws = make_ws(config())
    other = Observation(step=987, last_action=None, event=Event.REWARD, reward=42.0)
    assert appraise_prompt(ws) == appraise_prompt(ws.model_copy(update={"observation": other}))


def test_prompts_sent_during_a_run_hold_no_hidden_world_information() -> None:
    model, calls = respond(VALID)
    with appraise_agent.override(model=model):
        result = asyncio.run(run(config(ticks=25)))

    assert len(calls) == len(result.records) == 25
    for messages in calls:
        assert_no_hidden(user_prompt(messages))


@pytest.mark.parametrize(
    ("last_action", "memory_action"),
    [(Action.PRESS_A, Action.PRESS_B), (None, Action.PRESS_B), (Action.PRESS_B, None)],
)
def test_prompt_refuses_a_percept_that_does_not_match_the_newest_episode(
    last_action: Action | None, memory_action: Action | None
) -> None:
    ws = make_ws(config())
    episodes = ws.agent.memory.episodes
    if memory_action is None:
        episodes = []
    else:
        episodes = [*episodes[:-1], episodes[-1].model_copy(update={"action": memory_action})]
    agent = ws.agent.model_copy(update={"memory": Memory(episodes=episodes)})
    percept = ws.percept.model_copy(update={"last_action": last_action})
    with pytest.raises(ValueError, match="newest episode in memory"):
        appraise_prompt(ws.model_copy(update={"agent": agent, "percept": percept}))


def test_a_real_run_with_re_entry_never_trips_the_episode_guard() -> None:
    """Real Reconcile with patience 1 and a habit threshold every bias meets, and a Deliberate that
    always over-predicts lever A: re-entries happen, and every tick's appraisal prompt shows the
    prediction of the action actually perceived (the second pass's after a re-entry)."""
    over_predict = {
        "reasoning": "A again",
        "action": "press_a",
        "expected_event": "reward",
        "expected_reward": 5.0,
        "confidence": 1.0,
    }
    deliberate_model, _ = respond(over_predict)
    appraise_model, calls = respond(VALID)
    base = config(ticks=30, backends=Backends(appraise="llm", deliberate="llm"))
    cfg = base.model_copy(
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
    with (
        deliberate_agent.override(model=deliberate_model),
        appraise_agent.override(model=appraise_model),
    ):
        result = asyncio.run(run(cfg))

    # The world (40 steps) can end before 30 ticks: re-entries take extra steps.
    assert len(result.records) >= 20
    assert any(len(r.passes) == 2 for r in result.records), "the setup should re-enter"
    assert len(calls) == len(result.records)
    for messages in calls[1:]:
        assert "You had predicted event 'reward' with reward 5.00" in user_prompt(messages)


def test_prompt_requires_a_percept() -> None:
    with pytest.raises(ValueError, match="percept"):
        appraise_prompt(make_ws(config()).model_copy(update={"percept": None}))


# --- output model ---------------------------------------------------------------------------------


def test_output_schema_carries_descriptions_and_ranges() -> None:
    schema = AppraiseOutput.model_json_schema()
    props = schema["properties"]
    assert list(props) == ["note", "valence", "arousal"]
    assert all(props[name].get("description") for name in props)
    assert props["valence"]["minimum"] == -1 and props["valence"]["maximum"] == 1
    assert props["arousal"]["minimum"] == 0 and props["arousal"]["maximum"] == 1
    assert props["note"]["minLength"] == 1
    assert schema["additionalProperties"] is False


def test_note_is_collapsed_to_one_line() -> None:
    out = AppraiseOutput.model_validate({**VALID, "note": "  a jolt\n\nof   hope  "})
    assert out.note == "a jolt of hope"


# --- mapping --------------------------------------------------------------------------------------


def test_valid_response_maps_to_the_appraisal() -> None:
    cfg = config(llm=LLMParams(output_retries=2, temperature=0.3, timeout_s=9.0))
    ws = make_ws(cfg).model_copy(update={"appraisal": None})
    seen: list[AgentInfo] = []

    def function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info)
        assert user_prompt(messages) == appraise_prompt(ws)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, VALID)])

    out = asyncio.run(appraise_llm(ws, runtime(cfg, FunctionModel(function))))

    assert out.appraisal == Appraisal(valence=-0.45, arousal=0.61, note="stung by the empty lever")
    assert out == ws.model_copy(update={"appraisal": out.appraisal})
    [info] = seen
    assert info.instructions == APPRAISE_INSTRUCTIONS
    assert info.output_tools[0].name == "final_result"
    assert info.model_settings is not None
    assert info.model_settings.get("temperature") == 0.3
    assert info.model_settings.get("timeout") == 9.0


def test_dispatcher_selects_the_llm_backend() -> None:
    cfg = config()
    model, calls = respond(VALID)
    out = asyncio.run(appraise(make_ws(cfg), runtime(cfg, model)))
    assert len(calls) == 1
    assert out.appraisal is not None and out.appraisal.note == VALID["note"]


def test_llm_backend_without_a_runtime_model_raises_config_error() -> None:
    cfg = config()
    with pytest.raises(ConfigError, match="no resolved model"):
        asyncio.run(appraise_llm(make_ws(cfg), runtime(cfg, None)))


# --- validation and retry -------------------------------------------------------------------------

INVALID = [
    pytest.param({**VALID, "valence": 1.5}, "valence", id="valence-above-1"),
    pytest.param({**VALID, "valence": -1.2}, "valence", id="valence-below-minus-1"),
    pytest.param({**VALID, "arousal": -0.1}, "arousal", id="arousal-below-0"),
    pytest.param({**VALID, "arousal": 1.3}, "arousal", id="arousal-above-1"),
    pytest.param({**VALID, "valence": "happy"}, "valence", id="not-a-number"),
    pytest.param({**VALID, "note": "  \n "}, "note", id="blank-note"),
    pytest.param({**VALID, "mood": "fine"}, "mood", id="extra-field"),
    pytest.param({k: v for k, v in VALID.items() if k != "note"}, "note", id="missing-note"),
]


@pytest.mark.parametrize(("bad", "field"), INVALID)
def test_invalid_output_is_retried_then_accepted(bad: dict[str, Any], field: str) -> None:
    cfg = config(llm=LLMParams(output_retries=2))
    model, calls = respond(bad, VALID)

    out = asyncio.run(appraise_llm(make_ws(cfg), runtime(cfg, model)))

    assert len(calls) == 2
    assert field in retry_feedback(calls[1])
    assert out.appraisal == Appraisal(**VALID)


@pytest.mark.parametrize("retries", [0, 1, 3])
@pytest.mark.parametrize(("bad", "field"), INVALID[:4])
def test_persistently_invalid_output_raises_after_the_retry_budget(
    bad: dict[str, Any], field: str, retries: int
) -> None:
    cfg = config(llm=LLMParams(output_retries=retries))
    model, calls = respond(bad)

    with pytest.raises(UnexpectedModelBehavior, match="output retries") as info:
        asyncio.run(appraise_llm(make_ws(cfg), runtime(cfg, model)))

    assert len(calls) == retries + 1
    assert any("appraise failed at tick 8, pass 0" in n for n in info.value.__notes__)


def test_invalid_output_stops_the_run_without_stub_fallback(tmp_path: Path) -> None:
    model, _ = respond(VALID, VALID, {**VALID, "arousal": 2.0})
    run_dir = tmp_path / "run"

    with appraise_agent.override(model=model), pytest.raises(UnexpectedModelBehavior):
        asyncio.run(run(config(llm=LLMParams(output_retries=0)), run_dir))

    dumps = load_dumps(run_dir)
    assert [d.tick for d in dumps] == [1, 2]
    assert all(d.record.appraisal.note == VALID["note"] for d in dumps)


# --- whole runs -----------------------------------------------------------------------------------


def test_twenty_tick_llm_run_writes_valid_dumps_and_replays(tmp_path: Path) -> None:
    cfg = config(model="test", ticks=20)
    run_dir = tmp_path / "run"

    result = asyncio.run(run(cfg, run_dir))

    assert len(result.records) == 20
    dumps = load_dumps(run_dir)
    assert [d.tick for d in dumps] == list(range(1, 21))
    for path in sorted(run_dir.glob("tick_*.json")):
        appraisal = TickDump.model_validate_json(path.read_bytes()).record.appraisal
        assert appraisal.note  # the stub leaves the note empty: this came from the model
        assert -1 <= appraisal.valence <= 1 and 0 <= appraisal.arousal <= 1
    # Deliberate and Evaluate stayed on their stubs.
    assert all(p.decision.rationale == "stub" for r in result.records for p in r.passes)

    with appraise_agent.override(model=failing_model()):
        assert asyncio.run(replay(run_dir)) == 20
