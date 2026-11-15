"""Helpers shared by the tests of the LLM-backed phases (TestModel / FunctionModel only)."""

import json
from typing import Any

from pydantic_ai import ModelResponse, ToolCallPart
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from surprise_loop.config import (
    AgentSpec,
    Backends,
    LLMParams,
    Params,
    RunConfig,
    Runtime,
    initial_agent_state,
)
from surprise_loop.state import (
    Appraisal,
    Conflict,
    Episode,
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
    PhaseKind,
    PhaseSpec,
    SelfAdminWorld,
    WorldConfig,
)

HIDDEN = ["phase_kind", "ratio_required", "active_press", "ratio_completed", "active_lever"]
HIDDEN += [kind.value for kind in PhaseKind] + ["step"]

TRAITS = Traits(reward_sensitivity=1.37, surprise_learning_rate=0.41, mood_inertia=0.63)
PRIORITIES = {GoalName.SEEK_REWARD: 1.23, GoalName.EXPLORE: 0.47, GoalName.CONSERVE: 0.58}


def make_config(
    *,
    backends: Backends | None = None,
    model: str | None = "test",
    llm: LLMParams | None = None,
    ticks: int | None = 20,
    run_id: str = "llm-deliberate",
) -> RunConfig:
    schedule = [
        PhaseSpec(kind=PhaseKind.FIXED_RATIO, steps=10, ratio=1),
        PhaseSpec(kind=PhaseKind.PROGRESSIVE_RATIO, steps=10, pr_start=1, pr_increment=1),
        PhaseSpec(kind=PhaseKind.EXTINCTION, steps=20),
    ]
    return RunConfig(
        run_id=run_id,
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
    recalled = [
        episode(5, Action.PRESS_A, Event.REWARD, 1.0, "good, that served seek_reward"),
        episode(6, Action.PRESS_B, Event.NO_REWARD, 0.0, "wasted effort on lever B"),
    ]
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
            "memory": Memory(episodes=recalled),
            "tick": 7,
        }
    )
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


def retry_feedback(messages: list[ModelMessage]) -> str:
    last = messages[-1]
    assert isinstance(last, ModelRequest)
    returns = [p for p in last.parts if isinstance(p, ToolReturnPart) and p.outcome == "retried"]
    assert returns, "expected a retry prompt for the output tool"
    return json.dumps([str(p.content) for p in returns])


def assert_no_hidden(text: str) -> None:
    """`text` names none of the world's hidden fields, phase kinds or its step counter."""
    for hidden in HIDDEN:
        assert hidden not in text.lower(), f"{hidden!r} leaked into: {text[:200]}"


def failing_model() -> FunctionModel:
    """A model that fails on any call: proves that replay makes none."""

    def boom(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise AssertionError("replay must not call the model")

    return FunctionModel(boom)
