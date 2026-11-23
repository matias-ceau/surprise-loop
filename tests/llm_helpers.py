"""Helpers shared by the tests of the model-backed phases (TestModel / FunctionModel only)."""

import json
import re
from collections.abc import Callable
from typing import Any

from creatures import run_config
from pydantic_ai import ModelResponse, ToolCallPart
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from surprise_loop.agents import APPRAISE_INSTRUCTIONS, DELIBERATE_INSTRUCTIONS
from surprise_loop.config import Backends, RunConfig, Runtime
from surprise_loop.world import GridWorld

# What no text a model reads may hold: a word that frames the place as anything but a place.
FORBIDDEN = re.compile(r"\b(game|simulation|experiment)", re.IGNORECASE)

BOTH_LLM = Backends(deliberate="llm", appraise="llm")

DECISION: dict[str, Any] = {
    "why": "The east looks open.",
    "direction": "east",
    "moved": True,
    "hunger": "slight",
    "thirst": "slight",
    "ache": "none",
    "sickness": "none",
    "unease": "none",
    "pleasure": "none",
    "smell": "none",
    "damp": "faint",
    "sure": "guessing",
}
APPRAISAL: dict[str, Any] = {"reading": "Nothing new here.", "changes": []}


def beliefs(blocks: float, hurts: float, feeds: float, quenches: float, pleases: float) -> dict:
    return {
        "blocks": blocks,
        "hurts": hurts,
        "feeds": feeds,
        "quenches": quenches,
        "pleases": pleases,
    }


def llm_config(
    agent: str = "plain",
    *,
    backends: Backends = BOTH_LLM,
    model: str | None = "test",
    ticks: int | None = 20,
    **changes: Any,
) -> RunConfig:
    """A run of the two-maps layout with model-backed phases, capped at `ticks`."""
    return run_config(agent, 1, backends=backends, model=model, ticks=ticks, **changes)


def runtime(config: RunConfig, model: object) -> Runtime:
    return Runtime(config=config, world=GridWorld(config.world, config.seed), model=model)


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


Answer = Callable[[str], dict[str, Any]]


class Recorder:
    """A FunctionModel for both agents: Deliberate answers `decide(prompt)`, Appraise answers
    `appraise(prompt)`. Every call is kept as (agent, instructions, user prompt), with the output
    tools the model was shown. `fail_on` makes the n-th Deliberate call (from 1) raise."""

    def __init__(
        self,
        decide: Answer | None = None,
        appraise: Answer | None = None,
        fail_on: int | None = None,
    ) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.tools: list[Any] = []
        self.decide = decide or (lambda prompt: DECISION)
        self.appraise = appraise or (lambda prompt: APPRAISAL)
        self.fail_on = fail_on
        self.model = FunctionModel(self._answer)

    def _answer(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = user_prompt(messages)
        instructions = info.instructions or ""
        if instructions == DELIBERATE_INSTRUCTIONS:
            name, answer = "deliberate", self.decide
            if self.fail_on is not None and len(self.prompts("deliberate")) + 1 == self.fail_on:
                raise RuntimeError("the model is gone")
        elif instructions == APPRAISE_INSTRUCTIONS:
            name, answer = "appraise", self.appraise
        else:
            raise AssertionError(f"unknown agent with instructions {instructions!r}")
        self.calls.append((name, instructions, prompt))
        self.tools.extend(info.output_tools)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, answer(prompt))])

    def prompts(self, name: str) -> list[str]:
        return [p for n, _, p in self.calls if n == name]


def scripted(*answers: dict[str, Any]) -> tuple[FunctionModel, list[list[ModelMessage]]]:
    """A model that gives `answers` in turn, repeating the last; and the messages of each call."""
    calls: list[list[ModelMessage]] = []

    def function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(list(messages))
        args = answers[min(len(calls), len(answers)) - 1]
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])

    return FunctionModel(function), calls


def failing_model() -> FunctionModel:
    """A model that fails on any call: proves that a replay makes none."""

    def boom(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise AssertionError("replay must not call the model")

    return FunctionModel(boom)
