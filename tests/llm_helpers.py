"""Helpers shared by the tests of the model-backed phases (TestModel / FunctionModel only)."""

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic_ai import ModelResponse, ToolCallPart
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from surprise_loop.config import Backends, LLMParams, RunConfig, Runtime, load_config
from surprise_loop.phrases import APPRAISE_INSTRUCTION, DELIBERATE_INSTRUCTION, OFFER_PHRASES
from surprise_loop.world import Action, ClearingWorld

ROOT = Path(__file__).resolve().parents[1]
AFTERNOON_LLM = ROOT / "experiments" / "clearing_afternoon_llm.yaml"

# What no text a model reads may hold: a digit, an identifier-like token, or a word that frames
# the clearing as anything but a place.
MACHINE_LIKE = re.compile(r"[0-9_]")
FORBIDDEN = re.compile(r"\b(game|simulation|experiment)", re.IGNORECASE)

APPRAISAL = {"reading": "The sun is too much for me.", "about": "heat"}
DECISION = {
    "letter": "A",
    "heat": "much cooler",
    "thirst": "about the same",
    "tiredness": "a bit more tired",
    "expect": "yes",
    "sure": "fairly sure",
    "why": "The shade is close.",
}


def afternoon(
    agent: int = 0,
    *,
    backends: Backends | None = None,
    model: str | None = "test",
    ticks: int | None = 20,
    llm: LLMParams | None = None,
) -> RunConfig:
    """A run of the model afternoon (0 is ilo, 1 is tavi)."""
    config = load_config(AFTERNOON_LLM).runs()[agent]
    update: dict[str, Any] = {"model": model, "ticks": ticks}
    if backends is not None:
        update["backends"] = backends
    if llm is not None:
        update["params"] = config.params.model_copy(update={"llm": llm})
    return config.model_copy(update=update)


def runtime(config: RunConfig, model: object) -> Runtime:
    world = ClearingWorld(config.world, config.seed, config.agent.body, config.start_place)
    return Runtime(config=config, world=world, model=model)


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


def offered_letters(prompt: str) -> dict[str, Action]:
    """The lettered offers of a Deliberate prompt, letter to action, read back from its phrases."""
    by_phrase: dict[str, set[Action]] = {}
    for (_, action), phrase in OFFER_PHRASES.items():
        by_phrase.setdefault(phrase, set()).add(action)
    out: dict[str, Action] = {}
    for line in prompt.split("\n"):
        letter, sep, text = line.partition(". ")
        if sep and len(letter) == 1 and letter in "ABCDE" and text in by_phrase:
            [action] = by_phrase[text]
            out[letter] = action
    return out


Answer = Callable[[str], dict[str, Any]]


class Recorder:
    """A FunctionModel for both agents: Appraise answers `appraisal`, Deliberate answers what
    `decide(prompt)` returns. Every call is kept as (agent, instructions, user prompt)."""

    def __init__(
        self, decide: Answer | None = None, appraisal: dict[str, Any] | None = None
    ) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.decide = decide or (lambda prompt: DECISION)
        self.appraisal = appraisal or APPRAISAL
        self.model = FunctionModel(self._answer)

    def _answer(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = user_prompt(messages)
        instructions = info.instructions or ""
        if instructions == APPRAISE_INSTRUCTION:
            name, args = "appraise", self.appraisal
        elif instructions == DELIBERATE_INSTRUCTION:
            name, args = "deliberate", self.decide(prompt)
        else:
            raise AssertionError(f"unknown agent with instructions {instructions!r}")
        self.calls.append((name, instructions, prompt))
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])

    def prompts(self, name: str) -> list[str]:
        return [p for n, _, p in self.calls if n == name]


def prefer(*actions: Action) -> Answer:
    """A Deliberate answer choosing the first of `actions` on offer (else the first letter),
    expecting to feel much cooler: whatever it does, the heat comes out worse than expected."""

    def decide(prompt: str) -> dict[str, Any]:
        offered = offered_letters(prompt)
        letter = next((ltr for a in actions for ltr, b in offered.items() if a is b), min(offered))
        return {**DECISION, "letter": letter}

    return decide


def failing_model() -> FunctionModel:
    """A model that fails on any call: proves that a replay makes none."""

    def boom(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise AssertionError("replay must not call the model")

    return FunctionModel(boom)
