"""Everything a model call needs: the model by name, the request settings, the two phase agents
(Deliberate and Appraise) with their output models, and the prompts they are sent.

Agents are built without a model: the model is resolved once per run and passed to every call, so
all calls share one HTTP client. The instructions are fixed text from `phrases.py`; what changes
per tick is the user prompt, the creature's view as JSON. The view is built from what the creature
has (its felt record, its own map, its tries and its recent episodes) with feelings as words. No
true body number, no tile kind, no temperament setting and nothing of the careful reasoner reaches
it; the presumption reaches it only as the `untried` beliefs of the creature's own map.

No import from `surprise_loop.phases`: the phases import this module.
"""

import json
from collections.abc import Sequence
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, Field
from pydantic_ai import Agent, ModelRetry, ModelSettings, RunContext, UserError, models
from pydantic_ai.models import Model

from surprise_loop.config import ConfigError, LLMParams, MonitorParams, Params, Runtime
from surprise_loop.lens import ladder_step
from surprise_loop.phrases import (
    APPRAISE_TEXT,
    AT_HELP,
    CREATURE_TEXT,
    DELIBERATE_TEXT,
    FEEL_WORDS,
    NEAR_WORDS,
    STIR_WORDS,
    SURPRISE_WORDS,
    TONE_WORDS,
    TOO_FAR,
    TOO_MANY_CHANGES,
    WENT_BETTER,
    WENT_SAME,
    WENT_WORSE,
)
from surprise_loop.state import (
    AgentState,
    Appraisal,
    Change,
    Comparison,
    CreatureMap,
    Decision,
    Episode,
    Expectation,
    FeltRecord,
    Sureness,
    Workspace,
)
from surprise_loop.world import Belief, Channel, Direction, Frozen, Unit, key, position_of

DELIBERATE_INSTRUCTIONS = f"{CREATURE_TEXT}\n\n{DELIBERATE_TEXT}"
APPRAISE_INSTRUCTIONS = f"{CREATURE_TEXT}\n\n{APPRAISE_TEXT}"

# --- the model and the request --------------------------------------------------------------------


def load_model(name: str | None) -> models.Model | None:
    """The pydantic-ai model for `name` (e.g. "openrouter:<id>", or "test" for `TestModel`).

    Resolved once per run, so every call of the run shares one model and one HTTP client.
    """
    if name is None:
        return None
    try:
        return models.infer_model(name)
    except UserError as e:
        raise ConfigError(f"model {name!r}: {e}") from None


def model_settings(params: LLMParams) -> ModelSettings:
    """The request settings every model call sends, from `Params.llm`."""
    settings = ModelSettings(timeout=params.timeout_s)
    if params.temperature is not None:
        settings["temperature"] = params.temperature
    return settings


async def ask[D, T](agent: Agent[D, T], prompt: str, model: Model, params: LLMParams, deps: D) -> T:
    """One call of `agent` on `model`: its validated output.

    Output that fails validation is sent back to the model up to `params.output_retries` times;
    after that pydantic-ai raises `UnexpectedModelBehavior`."""
    result = await agent.run(
        prompt,
        model=model,
        deps=deps,
        model_settings=model_settings(params),
        retries={"output": params.output_retries},
    )
    return result.output


async def run_agent[D, T](
    agent: Agent[D, T], prompt: str, ws: Workspace, rt: Runtime, deps: D
) -> T:
    """`ask` on the run's model. A failure stops the run: there is no fallback to a stand-in. The
    error is annotated with the tick."""
    if not isinstance(rt.model, Model):
        raise ConfigError(f"{agent.name}: backend is llm but the runtime has no resolved model")
    try:
        return await ask(agent, prompt, rt.model, rt.config.params.llm, deps)
    except Exception as e:
        e.add_note(f"{agent.name} failed at tick {ws.agent.tick + 1}")
        raise


# --- output models ------------------------------------------------------------------------------


def _one_line(text: str) -> str:
    return " ".join(text.split())


# Free text from a model that later prompts show again (a reading), kept to one line.
OneLine = Annotated[str, AfterValidator(_one_line)]
FeelWord = Literal[*FEEL_WORDS]
NearWord = Literal[*NEAR_WORDS]

# Field names (and the one description) are what the model reads of the output tool; no
# docstrings on the output classes, as pydantic-ai would send them as the tool's description.


class DeliberateOutput(Frozen):
    why: OneLine
    direction: Literal["north", "east", "south", "west"]
    moved: bool
    hunger: FeelWord
    thirst: FeelWord
    ache: FeelWord
    sickness: FeelWord
    unease: FeelWord
    pleasure: FeelWord
    smell: NearWord
    damp: NearWord
    sure: Literal["guessing", "fairly sure", "sure"]


class ChangeOutput(Frozen):
    at: list[int] = Field(min_length=2, max_length=2, description=AT_HELP)
    blocks: Unit
    hurts: Unit
    feeds: Unit
    quenches: Unit
    pleases: Unit


class AppraiseOutput(Frozen):
    reading: OneLine
    changes: list[ChangeOutput]


def to_decision(output: DeliberateOutput) -> Decision:
    expectation = Expectation(
        moved=output.moved,
        levels={c: FEEL_WORDS.index(getattr(output, c.value)) for c in Channel},
        smell=NEAR_WORDS.index(output.smell),
        damp=NEAR_WORDS.index(output.damp),
        sure=Sureness(output.sure),
    )
    return Decision(direction=Direction(output.direction), expectation=expectation, why=output.why)


def to_appraisal(output: AppraiseOutput) -> Appraisal:
    changes = tuple(
        Change(at=(c.at[0], c.at[1]), beliefs={b: getattr(c, b.value) for b in Belief})
        for c in output.changes
    )
    return Appraisal(reading=output.reading, changes=changes)


# --- agents -------------------------------------------------------------------------------------


class AppraiseDeps(Frozen):
    # the most changes one answer may make, and how far from the start a changed spot may be
    max_changes: int
    reach: int


deliberate_agent: Agent[None, DeliberateOutput] = Agent(
    output_type=DeliberateOutput, instructions=DELIBERATE_INSTRUCTIONS, name="deliberate"
)

appraise_agent: Agent[AppraiseDeps, AppraiseOutput] = Agent(
    output_type=AppraiseOutput,
    deps_type=AppraiseDeps,
    instructions=APPRAISE_INSTRUCTIONS,
    name="appraise",
)


@appraise_agent.output_validator
def _within_bounds(ctx: RunContext[AppraiseDeps], output: AppraiseOutput) -> AppraiseOutput:
    """Too many changes, or a change beyond reach of the start, is sent back to the model."""
    if len(output.changes) > ctx.deps.max_changes:
        raise ModelRetry(TOO_MANY_CHANGES.format(max_changes=ctx.deps.max_changes))
    for change in output.changes:
        if any(abs(v) > ctx.deps.reach for v in change.at):
            raise ModelRetry(TOO_FAR.format(at=change.at, reach=ctx.deps.reach))
    return output


# --- the view and the prompts -------------------------------------------------------------------


def _felt_words(felt: FeltRecord) -> dict[str, Any]:
    return {
        "moved": felt.moved,
        **{c.value: FEEL_WORDS[felt.levels[c]] for c in Channel},
        "smell": NEAR_WORDS[felt.smell],
        "damp": NEAR_WORDS[felt.damp],
        "tone": TONE_WORDS[felt.tone],
        "stir": STIR_WORDS[felt.stir],
    }


def _expected_words(e: Expectation) -> dict[str, Any]:
    return {
        "moved": e.moved,
        **{c.value: FEEL_WORDS[e.levels[c]] for c in Channel},
        "smell": NEAR_WORDS[e.smell],
        "damp": NEAR_WORDS[e.damp],
        "sure": e.sure.value,
    }


def _beliefs(beliefs: dict[Belief, float], decimals: int) -> dict[str, float]:
    return {b.value: round(beliefs[b], decimals) for b in Belief}


def _how_it_went(comparison: Comparison, p: MonitorParams) -> dict[str, str]:
    if comparison.worse >= p.worse_edge:
        went = WENT_WORSE
    elif comparison.worse <= -p.worse_edge:
        went = WENT_BETTER
    else:
        went = WENT_SAME
    return {
        "surprise": SURPRISE_WORDS[ladder_step(comparison.surprise, p.surprise_edges)],
        "it_went": went,
    }


def _map_view(own_map: CreatureMap, tries: dict[str, int], decimals: int) -> dict[str, Any]:
    keys = {k for k, n in tries.items() if n >= 1} | set(own_map.tiles)
    spots = sorted((position_of(k) for k in keys), key=lambda at: (-at[1], at[0]))
    return {
        "untried": _beliefs(own_map.untried, decimals),
        "spots": [
            {
                "at": list(at),
                "tries": tries.get(key(at), 0),
                **_beliefs(own_map.at(at), decimals),
            }
            for at in spots
        ],
    }


def _step_view(e: Episode, p: Params) -> dict[str, Any]:
    return {
        "went": e.direction.value,
        "expected": _expected_words(e.expectation),
        "felt": _felt_words(e.felt),
        **_how_it_went(e.comparison, p.monitor),
        "reading": e.reading,
    }


def creature_view(agent: AgentState, recent: Sequence[Episode], p: Params) -> dict[str, Any]:
    """What the creature is shown of itself: how it feels, where it is, its own map, and its last
    `memory.recent_k` steps, oldest first."""
    k = p.memory.recent_k
    last = list(recent)[-k:] if k else []
    return {
        "you_feel": _felt_words(agent.felt),
        "you_are_at": list(agent.felt.at),
        "your_map": _map_view(agent.own_map, agent.tries, p.prompt.decimals),
        "your_last_steps": [_step_view(e, p) for e in last],
    }


def deliberate_prompt(view: dict[str, Any]) -> str:
    return json.dumps(view, ensure_ascii=False)


def appraise_prompt(
    view: dict[str, Any],
    decision: Decision,
    felt_after: FeltRecord,
    comparison: Comparison,
    p: Params,
) -> str:
    """The view of the creature before the step, with how it feels and where it is after it, and
    the step just taken: where it went, what it expected, what it felt, and how that compared."""
    felt = _felt_words(felt_after)
    shown = {
        **view,
        "you_feel": felt,
        "you_are_at": list(felt_after.at),
        "you_went": decision.direction.value,
        "you_expected": _expected_words(decision.expectation),
        "you_felt": felt,
        **_how_it_went(comparison, p.monitor),
    }
    return json.dumps(shown, ensure_ascii=False)
