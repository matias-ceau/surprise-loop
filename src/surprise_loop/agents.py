"""Everything a model call needs: the model by name, the request settings, the two phase agents
(Appraise and Deliberate) with their output models, and their prompt builders.

Agents are built without a model: the model is resolved once per run (or per ladder check) and
passed to every call, so all calls share one HTTP client. Instructions are fixed text; what
changes per tick goes into the user prompt. Every sentence of either comes from `phrases.py` or
is the model's own earlier output (a reading); no number, no identifier and no clock reaches a
model.

No import from `surprise_loop.phases`: the phases import this module.
"""

from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
)
from pydantic_ai import Agent, ModelSettings, UserError, models
from pydantic_ai.models import Model

from surprise_loop.config import ConfigError, LLMParams, Runtime
from surprise_loop.phrases import (
    ACTION_QUESTION,
    APPRAISE_ABOUT_HELP,
    APPRAISE_INSTRUCTION,
    APPRAISE_QUESTION,
    APPRAISE_READING_HELP,
    COLLAPSED_LINE,
    DELIBERATE_INSTRUCTION,
    DID_LINES,
    EXPECT_HELP,
    EXPECTED_LINE,
    FEEL_QUESTION,
    HEAT_HELP,
    LADDER_LETTER_HELP,
    LADDER_WHY_HELP,
    LADDER_WORDS,
    OFFERS_HEADING,
    OPEN_QUESTION_LEAD,
    QUESTION_PHRASES,
    READING_HEADING,
    RECALLED_HEADING,
    STUMBLED_LINE,
    SURE_HELP,
    SURE_QUESTION,
    SURPRISE_PHRASES,
    THIRST_HELP,
    TIREDNESS_HELP,
    WHY_QUESTION,
    WORLD_SURPRISE_PHRASES,
)
from surprise_loop.state import ChangeWord, Episode, Offer, Sureness, SurpriseNote, Workspace
from surprise_loop.world import QUESTION_OF, Channel, Event

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
    """`ask` on the run's model. A failure stops the run: there is no fallback to the stub. The
    error is annotated with the tick and the pass."""
    if not isinstance(rt.model, Model):
        raise ConfigError(f"{agent.name}: backend is llm but the runtime has no resolved model")
    try:
        return await ask(agent, prompt, rt.model, rt.config.params.llm, deps)
    except Exception as e:
        e.add_note(f"{agent.name} failed at tick {ws.agent.tick + 1}, pass {ws.pass_index}")
        raise


# --- output models ------------------------------------------------------------------------------


def _one_line(text: str) -> str:
    """Collapse whitespace so model text fits on one prompt line; refuse text with no words."""
    text = " ".join(text.split())
    if not text:
        raise ValueError("must not be empty")
    return text


# Free text from a model that later prompts may show again (a reading, a why).
OneLine = Annotated[str, Field(min_length=1), AfterValidator(_one_line)]


# Field names and descriptions are what the model reads of the output tool; no docstrings on the
# output classes, as pydantic-ai would send them as the tool's description.
class _Output(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


ABOUT: dict[str, Channel | None] = {
    "heat": Channel.HEAT,
    "thirst": Channel.WATER,
    "tiredness": Channel.ENERGY,
    "nothing": None,
}


class AppraiseOutput(_Output):
    reading: OneLine = Field(description=APPRAISE_READING_HELP)
    about: Literal["heat", "thirst", "tiredness", "nothing"] = Field(
        description=APPRAISE_ABOUT_HELP
    )


LETTERS = "ABCDE"
HeatWord = Literal[*LADDER_WORDS[Channel.HEAT].values()]
ThirstWord = Literal[*LADDER_WORDS[Channel.WATER].values()]
TirednessWord = Literal[*LADDER_WORDS[Channel.ENERGY].values()]

# the answer field of each channel
CHANNEL_FIELD: dict[Channel, str] = {
    Channel.HEAT: "heat",
    Channel.WATER: "thirst",
    Channel.ENERGY: "tiredness",
}
EXPECT: dict[str, bool | None] = {"yes": True, "no": False, "no expectation": None}
SURENESS: dict[str, Sureness] = {
    "guessing": Sureness.GUESSING,
    "fairly sure": Sureness.FAIRLY_SURE,
    "sure": Sureness.SURE,
}


class DeliberateOutput(_Output):
    letter: Literal["A", "B", "C", "D", "E"] = Field(description=LADDER_LETTER_HELP)
    heat: HeatWord = Field(description=HEAT_HELP)
    thirst: ThirstWord = Field(description=THIRST_HELP)
    tiredness: TirednessWord = Field(description=TIREDNESS_HELP)
    expect: Literal["yes", "no", "no expectation"] = Field(description=EXPECT_HELP)
    sure: Literal["guessing", "fairly sure", "sure"] = Field(description=SURE_HELP)
    why: OneLine = Field(description=LADDER_WHY_HELP)

    @field_validator("letter")
    @classmethod
    def _on_offer(cls, letter: str, info: ValidationInfo) -> str:
        """A letter beyond the offers is an error the model is told of and asked to fix. The
        number of offers comes in the validation context; without it, any of the five letters."""
        offers = (info.context or {}).get("offers")
        if offers is not None and LETTERS.index(letter) >= offers:
            raise ValueError(
                f"{letter} is not offered: choose one of {', '.join(LETTERS[:offers])}"
            )
        return letter

    def words(self) -> dict[Channel, ChangeWord]:
        """The ladder phrases as change words, per channel."""
        out: dict[Channel, ChangeWord] = {}
        for c, field in CHANNEL_FIELD.items():
            phrase = getattr(self, field)
            out[c] = next(w for w, p in LADDER_WORDS[c].items() if p == phrase)
        return out


# --- agents -------------------------------------------------------------------------------------

appraise_agent: Agent[None, AppraiseOutput] = Agent(
    output_type=AppraiseOutput, instructions=APPRAISE_INSTRUCTION, name="appraise"
)

# deps: how many actions are on offer, the bound on the letter (a validation error, so a letter
# off the list is sent back to the model like any other invalid output)
deliberate_agent: Agent[int, DeliberateOutput] = Agent(
    output_type=DeliberateOutput,
    deps_type=int,
    instructions=DELIBERATE_INSTRUCTION,
    name="deliberate",
    validation_context=lambda ctx: {"offers": ctx.deps},
)


# --- prompts ------------------------------------------------------------------------------------


def appraise_prompt(scene_text: str) -> str:
    """The felt scene, then one question."""
    return f"{scene_text}\n\n{APPRAISE_QUESTION}"


def _note_sentence(note: SurpriseNote | None) -> str | None:
    if note is None:
        return None
    if note.channel is not None and note.worse is not None:
        return SURPRISE_PHRASES[(note.channel, note.worse)]
    if note.question is not None and note.actual is not None:
        return WORLD_SURPRISE_PHRASES[(note.question, note.actual)]
    return None


def recalled_line(episode: Episode, note: SurpriseNote | None) -> str:
    """A recalled moment in words: what was done (or what the body did instead), what was
    expected in ladder phrases when a model said it, the surprise `note` voiced of it, and the
    regulation line it was later given."""
    if episode.event is Event.COLLAPSED:
        parts = [COLLAPSED_LINE]
    elif episode.event is Event.STUMBLED:
        parts = [STUMBLED_LINE]
    else:
        parts = [DID_LINES[(episode.place, episode.action)]]
    words = episode.expectation.words
    if words is not None:
        said = {CHANNEL_FIELD[c]: LADDER_WORDS[c][words[c]] for c in Channel}
        parts.append(EXPECTED_LINE.format(**said))
    if (sentence := _note_sentence(note)) is not None:
        parts.append(sentence)
    if episode.regulation:
        parts.append(episode.regulation)
    return " ".join(parts)


def deliberate_prompt(
    scene_text: str,
    reading: str,
    offers: list[Offer],
    recalled: list[str],
    pull_line: str,
    override_line: str,
) -> str:
    """The felt scene; the reading; the offers lettered in the order given; the recalled moments
    (`recalled_line`), oldest first; the pull line; the override line; then the questions, with
    the open question of each offer that has one. Empty parts are left out."""
    if not 1 <= len(offers) <= len(LETTERS):
        raise ValueError(f"deliberate_prompt: {len(offers)} offers, not 1 to {len(LETTERS)}")
    blocks = [
        [scene_text],
        [f"{READING_HEADING} {reading}"],
        [OFFERS_HEADING, *(f"{ltr}. {o.phrase}" for ltr, o in zip(LETTERS, offers, strict=False))],
    ]
    if recalled:
        blocks.append([RECALLED_HEADING, *(f"- {line}" for line in recalled)])
    blocks += [[line] for line in (pull_line, override_line) if line]
    questions = [ACTION_QUESTION, FEEL_QUESTION]
    for letter, offer in zip(LETTERS, offers, strict=False):
        if (question := QUESTION_OF[offer.action]) is not None:
            lead = OPEN_QUESTION_LEAD.format(letter=letter)
            questions.append(f"{lead} {QUESTION_PHRASES[question]}")
    blocks.append([*questions, SURE_QUESTION, WHY_QUESTION])
    return "\n\n".join("\n".join(block) for block in blocks)
