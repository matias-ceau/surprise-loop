"""pydantic-ai agents for the LLM-backed phases, their output models and prompt builders.

Each agent is built without a model: the run's model is resolved once (loop.py) and passed to
every `agent.run(..., model=rt.model)`. Instructions are static; everything that changes per tick
goes into the user prompt built from the `Workspace`. A prompt shows only what the agent may know:
the percept (event, reward, last action, recalled episodes) and its own mental state. It never
shows the world's `StepInfo` (phase, ratio, which lever is active) or its step counter.
"""

from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.models import Model

# No import from surprise_loop.phases: the phases import this module.
from surprise_loop.config import ConfigError, LLMParams, Runtime
from surprise_loop.state import AgentState, Episode, Percept, Workspace
from surprise_loop.world import Action, Event


class _Output(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def model_settings(params: LLMParams) -> ModelSettings:
    """The request settings every phase agent sends, from `Params.llm`."""
    settings = ModelSettings(timeout=params.timeout_s)
    if params.temperature is not None:
        settings["temperature"] = params.temperature
    return settings


async def run_agent[T](agent: Agent[None, T], prompt: str, ws: Workspace, rt: Runtime) -> T:
    """Run a phase agent on the run's model and return its validated output.

    Output that fails validation is sent back to the model up to `Params.llm.output_retries`
    times; after that pydantic-ai raises `UnexpectedModelBehavior` and the run stops. There is
    no fallback to the stub (plan call 12). The error is annotated with the tick and pass.
    """
    if not isinstance(rt.model, Model):
        raise ConfigError(f"{agent.name}: backend is llm but the runtime has no resolved model")
    params = rt.config.params.llm
    try:
        result = await agent.run(
            prompt,
            model=rt.model,
            model_settings=model_settings(params),
            retries={"output": params.output_retries},
        )
    except Exception as e:
        e.add_note(f"{agent.name} failed at tick {ws.agent.tick + 1}, pass {ws.pass_index}")
        raise
    return result.output


def _one_line(text: str) -> str:
    """Collapse whitespace so model text fits on one prompt line; refuse text with no words."""
    text = " ".join(text.split())
    if not text:
        raise ValueError("must not be empty")
    return text


# Free text from a model that later prompts show again (appraisal note, critique).
OneLine = Annotated[str, Field(min_length=1), AfterValidator(_one_line)]


# The events a step can produce; START only ever precedes the first step, so it is not offered.
OutcomeEvent = Literal[Event.REWARD, Event.NO_REWARD, Event.EXPLORED, Event.RESTED]


# --- Deliberate ---------------------------------------------------------------------------------


class DeliberateOutput(_Output):
    reasoning: str = Field(
        description="One or two sentences: why this action, and why you expect this outcome."
    )
    action: Action = Field(description="The action to take now.")
    expected_event: OutcomeEvent = Field(
        description="The event you expect this action to produce: reward or no_reward for a "
        "lever press, explored for explore, rested for rest."
    )
    expected_reward: float = Field(
        ge=0,
        description="The reward you expect this action to deliver (0 if you expect none).",
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="How sure you are of this prediction, from 0 (pure guess) to 1 (certain).",
    )

    @model_validator(mode="after")
    def _prediction_fits_action(self) -> "DeliberateOutput":
        """A press predicts reward or no_reward; explore and rest predict their event and 0."""
        if self.action in _PRESS_EVENTS:
            if self.expected_event not in _PRESS_EVENTS[self.action]:
                raise ValueError(
                    f"{self.action.value} can only produce reward or no_reward, "
                    f"not {self.expected_event.value}"
                )
            return self
        event = _NO_REWARD_EVENT[self.action]
        if self.expected_event != event:
            raise ValueError(f"{self.action.value} always produces {event.value}")
        if self.expected_reward != 0:
            raise ValueError(f"{self.action.value} never delivers a reward: expected_reward is 0")
        return self


_PRESS_EVENTS = {
    Action.PRESS_A: (Event.REWARD, Event.NO_REWARD),
    Action.PRESS_B: (Event.REWARD, Event.NO_REWARD),
}
_NO_REWARD_EVENT = {Action.EXPLORE: Event.EXPLORED, Action.REST: Event.RESTED}


DELIBERATE_INSTRUCTIONS = """\
You are the decision-making part of an agent living in a small world. Each turn you choose one \
action and predict what it will bring.

Actions:
- press_a: press lever A. It may or may not deliver a reward (event reward or no_reward).
- press_b: press lever B. It may or may not deliver a reward (event reward or no_reward).
- explore: look around (event explored, no reward).
- rest: do nothing for a moment (event rested, no reward).

You do not know the rules of the world; learn them from your recalled episodes. Weigh the \
rewards you hope for against your goals, your mood and your habits.

A reward always comes in the same fixed amount, which you can read from the episodes in which \
you were rewarded. Explore and rest never deliver one, so predict 0 for them. Until you have \
seen a reward, keep expected_reward small.

After acting, your prediction is compared with what actually happens; the difference (your \
surprise) shapes your mood, memory, goals and habits. Predict honestly: expected_reward is the \
reward you actually expect, and confidence says how sure you are.

Answer with the structured output only: reasoning, action, expected_event, expected_reward and \
confidence."""

deliberate_agent: Agent[None, DeliberateOutput] = Agent(
    output_type=DeliberateOutput,
    instructions=DELIBERATE_INSTRUCTIONS,
    name="deliberate",
)


def deliberate_prompt(ws: Workspace) -> str:
    """The per-tick prompt: what the agent perceives and its own mental state, nothing else."""
    percept, appraisal = ws.percept, ws.appraisal
    if percept is None or appraisal is None:
        missing = "percept" if percept is None else "appraisal"
        raise ValueError(f"deliberate_prompt needs ws.{missing}, which is missing")
    agent = ws.agent

    lines = [
        "## What just happened",
        _last_line(percept),
        "",
        *_mood_lines(agent),
        "",
        "## Your appraisal of this moment",
        f"Valence: {_signed(appraisal.valence)}. Arousal: {_num(appraisal.arousal)}.",
    ]
    if appraisal.note:
        lines.append(f"Note: {appraisal.note}")
    lines += [
        "",
        *_goal_lines(agent, "## Your goal priorities (higher matters more)"),
        "",
        "## Your habits (learned bias towards each action; positive pulls you towards it)",
        *(f"- {action.value}: {_signed(b)}" for action, b in agent.dispositions.biases.items()),
        "",
        *_trait_lines(agent),
        "",
        *_recalled_lines(percept),
    ]
    if ws.conflict is not None:
        lines += [
            "",
            "## Conflict",
            f"Your habit of {ws.conflict.action.value} conflicts with your goals: the goal it "
            "serves has just lost priority, yet the habit is still strong. Decide again on the "
            "evidence and your goals, not out of habit.",
            f"Detail: {ws.conflict.detail}",
        ]
    lines += ["", "Choose your next action and predict its outcome."]
    return "\n".join(lines)


# --- Appraise -----------------------------------------------------------------------------------


class AppraiseOutput(_Output):
    note: OneLine = Field(
        description="One short sentence, in your own words, describing how this moment feels."
    )
    valence: float = Field(
        ge=-1,
        le=1,
        description="How pleasant this moment feels: -1 very unpleasant, 0 neutral, "
        "+1 very pleasant.",
    )
    arousal: float = Field(
        ge=0,
        le=1,
        description="How activated you feel: 0 completely calm, 1 highly agitated or excited.",
    )


APPRAISE_INSTRUCTIONS = """\
You are the feeling part of an agent living in a small world. Each turn you read what just \
happened and say how that moment feels to the agent, given its mood, goals and temperament. \
You do not choose actions.

Answer with three things:
- note: one short sentence, in the agent's own words, describing the feeling.
- valence: a number from -1 to +1. -1 means very unpleasant, 0 neutral, +1 very pleasant.
- arousal: a number from 0 to 1. 0 means completely calm, 1 highly agitated or excited.

How to judge:
- Start from the current mood. One moment shifts the feeling but rarely overturns it, and the \
higher mood_inertia is, the less it shifts.
- A reward feels pleasant, more so the higher reward_sensitivity and the priority of \
seek_reward. Missing a reward you expected feels unpleasant.
- Surprise raises arousal: an outcome far from what you predicted is activating, an outcome you \
expected is calming. Large positive or negative surprise both raise arousal.

Answer with the structured output only: note, valence and arousal."""

appraise_agent: Agent[None, AppraiseOutput] = Agent(
    output_type=AppraiseOutput,
    instructions=APPRAISE_INSTRUCTIONS,
    name="appraise",
)


def appraise_prompt(ws: Workspace) -> str:
    """The per-tick prompt: the percept, the last prediction and the agent's mood, goals and
    temperament. No world information beyond the percept."""
    percept = ws.percept
    if percept is None:
        raise ValueError("appraise_prompt needs ws.percept, which is missing")
    agent = ws.agent
    episodes = agent.memory.episodes
    lines = ["## What just happened", _last_line(percept)]
    if episodes:
        # The newest episode in memory is the action whose outcome is being perceived now.
        last = episodes[-1]
        p = last.prediction
        lines.append(
            f"You had predicted: {p.expected_event.value}, reward {_num(p.expected_reward)} "
            f"(confidence {_num(p.confidence)}). Surprise: {_signed(last.prediction_error.signed)}."
        )
    else:
        lines.append("Nothing has happened yet: this is your first moment in this world.")
    lines += [
        "",
        *_mood_lines(agent),
        "",
        *_goal_lines(agent, "## Your goal priorities (higher matters more)"),
        "",
        *_trait_lines(agent),
        "",
        *_recalled_lines(percept),
        "",
        "How does this moment feel?",
    ]
    return "\n".join(lines)


# --- prompt sections shared by the phase prompts -------------------------------------------------


def _last_line(percept: Percept) -> str:
    last_action = percept.last_action.value if percept.last_action else "none yet"
    return (
        f"Last action: {last_action}. Event: {percept.event.value}. Reward: {_num(percept.reward)}."
    )


def _mood_lines(agent: AgentState) -> list[str]:
    return [
        "## Your mood",
        f"Valence: {_signed(agent.mood.valence)} (-1 unpleasant .. +1 pleasant). "
        f"Arousal: {_num(agent.mood.arousal)} (0 calm .. 1 agitated).",
    ]


def _goal_lines(agent: AgentState, heading: str) -> list[str]:
    return [
        heading,
        *(f"- {goal.value}: {_num(p)}" for goal, p in agent.goals.priorities.items()),
        "  (seek_reward is served by pressing a lever, explore by explore, conserve by rest)",
    ]


def _trait_lines(agent: AgentState) -> list[str]:
    return [
        "## Your temperament",
        f"- reward_sensitivity: {_num(agent.traits.reward_sensitivity)} "
        "(how much rewards matter to you)",
        f"- surprise_learning_rate: {_num(agent.traits.surprise_learning_rate)} "
        "(how quickly surprise changes your expectations)",
        f"- mood_inertia: {_num(agent.traits.mood_inertia)} (how slowly your mood changes)",
    ]


def _recalled_lines(percept: Percept) -> list[str]:
    lines = [
        "## Recalled episodes (oldest first)",
        "(surprise: actual minus expected reward, weighted by your confidence; salience: how "
        "much the episode stood out, 0..1; alignment: the critic's score of how well the "
        "action served your goals, -1..1)",
    ]
    if percept.recalled:
        lines += [_episode_line(i, e) for i, e in enumerate(percept.recalled, start=1)]
    else:
        lines.append("None yet: you have no experience of this world.")
    return lines


def _episode_line(index: int, e: Episode) -> str:
    p = e.prediction
    line = (
        f"{index}. {e.action.value}: predicted {p.expected_event.value}, reward "
        f"{_num(p.expected_reward)} (confidence {_num(p.confidence)}); got {e.event.value}, "
        f"reward {_num(e.reward)}; surprise {_signed(e.prediction_error.signed)}; "
        f"salience {_num(e.salience)}; alignment {_signed(e.alignment)}"
    )
    if e.critique:
        line += f"; critique: {e.critique}"
    return line


def _num(x: float) -> str:
    return f"{x:.2f}"


def _signed(x: float) -> str:
    return f"{x:+.2f}"
