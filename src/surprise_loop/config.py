"""Experiment configuration, parameters and run setup.

Every number that shapes behaviour lives in `Params` (or in `Traits`). The defaults below are
tuned so the stub agent passes the behavioural gate in tests/test_behaviour.py on the
experiments/trait_grid.yaml schedule. Each can be overridden from YAML.
"""

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from surprise_loop.state import (
    AgentState,
    Dispositions,
    GoalName,
    Goals,
    Memory,
    Mood,
    Traits,
    _require_keys,
)
from surprise_loop.world import Action, World, WorldConfig

MODEL_ENV_VAR = "SURPRISE_LOOP_MODEL"

Unit = Annotated[float, Field(ge=0, le=1)]
NonNegative = Annotated[float, Field(ge=0)]
Positive = Annotated[float, Field(gt=0)]
SafeName = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.-]+$")]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


class MonitorParams(_Frozen):
    # minimum weight of a prediction error when the prediction had zero confidence
    confidence_floor: Unit = 0.3
    # extra unsigned error added when the predicted event was wrong
    event_mismatch_weight: NonNegative = 0.5


class MoodParams(_Frozen):
    # how strongly unsigned prediction error drives arousal
    arousal_gain: NonNegative = 1.0


class MemoryParams(_Frozen):
    # how many recent episodes Perceive recalls; above quit_steps, so a pause of quit_steps steps
    # does not empty the window and reset a press's expected reward to the prior (was 10)
    retrieval_k: int = Field(default=20, ge=1)
    # salience of an episode with zero prediction error
    base_salience: Unit = 0.1


class GoalParams(_Frozen):
    # EMA rate of per-goal signed error (scaled by surprise_learning_rate)
    ema_rate: Unit = 0.2
    # |EMA| above which a pass counts towards an error streak
    error_threshold: NonNegative = 0.05
    # consecutive same-sign over-threshold passes before a goal priority is revised
    patience: int = Field(default=3, ge=1)
    # size of one goal priority revision
    revision_step: NonNegative = 0.1
    # lowest a goal priority can be revised to
    min_priority: NonNegative = 0.1
    # highest a goal priority can be revised to
    max_priority: NonNegative = 2.0

    @model_validator(mode="after")
    def _ordered(self) -> "GoalParams":
        if self.min_priority > self.max_priority:
            raise ValueError("min_priority must be <= max_priority")
        return self


class DispositionParams(_Frozen):
    # rate at which an action bias drifts towards its outcome value (scaled by learning rate);
    # fast enough that the press habit wanes within PR and extinction, not after them (was 0.1)
    rate: Unit = 0.2
    # action biases are clamped to [-bound, bound]
    bound: Positive = 1.0
    # weight of the evaluator's alignment in the outcome value
    alignment_weight: NonNegative = 0.2


class ConflictParams(_Frozen):
    # whether Reconcile may send control back to Deliberate on a habit-vs-goal conflict
    enabled: bool = True
    # action bias at or above which a downward goal revision counts as a conflict
    habit_threshold: float = 0.1


def _default_action_costs() -> dict[Action, float]:
    return {Action.PRESS_A: 0.05, Action.PRESS_B: 0.05, Action.EXPLORE: 0.02, Action.REST: 0.0}


class DeliberateParams(_Frozen):
    # softmax temperature of the stub's action choice
    temperature: Positive = 0.1
    # expected reward of a press before any experience of it
    prior_expected_reward: float = 0.3
    # pseudo-count weight of that prior
    prior_weight: Positive = 1.0
    # effort cost of each action
    action_costs: dict[Action, float] = Field(default_factory=_default_action_costs)
    # how much appraised valence pulls towards pressing
    approach_weight: float = 0.2
    # how much appraised arousal pulls towards pressing; strong enough that arousal lingering
    # under high mood_inertia measurably prolongs pressing in extinction (was 0.5)
    vigor_weight: float = 2.0
    # base value of exploring; with rest_value, a clear alternative to a press that no longer
    # pays, so the agent quits within a phase instead of choosing near-randomly (was 0.1)
    explore_value: float = 0.3
    # base value of resting (was 0.05)
    rest_value: float = 0.3
    # recalled episodes of an action at which prediction confidence reaches 0.5
    confidence_halfsat: Positive = 3.0

    @field_validator("action_costs")
    @classmethod
    def _every_action(cls, value: dict[Action, float]) -> dict[Action, float]:
        return _require_keys(value, Action)


class AppraiseParams(_Frozen):
    # how strongly an observed reward lifts the stub's appraised valence
    reward_gain: NonNegative = 0.5


class EvaluateParams(_Frozen):
    # utility difference that maps to alignment tanh(1) in the stub evaluator
    scale: Positive = 0.2


class MetricsParams(_Frozen):
    # consecutive steps without an active press that count as quitting
    quit_steps: int = Field(default=10, ge=1)


class LLMParams(_Frozen):
    # how many times a phase agent may resend an output that fails validation (out of range,
    # unknown action, malformed) before the run stops with an error; there is no stub fallback
    output_retries: int = Field(default=2, ge=0)
    # sampling temperature sent to the model; 1.0 keeps the model's own action distribution,
    # the analogue of the stub's softmax sampling. null sends none (for models that reject it)
    temperature: NonNegative | None = 1.0
    # per-request timeout in seconds
    timeout_s: Positive = 60.0


class Params(_Frozen):
    monitor: MonitorParams = MonitorParams()
    mood: MoodParams = MoodParams()
    memory: MemoryParams = MemoryParams()
    goals: GoalParams = GoalParams()
    dispositions: DispositionParams = DispositionParams()
    conflict: ConflictParams = ConflictParams()
    deliberate: DeliberateParams = DeliberateParams()
    appraise: AppraiseParams = AppraiseParams()
    evaluate: EvaluateParams = EvaluateParams()
    metrics: MetricsParams = MetricsParams()
    llm: LLMParams = LLMParams()


Backend = Literal["stub", "llm"]


class Backends(_Frozen):
    appraise: Backend = "stub"
    deliberate: Backend = "stub"
    evaluate: Backend = "stub"


def _default_goal_priorities() -> dict[GoalName, float]:
    return {GoalName.SEEK_REWARD: 1.0, GoalName.EXPLORE: 0.5, GoalName.CONSERVE: 0.5}


class AgentSpec(_Frozen):
    name: SafeName
    traits: Traits
    mood: Mood = Mood(valence=0.0, arousal=0.0)
    goal_priorities: dict[GoalName, float] = Field(default_factory=_default_goal_priorities)

    @field_validator("goal_priorities")
    @classmethod
    def _every_goal(cls, value: dict[GoalName, float]) -> dict[GoalName, float]:
        return _require_keys(value, GoalName)


class RunConfig(_Frozen):
    run_id: str
    seed: int
    ticks: int | None = Field(ge=1)
    world: WorldConfig
    agent: AgentSpec
    params: Params
    backends: Backends
    model: str | None


class ExperimentConfig(_Frozen):
    name: SafeName
    seeds: list[int] = Field(min_length=1)
    ticks: int | None = Field(default=None, ge=1)
    world: WorldConfig
    agents: list[AgentSpec] = Field(min_length=1)
    params: Params = Params()
    backends: Backends = Backends()
    model: str | None = None

    @field_validator("agents")
    @classmethod
    def _unique_names(cls, value: list[AgentSpec]) -> list[AgentSpec]:
        names = [a.name for a in value]
        if len(set(names)) != len(names):
            raise ValueError("agent names must be unique")
        return value

    def runs(self) -> list[RunConfig]:
        """One run per agent and seed: all seeds of the first agent first."""
        return [
            RunConfig(
                run_id=f"{self.name}-{agent.name}-s{seed}",
                seed=seed,
                ticks=self.ticks,
                world=self.world,
                agent=agent,
                params=self.params,
                backends=self.backends,
                model=self.model,
            )
            for agent in self.agents
            for seed in self.seeds
        ]


class ConfigError(ValueError):
    """A config file that cannot be read or does not validate."""


def load_config(path: Path) -> ExperimentConfig:
    try:
        text = Path(path).read_text()
    except FileNotFoundError:
        raise ConfigError(f"{path}: file not found") from None
    except OSError as e:
        raise ConfigError(f"{path}: cannot read: {e.strerror}") from None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        problem = getattr(e, "problem", None) or type(e).__name__
        raise ConfigError(f"{path}: invalid YAML{where}: {problem}") from None
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: the top level must be a mapping of config keys")
    try:
        return ExperimentConfig.model_validate(data)
    except ValidationError as e:
        lines = [f"{path}: invalid config"]
        for err in e.errors():
            key = ".".join(str(part) for part in err["loc"]) or "(top level)"
            lines.append(f"  {key}: {err['msg']}")
        raise ConfigError("\n".join(lines)) from None


def resolve_model(config: RunConfig) -> str | None:
    """The model name when any backend is "llm": config first, then the env var."""
    if "llm" not in config.backends.model_dump().values():
        return None
    if config.model:
        return config.model
    if env := os.environ.get(MODEL_ENV_VAR):
        return env
    raise ConfigError(f'a backend is "llm" but neither config `model` nor {MODEL_ENV_VAR} is set')


def initial_agent_state(config: RunConfig) -> AgentState:
    spec = config.agent
    return AgentState(
        traits=spec.traits,
        mood=spec.mood,
        memory=Memory(episodes=[]),
        goals=Goals(
            priorities=dict(spec.goal_priorities),
            error_ema={g: 0.0 for g in GoalName},
            error_streak={g: 0 for g in GoalName},
            revisions=0,
        ),
        dispositions=Dispositions(biases={a: 0.0 for a in Action}),
        tick=0,
    )


def derive_rng(seed: int, *key: int | str) -> random.Random:
    """A generator keyed by seed and context; string seeding is stable across processes."""
    return random.Random(":".join(map(str, (seed, *key))))


@dataclass(frozen=True)
class Runtime:
    config: RunConfig
    world: World
    model: object | None = None  # resolved pydantic-ai Model when a backend is "llm"
