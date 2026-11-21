"""Experiment configuration, parameters and run setup.

Every number that shapes behaviour lives here or in a creature's `Temperament`. Each default can
be overridden from YAML.
"""

import os
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import Field, ValidationError, field_validator

from surprise_loop.state import ChannelWeights, NonNegative, Step5, Sureness, Temperament, every
from surprise_loop.world import (
    Channel,
    Frozen,
    Unit,
    World,
    WorldConfig,
)
from surprise_loop.world import derive_rng as derive_rng  # re-exported for the phases

MODEL_ENV_VAR = "SURPRISE_LOOP_MODEL"

Positive = Annotated[float, Field(gt=0)]
SafeName = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.-]+$")]


def _edges(value: tuple[float, ...], count: int) -> tuple[float, ...]:
    """`count` edges between the words of a ladder of `count + 1`, strictly ascending."""
    if len(value) != count:
        raise ValueError(f"needs exactly {count} edges")
    if any(a >= b for a, b in pairwise(value)):
        raise ValueError("edges must be strictly ascending")
    return value


class LensParams(Frozen):
    # where a felt level crosses from one of the five words to the next
    edges: tuple[float, ...] = (0.1, 0.3, 0.55, 0.8)

    @field_validator("edges")
    @classmethod
    def _five_words(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        return _edges(value, 4)


def _tone_weights() -> dict[Channel, float]:
    return dict.fromkeys(Channel, 1.0) | {Channel.HUNGER: 0.5, Channel.THIRST: 0.5}


class FeelingParams(Frozen):
    # how much each channel weighs in the tone of feeling, and the edges of its five words
    tone_weights: ChannelWeights = Field(default_factory=_tone_weights)
    tone_edges: tuple[float, ...] = (-0.75, -0.3, 0.15, 0.6)
    # how fast stir follows the size of the surprise, and the edges of its three words
    stir_rate: Unit = 0.5
    stir_edges: tuple[float, ...] = (0.15, 0.4)

    @field_validator("tone_edges")
    @classmethod
    def _tone(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        return _edges(value, 4)

    @field_validator("stir_edges")
    @classmethod
    def _stir(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        return _edges(value, 2)


def _sure_factor() -> dict[Sureness, float]:
    return {Sureness.GUESSING: 0.5, Sureness.FAIRLY_SURE: 0.75, Sureness.SURE: 1.0}


class MonitorParams(Frozen):
    # the weight of each kind of miss in the surprise, and the miss that makes it whole
    moved_weight: NonNegative = 1.0
    channel_weight: ChannelWeights = Field(default_factory=lambda: dict.fromkeys(Channel, 1.0))
    sense_weight: NonNegative = 0.5
    scale: Positive = 2.0
    # how much of the surprise a miss counts for, by how sure the expectation was
    sure_factor: Annotated[dict[Sureness, NonNegative], every(Sureness)] = Field(
        default_factory=_sure_factor
    )
    # none, small, large, for what the model is shown
    surprise_edges: tuple[float, ...] = (0.05, 0.4)
    # |worse| below this is "as expected"
    worse_edge: NonNegative = 0.1

    @field_validator("surprise_edges")
    @classmethod
    def _three_words(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        return _edges(value, 2)


class ReasonerParams(Frozen):
    # the chance that an event is seen when its cause is absent, or missed when present
    slip: Unit = 0.05
    # the chance of an ache from eating when only slightly hungry
    ache_when_slight: Unit = 0.5


class RuleParams(Frozen):
    # the least felt step of hunger or thirst that is a need
    need_at: Step5 = 2
    # a belief at or above these serves a need, makes a tile unsafe, is believed
    serve_at: Unit = 0.5
    avoid_at: Unit = 0.5
    believe_at: Unit = 0.5
    # how far from the start the search looks, in steps along each axis
    reach: int = Field(default=12, ge=1)


class MemoryParams(Frozen):
    # how many recent episodes are kept and recalled
    recent_k: int = Field(default=6, ge=0)


class AppraiseParams(Frozen):
    # the most changes to the map one appraisal may make
    max_changes: int = Field(default=12, ge=1)


class PromptParams(Frozen):
    # decimals of every belief a model reads
    decimals: int = Field(default=2, ge=0)


def _comfort_at() -> dict[Channel, float]:
    return {
        Channel.HUNGER: 0.55,
        Channel.THIRST: 0.55,
        Channel.ACHE: 0.3,
        Channel.SICKNESS: 0.3,
        Channel.UNEASE: 0.3,
        Channel.PLEASURE: 1.0,
    }


class MetricsParams(Frozen):
    # a channel counts as comfortable while its true level is at or below this
    comfort_at: Annotated[dict[Channel, Unit], every(Channel)] = Field(default_factory=_comfort_at)
    # a belief is on the wrong side when the truth is on the other side of this
    wrong_side: Unit = 0.5
    # the film rings a tile when a belief moved by at least this
    rewritten_by: Unit = 0.1


class LLMParams(Frozen):
    # how many times a phase agent may resend an output that fails validation before the run
    # stops with an error; there is no stand-in fallback
    output_retries: int = Field(default=3, ge=0)
    # sampling temperature sent to the model; null sends none (for models that reject it)
    temperature: NonNegative | None = 1.0
    # per-request timeout in seconds
    timeout_s: Positive = 60.0


class Params(Frozen):
    lens: LensParams = LensParams()
    feeling: FeelingParams = FeelingParams()
    monitor: MonitorParams = MonitorParams()
    reasoner: ReasonerParams = ReasonerParams()
    rule: RuleParams = RuleParams()
    memory: MemoryParams = MemoryParams()
    appraise: AppraiseParams = AppraiseParams()
    prompt: PromptParams = PromptParams()
    metrics: MetricsParams = MetricsParams()
    llm: LLMParams = LLMParams()


class Backends(Frozen):
    deliberate: Literal["rule", "blind", "llm"] = "rule"
    appraise: Literal["reasoner", "llm"] = "reasoner"


class AgentSpec(Frozen):
    name: SafeName
    temperament: Temperament


class RunConfig(Frozen):
    run_id: str
    seed: int
    ticks: int | None = Field(ge=1)
    world: WorldConfig
    agent: AgentSpec
    params: Params
    backends: Backends
    model: str | None


class ExperimentConfig(Frozen):
    name: SafeName
    seeds: list[int] = Field(min_length=1)
    ticks: int | None = Field(default=None, ge=1)
    world: WorldConfig
    agents: list[AgentSpec] = Field(min_length=1)
    params: Params = Params()
    backends: Backends = Backends()
    model: str | None = None

    @field_validator("seeds")
    @classmethod
    def _unique_seeds(cls, value: list[int]) -> list[int]:
        # a repeated seed would give two runs the same run_id and run directory
        if len(set(value)) != len(value):
            raise ValueError("seeds must be unique")
        return value

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
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"{path}: file not found") from None
    except UnicodeDecodeError:
        raise ConfigError(f"{path}: cannot read: not UTF-8 text") from None
    except OSError as e:
        raise ConfigError(f"{path}: cannot read: {e.strerror or e}") from None
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


@dataclass(frozen=True)
class Runtime:
    config: RunConfig
    world: World
    model: object | None = None  # resolved pydantic-ai Model when a backend is "llm"
