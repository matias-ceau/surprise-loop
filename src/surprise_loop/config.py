"""Experiment configuration, parameters and run setup.

Every number that shapes behaviour lives here or in a body's `Temperament`. Each default can be
overridden from YAML.
"""

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import Field, ValidationError, field_validator, model_validator

from surprise_loop.state import (
    AgentState,
    Feeling,
    Memory,
    Pulls,
    SurpriseNote,
    Temperament,
    require_keys,
)
from surprise_loop.world import (
    Action,
    BodyState,
    Channel,
    FactKind,
    Frozen,
    Levels,
    Place,
    Unit,
    World,
    WorldConfig,
)

MODEL_ENV_VAR = "SURPRISE_LOOP_MODEL"

NonNegative = Annotated[float, Field(ge=0)]
Positive = Annotated[float, Field(gt=0)]
SafeName = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.-]+$")]


def _ascending_edges(value: tuple[float, ...]) -> tuple[float, ...]:
    if not value:
        raise ValueError("needs at least one edge")
    if any(a >= b for a, b in zip(value, value[1:], strict=False)):
        raise ValueError("edges must be strictly ascending")
    return value


class LensParams(Frozen):
    # sharpness of the softmax over fact relevance, and how arousal sharpens it
    gain_base: NonNegative = 0.5
    gain_arousal: NonNegative = 1.5
    # how many facts get through: fewer when stirred up, never fewer than min_noticed
    min_noticed: int = Field(default=3, ge=1)
    drop_per_arousal: NonNegative = 4.0
    # extra relevance for a fact bearing on the channel the last reading was about
    attention_bonus: NonNegative = 0.3
    # how many body channels may speak, at or above narrow_arousal and below it
    narrow_arousal: Unit = 0.5
    body_lines_narrow: int = Field(default=2, ge=0)
    body_lines_wide: int = Field(default=3, ge=0)
    # the least body grade that gets a sentence
    body_speak_grade: int = Field(default=2, ge=1)
    # grade edges for a fact's value and a channel's felt strength, positive and strictly
    # ascending; the first value edge is also where a value stops being neutral
    value_edges: tuple[Positive, ...] = (0.25, 1.0, 2.0, 3.0)
    body_edges: tuple[Positive, ...] = (0.5, 1.0, 2.0, 3.0)

    @field_validator("value_edges", "body_edges")
    @classmethod
    def _edges(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        return _ascending_edges(value)


class FeelingParams(Frozen):
    # how strongly standing discomfort lowers valence
    tone_gain: NonNegative = 0.12
    # the phasic lift from a better-than-expected outcome, and its decay per tick
    phasic_gain: NonNegative = 0.3
    phasic_decay: Unit = 0.5
    # arousal's target: surprise size, plus the loudest need relative to strain_full
    surprise_gain: NonNegative = 0.5
    strain_gain: NonNegative = 0.6
    strain_full: Positive = 3.0
    arousal_inertia: Unit = 0.5
    mood_inertia: Unit = 0.9


def _per_channel(heat: float, water: float, energy: float) -> dict[Channel, float]:
    return {Channel.HEAT: heat, Channel.WATER: water, Channel.ENERGY: energy}


class ExpectationParams(Frozen):
    # a model's change words in felt strength: "a bit" from bit to much, "much" beyond much
    bit: Positive = 0.25
    much: Positive = 1.0
    # half the width of a stub's band around its forecast, in channel units
    stub_tolerance: dict[Channel, NonNegative] = Field(
        default_factory=lambda: _per_channel(0.12, 0.06, 0.03)
    )

    @field_validator("stub_tolerance")
    @classmethod
    def _every_channel(cls, value: dict[Channel, float]) -> dict[Channel, float]:
        return require_keys(value, Channel)

    @model_validator(mode="after")
    def _ordered(self) -> "ExpectationParams":
        if self.bit > self.much:
            raise ValueError("bit must be <= much")
        return self


class MonitorParams(Frozen):
    # the size a world mismatch adds to a comparison, beside the body's felt error
    world_weight: NonNegative = 1.0


class BeliefParams(Frozen):
    # how strongly the expectation holds against the signal: k = trust / (trust + weight)
    expectation_weight: dict[Channel, Positive] = Field(
        default_factory=lambda: _per_channel(0.5, 0.5, 0.5)
    )

    @field_validator("expectation_weight")
    @classmethod
    def _every_channel(cls, value: dict[Channel, float]) -> dict[Channel, float]:
        return require_keys(value, Channel)


class StubModelParams(Frozen):
    """The stubs' private one-step model of their own body."""

    # heat moves by heat_rate toward the place's heat pull times heat_per_bearing
    heat_rate: Unit = 0.1
    heat_per_bearing: float = 0.5
    # water falls by water_drain, more by sweat_water per unit of heat, rises by drink_water
    water_drain: NonNegative = 0.012
    sweat_water: NonNegative = 0.01
    drink_water: NonNegative = 0.04
    # energy falls by walk_energy for a walk, rises by rest_energy for a rest
    walk_energy: NonNegative = 0.03
    rest_energy: NonNegative = 0.05


class MemoryParams(Frozen):
    # how many episodes Perceive recalls, and how they are chosen
    retrieval_k: int = Field(default=4, ge=1)
    # salience = min(1, base_salience + salience_gain * comparison size)
    base_salience: Unit = 0.2
    salience_gain: NonNegative = 0.3
    retrieval: Literal["last_k", "salient_k"] = "last_k"


class EvaluateParams(Frozen):
    # an episode is judged this many episodes after it was stored
    window: int = Field(default=4, ge=1)
    # change of need that maps to a score of 1
    score_scale: Positive = 1.0
    # |score| at which an action counts as having helped or hurt
    helped: Unit = 0.25


class PullParams(Frozen):
    # how far a judged score moves the pull of that action at that place
    rate: Unit = 0.5
    # the least |pull| that Perceive puts into words
    speak: NonNegative = 0.2


class OverrideParams(Frozen):
    enabled: bool = True
    # a need at least this loud after the update, and at least worse_margin worse than expected
    urgent_felt: NonNegative = 1.5
    worse_margin: NonNegative = 0.2


class DeliberateParams(Frozen):
    stub: Literal["reactive", "lookahead", "blind"] = "reactive"
    # the least |need| the reactive rule acts on
    act_threshold: NonNegative = 0.5
    # steps the lookahead stub plans over
    horizon: int = Field(default=6, ge=1)
    model: StubModelParams = StubModelParams()


class ReconcileParams(Frozen):
    # the least |felt error| on a channel that becomes the voiced surprise
    voice_threshold: NonNegative = 0.5


class MetricsParams(Frozen):
    # a channel counts as comfortable while its |need| is below this
    comfort_felt: NonNegative = 1.0


class LLMParams(Frozen):
    # how many times a phase agent may resend an output that fails validation (out of range,
    # unknown action, malformed) before the run stops with an error; there is no stub fallback
    output_retries: int = Field(default=3, ge=0)
    # sampling temperature sent to the model; null sends none (for models that reject it)
    temperature: NonNegative | None = 1.0
    # per-request timeout in seconds
    timeout_s: Positive = 60.0


Scene = Literal["A", "B"]


class LadderScene(Frozen):
    """The believed water and energy of one scene of the ladder check."""

    water: Unit
    energy: Unit


class LadderParams(Frozen):
    """The ladder check's protocol: a frozen noon in the meadow, the body's believed heat swept
    over the levels, and the pass mark on the share of answers choosing the oak's shade."""

    # believed heat at each level, coolest first: skin 21, 23, 25, 27, 30, 33 and 36 degrees C
    heat_levels: tuple[float, ...] = (-0.4, -0.2, 0.0, 0.2, 0.5, 0.8, 1.1)
    # the swept body: set point, tightness, arousal and the standing habit for every kind of fact
    setpoint: Levels = Levels(heat=0.0, water=0.80, energy=0.70)
    tightness: dict[Channel, Positive] = Field(default_factory=lambda: _per_channel(3.0, 5.0, 5.0))
    arousal: Unit = 0.40
    habit: dict[FactKind, float] = Field(default_factory=lambda: dict.fromkeys(FactKind, 0.3))
    # scene A: heat is the only need; scene B: a fixed moderate thirst and tiredness compete
    scenes: dict[Scene, LadderScene] = Field(
        default_factory=lambda: {
            "A": LadderScene(water=0.80, energy=0.70),
            "B": LadderScene(water=0.50, energy=0.46),
        }
    )
    # answers per cell (scene and level), and how many calls run at once
    samples: int = Field(default=30, ge=1)
    jobs: int = Field(default=8, ge=1)
    # the pass mark, per scene: slope of the shade share per level at least min_slope, share at
    # the hottest level minus the coolest at least min_range, and no level's share above the next
    # warmer level's by max_drop or more
    min_slope: float = 0.05
    min_range: float = 0.5
    max_drop: float = 0.25

    @field_validator("heat_levels")
    @classmethod
    def _levels(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if len(value) < 2:
            raise ValueError("needs at least two levels")
        if any(a >= b for a, b in zip(value, value[1:], strict=False)):
            raise ValueError("levels must be strictly ascending")
        return value

    @field_validator("tightness")
    @classmethod
    def _every_channel(cls, value: dict[Channel, float]) -> dict[Channel, float]:
        return require_keys(value, Channel)

    @field_validator("habit")
    @classmethod
    def _every_kind(cls, value: dict[FactKind, float]) -> dict[FactKind, float]:
        return require_keys(value, FactKind)

    @field_validator("scenes")
    @classmethod
    def _both_scenes(cls, value: dict[Scene, LadderScene]) -> dict[Scene, LadderScene]:
        if set(value) != {"A", "B"}:
            raise ValueError("needs scenes A and B")
        return value


class Params(Frozen):
    lens: LensParams = LensParams()
    feeling: FeelingParams = FeelingParams()
    expectation: ExpectationParams = ExpectationParams()
    monitor: MonitorParams = MonitorParams()
    belief: BeliefParams = BeliefParams()
    memory: MemoryParams = MemoryParams()
    evaluate: EvaluateParams = EvaluateParams()
    pulls: PullParams = PullParams()
    override: OverrideParams = OverrideParams()
    deliberate: DeliberateParams = DeliberateParams()
    reconcile: ReconcileParams = ReconcileParams()
    metrics: MetricsParams = MetricsParams()
    llm: LLMParams = LLMParams()
    ladder: LadderParams = LadderParams()


Backend = Literal["stub", "llm"]


class Backends(Frozen):
    appraise: Backend = "stub"
    deliberate: Backend = "stub"


class AgentSpec(Frozen):
    name: SafeName
    temperament: Temperament
    body: BodyState  # the true body at the start
    belief: Levels  # the belief at the start
    arousal: Unit = 0.2
    voiced: SurpriseNote | None = None


class RunConfig(Frozen):
    run_id: str
    seed: int
    ticks: int | None = Field(ge=1)
    world: WorldConfig
    start_place: Place
    agent: AgentSpec
    params: Params
    backends: Backends
    model: str | None


class ExperimentConfig(Frozen):
    name: SafeName
    seeds: list[int] = Field(min_length=1)
    ticks: int | None = Field(default=None, ge=1)
    world: WorldConfig = WorldConfig()
    start_place: Place = Place.MEADOW
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
                start_place=self.start_place,
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
    """The spec's temperament and belief, a feeling settled from the baseline with the spec's
    arousal, empty memory, every pull 0, the spec's voiced surprise."""
    # feeling.py imports this module for FeelingParams, so it is imported here, not at the top
    from surprise_loop.feeling import feel

    spec = config.agent
    t = spec.temperament
    start = Feeling(
        discomfort=0, valence=t.baseline, arousal=spec.arousal, mood=t.baseline, phasic=0
    )
    feeling = feel(spec.belief, t, start, None, config.params.feeling)
    return AgentState(
        temperament=t,
        belief=spec.belief,
        feeling=feeling.model_copy(update={"arousal": spec.arousal}),
        memory=Memory(episodes=[]),
        pulls=Pulls(values={place: {action: 0.0 for action in Action} for place in Place}),
        attention=None,
        voiced=spec.voiced,
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
