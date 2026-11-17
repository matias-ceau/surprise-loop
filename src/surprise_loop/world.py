"""The clearing's vocabulary: places, actions, facts, the body's true state, and the world config.

The World is the only owner of environment state. This module holds the shared types and tables;
the agent only ever sees `Observation`, and `StepInfo` (true air, sun, body) is for logging and
metrics, never for a prompt.

Units: heat is `(skin °C - 25) / 10`, so 0 is neutral for a reference body; water and energy run
from 0 to 1. A fact's bearing says how much it raises a channel's strain (heat strain, thirst,
tiredness); a negative bearing relieves it.
"""

from enum import StrEnum
from typing import Annotated, Protocol

from pydantic import BaseModel, ConfigDict, Field


class Frozen(BaseModel):
    """The base of every state, config and result model: immutable, no unknown fields, no NaN."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


Unit = Annotated[float, Field(ge=0, le=1)]


class Channel(StrEnum):
    HEAT = "heat"
    WATER = "water"
    ENERGY = "energy"


class Place(StrEnum):
    MEADOW = "meadow"
    OAK = "oak"
    STREAM = "stream"


class Action(StrEnum):
    GO_MEADOW = "go_meadow"
    GO_OAK = "go_oak"
    GO_STREAM = "go_stream"
    DRINK = "drink"
    REST = "rest"
    LOOK = "look"


class Event(StrEnum):
    START = "start"
    ARRIVED = "arrived"
    STUMBLED = "stumbled"
    DRANK = "drank"
    RESTED = "rested"
    LOOKED = "looked"
    COLLAPSED = "collapsed"


class FactKind(StrEnum):
    BURDEN = "burden"
    RELIEF = "relief"
    AMBIENT = "ambient"


class FactName(StrEnum):
    """This order is the canonical order of a fact list."""

    AIR = "air"
    COOL_AIR = "cool_air"
    SUN = "sun"
    SHADE = "shade"
    NO_SEAT = "no_seat"
    LOG = "log"
    LOG_SUN = "log_sun"
    OAK_SHADE = "oak_shade"
    OAK_SHADE_FAR = "oak_shade_far"
    OAK_LOG_SHADED = "oak_log_shaded"
    OAK_LOG_SUNLIT = "oak_log_sunlit"
    WATER = "water"
    TRICKLE = "trickle"
    STREAM = "stream"
    MEADOW_SUN = "meadow_sun"
    BREEZE = "breeze"
    BIRDSONG = "birdsong"


class OpenQuestion(StrEnum):
    LOG_SHADED = "log_shaded"
    STREAM_FULL = "stream_full"


GO_TARGET: dict[Action, Place] = {
    Action.GO_MEADOW: Place.MEADOW,
    Action.GO_OAK: Place.OAK,
    Action.GO_STREAM: Place.STREAM,
}

SERVES: dict[Action, Channel | None] = {
    Action.GO_MEADOW: Channel.HEAT,
    Action.GO_OAK: Channel.HEAT,
    Action.GO_STREAM: Channel.WATER,
    Action.DRINK: Channel.WATER,
    Action.REST: Channel.ENERGY,
    Action.LOOK: None,
}

QUESTION_OF: dict[Action, OpenQuestion | None] = {
    Action.GO_MEADOW: None,
    Action.GO_OAK: OpenQuestion.LOG_SHADED,
    Action.GO_STREAM: OpenQuestion.STREAM_FULL,
    Action.DRINK: None,
    Action.REST: None,
    Action.LOOK: None,
}

FACT_KIND: dict[FactName, FactKind] = {
    FactName.AIR: FactKind.BURDEN,
    FactName.COOL_AIR: FactKind.RELIEF,
    FactName.SUN: FactKind.BURDEN,
    FactName.SHADE: FactKind.RELIEF,
    FactName.NO_SEAT: FactKind.BURDEN,
    FactName.LOG: FactKind.RELIEF,
    FactName.LOG_SUN: FactKind.BURDEN,
    FactName.OAK_SHADE: FactKind.RELIEF,
    FactName.OAK_SHADE_FAR: FactKind.RELIEF,
    FactName.OAK_LOG_SHADED: FactKind.RELIEF,
    FactName.OAK_LOG_SUNLIT: FactKind.RELIEF,
    FactName.WATER: FactKind.RELIEF,
    FactName.TRICKLE: FactKind.RELIEF,
    FactName.STREAM: FactKind.RELIEF,
    FactName.MEADOW_SUN: FactKind.AMBIENT,
    FactName.BREEZE: FactKind.RELIEF,
    FactName.BIRDSONG: FactKind.AMBIENT,
}

FACT_ABOUT: dict[FactName, Place | None] = {
    FactName.AIR: None,
    FactName.COOL_AIR: None,
    FactName.SUN: None,
    FactName.SHADE: None,
    FactName.NO_SEAT: None,
    FactName.LOG: None,
    FactName.LOG_SUN: None,
    FactName.OAK_SHADE: Place.OAK,
    FactName.OAK_SHADE_FAR: Place.OAK,
    FactName.OAK_LOG_SHADED: Place.OAK,
    FactName.OAK_LOG_SUNLIT: Place.OAK,
    FactName.WATER: None,
    FactName.TRICKLE: None,
    FactName.STREAM: Place.STREAM,
    FactName.MEADOW_SUN: Place.MEADOW,
    FactName.BREEZE: None,
    FactName.BIRDSONG: None,
}


class Levels(Frozen):
    """One value per channel, in the channel's own units."""

    heat: float
    water: float
    energy: float

    def of(self, c: Channel) -> float:
        return getattr(self, c.value)

    def replace(self, c: Channel, value: float) -> "Levels":
        return self.model_copy(update={c.value: value})


class Fact(Frozen):
    name: FactName
    kind: FactKind
    about: Place | None
    bearing: dict[Channel, float]


class BodyState(Frozen):
    heat: float
    water: Unit
    energy: Unit
    gut: float = Field(default=0, ge=0)


class Observation(Frozen):
    step: int
    minute: int
    place: Place
    last_action: Action | None
    event: Event
    facts: list[Fact]
    signal: Levels
    offers: list[Action]


class StepInfo(Frozen):
    """The world's truth after a step. Never crosses into a prompt."""

    step: int
    minute: int
    place: Place
    air_c: float
    sun: Unit
    log_shaded: bool
    stream_full: bool
    body: BodyState
    stumbled: bool
    collapsed: bool


class Outcome(Frozen):
    observation: Observation
    info: StepInfo


class BodyParams(Frozen):
    heat_rate: float = 0.25
    air_gain: float = 0.5
    sun_gain: float = 0.5
    walk_heat: float = 0.03
    sweat_threshold: float = 0.3
    sweat_rate: float = 0.5
    sweat_cooling: float = 0.2
    sweat_water: float = 0.03
    water_drain: float = 0.004
    base_energy: float = 0.004
    walk_energy: float = 0.01
    hot_walk_factor: float = 0.5
    rest_gain: float = 0.03
    ground_rest_quality: float = 0.5
    drink_amount: float = 0.25
    absorb_rate: float = 0.4
    stumble_energy: float = 0.08
    stumble_cost: float = 0.005
    heat_hot_limit: float = 1.2
    heat_cold_limit: float = -1.0
    water_floor: float = 0.05
    energy_floor: float = 0.03
    collapse_ticks: int = 3
    signal_noise: dict[Channel, float] = {
        Channel.HEAT: 0.02,
        Channel.WATER: 0.02,
        Channel.ENERGY: 0.02,
    }
    signal_step: dict[Channel, float] = {
        Channel.HEAT: 0.05,
        Channel.WATER: 0.05,
        Channel.ENERGY: 0.05,
    }
    water_delay: int = 2


# The constant bearings of the fact table. AIR, COOL_AIR and SUN are computed from the hour and
# the place. Warm and cool air are two facts because a phrase knows only name, sign and grade;
# likewise the oak's shade seen from the meadow (near) and from the stream (far).
DEFAULT_BEARINGS: dict[FactName, dict[Channel, float]] = {
    FactName.SHADE: {Channel.HEAT: -1.0},
    FactName.NO_SEAT: {Channel.ENERGY: 1.0},
    FactName.LOG: {Channel.ENERGY: -1.0},
    FactName.LOG_SUN: {Channel.ENERGY: 0.5},
    FactName.OAK_SHADE: {Channel.HEAT: -1.0},
    FactName.OAK_SHADE_FAR: {Channel.HEAT: -1.0},
    FactName.OAK_LOG_SHADED: {Channel.HEAT: -1.0, Channel.ENERGY: -0.6},
    FactName.OAK_LOG_SUNLIT: {Channel.HEAT: -1.0},
    FactName.WATER: {Channel.WATER: -1.0, Channel.HEAT: -0.3},
    FactName.TRICKLE: {Channel.WATER: -0.4},
    FactName.STREAM: {Channel.WATER: -1.0, Channel.HEAT: -0.3},
    FactName.MEADOW_SUN: {Channel.HEAT: 1.0},
    FactName.BREEZE: {Channel.HEAT: -0.3},
    FactName.BIRDSONG: {},
}


class WorldConfig(Frozen):
    start_minute: int = 360
    end_minute: int = 1200
    tick_minutes: int = 10
    # meadow air in °C, linear between points
    air_curve: list[tuple[int, float]] = [(360, 18.0), (780, 30.0), (960, 30.0), (1200, 22.0)]
    place_air_offset: dict[Place, float] = {Place.MEADOW: 0.0, Place.OAK: -3.0, Place.STREAM: -1.5}
    air_neutral_c: float = 25.0
    air_scale_c: float = 5.0
    # open sun from 0 to 1, linear between points
    sun_curve: list[tuple[int, float]] = [(360, 0.2), (660, 1.0), (900, 1.0), (1200, 0.1)]
    place_sun: dict[Place, float] = {Place.MEADOW: 1.0, Place.OAK: 0.0, Place.STREAM: 0.6}
    sun_fact_min: float = 0.3
    # the oak's log is in the sun from .. until
    log_sun_from: int = 660
    log_sun_until: int = 930
    stream_full_probability: float = 0.6
    trickle_factor: float = 0.35
    paces: dict[str, float] = {"meadow-oak": 1.0, "meadow-stream": 4.0, "oak-stream": 5.0}
    bearings: dict[FactName, dict[Channel, float]] = DEFAULT_BEARINGS
    body: BodyParams = BodyParams()


class WorldState(Frozen):
    seed: int
    step: int
    minute: int
    place: Place
    body: BodyState
    water_history: list[float]  # the last `water_delay` true water values, oldest first
    stream_full: bool
    forced_rest_left: int
    last_observation: Observation


class WorldDone(RuntimeError):
    """Raised by `step()` once the day has ended."""


class World(Protocol):
    @property
    def done(self) -> bool: ...
    def observe(self) -> Observation: ...
    def step(self, action: Action) -> Outcome: ...
    def snapshot(self) -> WorldState: ...


# The owner's example is the meadow at 13:00.
FROZEN_NOON_MINUTE = 13 * 60
FROZEN_NOON_FACTS = (
    FactName.AIR,
    FactName.SUN,
    FactName.NO_SEAT,
    FactName.OAK_SHADE,
    FactName.STREAM,
    FactName.BREEZE,
    FactName.BIRDSONG,
)


def _curve_value(curve: list[tuple[int, float]], minute: int) -> float:
    """Linear interpolation between the curve's points, flat beyond its ends."""
    if minute <= curve[0][0]:
        return curve[0][1]
    for (m0, v0), (m1, v1) in zip(curve, curve[1:], strict=False):
        if minute <= m1:
            return v0 + (v1 - v0) * (minute - m0) / (m1 - m0)
    return curve[-1][1]


def _fact(name: FactName, bearing: dict[Channel, float]) -> Fact:
    return Fact(name=name, kind=FACT_KIND[name], about=FACT_ABOUT[name], bearing=bearing)


def frozen_noon_facts() -> list[Fact]:
    """The seven facts of the owner's example, the meadow at 13:00, from the default config."""
    config = WorldConfig()
    here = Place.MEADOW
    air_c = _curve_value(config.air_curve, FROZEN_NOON_MINUTE) + config.place_air_offset[here]
    sun = _curve_value(config.sun_curve, FROZEN_NOON_MINUTE) * config.place_sun[here]
    computed = {
        FactName.AIR: {Channel.HEAT: (air_c - config.air_neutral_c) / config.air_scale_c},
        FactName.SUN: {Channel.HEAT: sun},
    }
    return [
        _fact(name, computed[name] if name in computed else dict(config.bearings[name]))
        for name in FROZEN_NOON_FACTS
    ]
