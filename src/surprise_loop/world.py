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

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)


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


NonNegative = Annotated[float, Field(ge=0)]


def _every_key[K, V](value: dict[K, V], keys: type[StrEnum], what: str) -> dict[K, V]:
    """Raise unless `value` has an entry for every member of `keys`."""
    missing = [k.value for k in keys if k not in value]
    if missing:
        raise ValueError(
            f"{what} needs an entry for every {keys.__name__.lower()}; "
            f"missing: {', '.join(missing)}"
        )
    return value


class BodyParams(Frozen):
    heat_rate: NonNegative = 0.2
    air_gain: NonNegative = 0.7
    sun_gain: NonNegative = 0.7
    walk_heat: NonNegative = 0.04
    sweat_threshold: float = 0.3
    sweat_rate: NonNegative = 0.5
    sweat_cooling: NonNegative = 0.2
    sweat_water: NonNegative = 0.03
    water_drain: NonNegative = 0.012
    base_energy: NonNegative = 0.015
    walk_energy: NonNegative = 0.005
    hot_walk_factor: NonNegative = 0.5
    rest_gain: NonNegative = 0.12
    ground_rest_quality: NonNegative = 0.5
    drink_amount: NonNegative = 0.25
    # the share of the gut absorbed per step: some, at most all
    absorb_rate: Annotated[float, Field(gt=0, le=1)] = 0.4
    stumble_energy: NonNegative = 0.08
    stumble_cost: NonNegative = 0.005
    heat_hot_limit: float = 1.2
    heat_cold_limit: float = -1.0
    water_floor: Annotated[float, Field(ge=0, lt=1)] = 0.05
    energy_floor: Annotated[float, Field(ge=0, lt=1)] = 0.03
    collapse_ticks: int = Field(default=3, ge=1)
    signal_noise: dict[Channel, NonNegative] = {
        Channel.HEAT: 0.005,
        Channel.WATER: 0.005,
        Channel.ENERGY: 0.005,
    }
    signal_step: dict[Channel, Annotated[float, Field(gt=0)]] = {
        Channel.HEAT: 0.01,
        Channel.WATER: 0.01,
        Channel.ENERGY: 0.01,
    }
    # how many steps late the water signal is
    water_delay: int = Field(default=2, ge=0)

    @field_validator("signal_noise", "signal_step")
    @classmethod
    def _every_channel(
        cls, value: dict[Channel, float], info: ValidationInfo
    ) -> dict[Channel, float]:
        return _every_key(value, Channel, info.field_name or "")

    @model_validator(mode="after")
    def _limits(self) -> "BodyParams":
        if self.heat_cold_limit >= self.heat_hot_limit:
            raise ValueError("heat_cold_limit must be below heat_hot_limit")
        return self


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
    air_curve: list[tuple[int, float]] = [(360, 23.0), (780, 30.0), (960, 30.0), (1200, 23.0)]
    place_air_offset: dict[Place, float] = {Place.MEADOW: 0.0, Place.OAK: -7.0, Place.STREAM: -3.0}
    air_neutral_c: float = 25.0
    air_scale_c: float = Field(default=5.0, gt=0)
    # open sun from 0 to 1, linear between points
    sun_curve: list[tuple[int, float]] = [(360, 0.2), (660, 1.0), (900, 1.0), (1200, 0.1)]
    place_sun: dict[Place, float] = {Place.MEADOW: 1.0, Place.OAK: 0.0, Place.STREAM: 0.6}
    sun_fact_min: float = 0.3
    # the oak's log is in the sun from .. until
    log_sun_from: int = 660
    log_sun_until: int = 930
    stream_full_probability: Unit = 0.6
    trickle_factor: Unit = 0.35
    paces: dict[str, float] = {"meadow-oak": 1.0, "meadow-stream": 4.0, "oak-stream": 5.0}
    bearings: dict[FactName, dict[Channel, float]] = DEFAULT_BEARINGS
    body: BodyParams = BodyParams()

    @field_validator("air_curve", "sun_curve")
    @classmethod
    def _curve(cls, value: list[tuple[int, float]]) -> list[tuple[int, float]]:
        if len(value) < 2:
            raise ValueError("a curve needs at least two points")
        if any(m0 >= m1 for (m0, _), (m1, _) in zip(value, value[1:], strict=False)):
            raise ValueError("a curve's minutes must be strictly increasing")
        return value

    @field_validator("place_air_offset", "place_sun")
    @classmethod
    def _every_place(cls, value: dict[Place, float], info: ValidationInfo) -> dict[Place, float]:
        return _every_key(value, Place, info.field_name or "")

    @field_validator("paces")
    @classmethod
    def _paces(cls, value: dict[str, float]) -> dict[str, float]:
        pairs = [frozenset(key.split("-")) for key in value]
        wanted = {frozenset((a, b)) for a in Place for b in Place if a is not b}
        if len(pairs) != len(wanted) or set(pairs) != wanted:
            raise ValueError(
                'paces needs exactly one "a-b" key for each pair of places: '
                + ", ".join(sorted("-".join(sorted(p)) for p in wanted))
            )
        return value

    @model_validator(mode="after")
    def _times(self) -> "WorldConfig":
        if self.start_minute >= self.end_minute:
            raise ValueError("start_minute must be before end_minute")
        if self.tick_minutes <= 0:
            raise ValueError("tick_minutes must be positive")
        if self.log_sun_from > self.log_sun_until:
            raise ValueError("log_sun_from must be <= log_sun_until")
        return self


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


def _curve_value(curve: list[tuple[int, float]], minute: int) -> float:
    """Linear interpolation between the curve's points, flat beyond its ends."""
    if minute <= curve[0][0]:
        return curve[0][1]
    for (m0, v0), (m1, v1) in zip(curve, curve[1:], strict=False):
        if minute <= m1:
            return v0 + (v1 - v0) * (minute - m0) / (m1 - m0)
    return curve[-1][1]


def air_at(config: WorldConfig, place: Place, minute: int) -> float:
    """The air at a place in °C: the meadow's curve plus the place's offset."""
    return _curve_value(config.air_curve, minute) + config.place_air_offset[place]


def sun_at(config: WorldConfig, place: Place, minute: int) -> float:
    """The sun at a place, 0 to 1: the open sun's curve times the place's share of it."""
    return _curve_value(config.sun_curve, minute) * config.place_sun[place]


def log_shaded_at(config: WorldConfig, minute: int) -> bool:
    """The oak's log is in the sun from `log_sun_from` until (not including) `log_sun_until`."""
    return not config.log_sun_from <= minute < config.log_sun_until


def _air_bearing(config: WorldConfig, place: Place, minute: int) -> float:
    return (air_at(config, place, minute) - config.air_neutral_c) / config.air_scale_c


def _fact(name: FactName, bearing: dict[Channel, float]) -> Fact:
    return Fact(name=name, kind=FACT_KIND[name], about=FACT_ABOUT[name], bearing=bearing)


def facts_at(
    config: WorldConfig, place: Place, minute: int, stream_full: bool, looked: bool
) -> list[Fact]:
    """The facts of a place at a minute, in `FactName` order. `looked` matters only at the
    meadow, where it shows whether the oak's log is shaded."""
    air = _air_bearing(config, place, minute)
    warm = air_at(config, place, minute) >= config.air_neutral_c
    sun = sun_at(config, place, minute)
    shaded = log_shaded_at(config, minute)
    present: dict[FactName, dict[Channel, float]] = {
        FactName.AIR if warm else FactName.COOL_AIR: {Channel.HEAT: air},
        FactName.BREEZE: dict(config.bearings[FactName.BREEZE]),
        FactName.BIRDSONG: dict(config.bearings[FactName.BIRDSONG]),
    }
    if sun >= config.sun_fact_min:
        present[FactName.SUN] = {Channel.HEAT: sun}
    named: list[FactName] = []
    match place:
        case Place.MEADOW:
            named += [FactName.NO_SEAT, FactName.STREAM]
            if not looked:
                named.append(FactName.OAK_SHADE)
            else:
                named.append(FactName.OAK_LOG_SHADED if shaded else FactName.OAK_LOG_SUNLIT)
        case Place.OAK:
            named += [FactName.SHADE, FactName.STREAM, FactName.LOG if shaded else FactName.LOG_SUN]
        case Place.STREAM:
            named += [FactName.NO_SEAT, FactName.OAK_SHADE_FAR]
            named.append(FactName.WATER if stream_full else FactName.TRICKLE)
    if place is not Place.MEADOW and sun_at(config, Place.MEADOW, minute) >= config.sun_fact_min:
        named.append(FactName.MEADOW_SUN)
    for name in named:
        present[name] = dict(config.bearings[name])
    return [_fact(name, present[name]) for name in FactName if name in present]


def frozen_noon_facts() -> list[Fact]:
    """The seven facts of the owner's example, the meadow at 13:00, from the default config."""
    return facts_at(WorldConfig(), Place.MEADOW, FROZEN_NOON_MINUTE, stream_full=True, looked=False)


def offers_at(place: Place) -> list[Action]:
    """The walks to the other places, drinking only at the stream, resting and looking always."""
    return [
        action
        for action in Action
        if (action not in GO_TARGET or GO_TARGET[action] is not place)
        and (action is not Action.DRINK or place is Place.STREAM)
    ]


def _pace(config: WorldConfig, a: Place, b: Place) -> float:
    """The pace between two places; `paces` may name a pair in either order."""
    key = f"{a.value}-{b.value}"
    return config.paces[key] if key in config.paces else config.paces[f"{b.value}-{a.value}"]


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


class ClearingWorld:
    """One forest clearing (meadow, oak, stream), a daytime clock and a true body.

    All state is in one `WorldState`; there is no stored generator (every draw comes from
    `derive_rng(seed, ...)`), so a restored world continues exactly as the original would.
    """

    def __init__(self, config: WorldConfig, seed: int, body: BodyState, place: Place) -> None:
        from surprise_loop.config import derive_rng  # config imports this module

        self._config = config
        stream_full = derive_rng(seed, "stream").random() < config.stream_full_probability
        history = [body.water] * config.body.water_delay
        signal, _ = self._signal(seed, 0, body, history)
        minute = config.start_minute
        observation = Observation(
            step=0,
            minute=minute,
            place=place,
            last_action=None,
            event=Event.START,
            facts=facts_at(config, place, minute, stream_full, looked=False),
            signal=signal,
            offers=offers_at(place),
        )
        self._state = WorldState(
            seed=seed,
            step=0,
            minute=minute,
            place=place,
            body=body,
            water_history=history,
            stream_full=stream_full,
            forced_rest_left=0,
            last_observation=observation,
        )

    @classmethod
    def restore(cls, config: WorldConfig, state: WorldState) -> "ClearingWorld":
        """The world `config` would be in at `state` (a snapshot)."""
        if len(state.water_history) != config.body.water_delay:
            raise ValueError(
                f"the state holds {len(state.water_history)} past water values, but the "
                f"config's water_delay is {config.body.water_delay}"
            )
        world = cls.__new__(cls)
        world._config = config
        world._state = state
        return world

    @property
    def done(self) -> bool:
        return self._state.minute >= self._config.end_minute

    def observe(self) -> Observation:
        return self._state.last_observation

    def snapshot(self) -> WorldState:
        return self._state

    def step(self, action: Action) -> Outcome:
        if self.done:
            raise WorldDone("the day has ended")
        s = self._state
        if action not in s.last_observation.offers:
            raise ValueError(f"{action.value} is not offered at the {s.place.value}")
        config, p = self._config, self._config.body
        heat, water, energy, gut = s.body.heat, s.body.water, s.body.energy, s.body.gut
        place, forced_rest_left = s.place, s.forced_rest_left
        collapsed = forced_rest_left > 0
        stumbled = looked = False

        # 1. the action
        done_action = Action.REST if collapsed else action
        if collapsed:
            forced_rest_left -= 1
        if done_action in GO_TARGET:
            if energy < p.stumble_energy:
                event, stumbled = Event.STUMBLED, True
                energy -= p.stumble_cost
            else:
                pace = _pace(config, place, GO_TARGET[done_action])
                event, place = Event.ARRIVED, GO_TARGET[done_action]
                energy -= p.walk_energy * pace * (1 + p.hot_walk_factor * max(0, heat))
                heat += p.walk_heat * pace * sun_at(config, Place.MEADOW, s.minute)
        elif done_action is Action.DRINK:
            event = Event.DRANK
            gut += p.drink_amount * (1 if s.stream_full else config.trickle_factor)
        elif done_action is Action.REST:
            event = Event.RESTED
            on_log = place is Place.OAK and log_shaded_at(config, s.minute)
            energy += p.rest_gain * (1 if on_log else p.ground_rest_quality)
        else:
            event, looked = Event.LOOKED, True
        if collapsed:
            event = Event.COLLAPSED

        # 2. the clock
        minute = s.minute + config.tick_minutes
        # 3. heat moves toward what the place pushes it to
        target = p.air_gain * _air_bearing(config, place, minute) + p.sun_gain * sun_at(
            config, place, minute
        )
        heat += p.heat_rate * (target - heat)
        # 4. sweat
        if heat > p.sweat_threshold:
            sweat = p.sweat_rate * (heat - p.sweat_threshold)
            heat -= p.sweat_cooling * sweat
            water -= p.sweat_water * sweat
        # 5. drain
        water -= p.water_drain
        energy -= p.base_energy
        # 6. absorption
        absorbed = gut * p.absorb_rate
        water += absorbed
        gut -= absorbed
        # 7. limits
        heat = _clamp(heat, p.heat_cold_limit, p.heat_hot_limit)
        water, energy = _clamp(water, 0, 1), _clamp(energy, 0, 1)
        at_limit = (
            heat in (p.heat_cold_limit, p.heat_hot_limit)
            or water <= p.water_floor
            or energy <= p.energy_floor
        )
        if at_limit and forced_rest_left == 0:
            forced_rest_left = p.collapse_ticks
        body = BodyState(heat=heat, water=water, energy=energy, gut=gut)

        # 8. the signal
        step = s.step + 1
        signal, history = self._signal(s.seed, step, body, s.water_history)
        observation = Observation(
            step=step,
            minute=minute,
            place=place,
            last_action=action,
            event=event,
            facts=facts_at(config, place, minute, s.stream_full, looked),
            signal=signal,
            offers=offers_at(place),
        )
        info = StepInfo(
            step=step,
            minute=minute,
            place=place,
            air_c=air_at(config, place, minute),
            sun=sun_at(config, place, minute),
            log_shaded=log_shaded_at(config, minute),
            stream_full=s.stream_full,
            body=body,
            stumbled=stumbled,
            collapsed=collapsed,
        )
        self._state = s.model_copy(
            update={
                "step": step,
                "minute": minute,
                "place": place,
                "body": body,
                "water_history": history,
                "forced_rest_left": forced_rest_left,
                "last_observation": observation,
            }
        )
        return Outcome(observation=observation, info=info)

    def _signal(
        self, seed: int, step: int, body: BodyState, history: list[float]
    ) -> tuple[Levels, list[float]]:
        """The body signal at `step` and the water history after it. Water is read
        `water_delay` steps late: the oldest value in the history."""
        from surprise_loop.config import derive_rng  # config imports this module

        p = self._config.body
        waters = [*history, body.water]
        true = Levels(heat=body.heat, water=waters[0], energy=body.energy)
        signal = true
        for c in Channel:
            noisy = true.of(c) + derive_rng(seed, "signal", step, c.value).gauss(
                0, p.signal_noise[c]
            )
            signal = signal.replace(c, round(noisy / p.signal_step[c]) * p.signal_step[c])
        return signal, waters[1:]
