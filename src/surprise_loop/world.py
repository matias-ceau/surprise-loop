"""The grid world and the body.

The World is the only owner of environment state, the creature's true body included. A layout is
a few rows of characters; the creature starts on `@` and moves one tile per step. Solid tiles
block it, and pushing against food or water eats or drinks. `Signal` is all the creature ever
gets; `Kind`, `StepInfo` and the true numbers are for logging and scoring, never for a prompt.

Positions are `(east, north)` in steps from the starting tile.
"""

import random
from enum import StrEnum
from itertools import pairwise
from typing import Annotated, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Frozen(BaseModel):
    """The base of every state, config and result model: immutable, no unknown fields, no NaN."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


Unit = Annotated[float, Field(ge=0, le=1)]
Position = tuple[int, int]


class Kind(StrEnum):
    GROUND = "ground"
    PLEASANT = "pleasant"
    UNPLEASANT = "unpleasant"
    OBSTACLE = "obstacle"
    FOOD = "food"
    TOXIC = "toxic"
    WATER = "water"
    FOUL = "foul"


class Belief(StrEnum):
    BLOCKS = "blocks"
    HURTS = "hurts"
    FEEDS = "feeds"
    QUENCHES = "quenches"
    PLEASES = "pleases"


class Direction(StrEnum):
    NORTH = "north"
    EAST = "east"
    SOUTH = "south"
    WEST = "west"


class Channel(StrEnum):
    HUNGER = "hunger"
    THIRST = "thirst"
    ACHE = "ache"
    SICKNESS = "sickness"
    UNEASE = "unease"
    PLEASURE = "pleasure"


STEP: dict[Direction, Position] = {
    Direction.NORTH: (0, 1),
    Direction.EAST: (1, 0),
    Direction.SOUTH: (0, -1),
    Direction.WEST: (-1, 0),
}

START = "@"
LAYOUT_CHARS: dict[str, Kind] = {
    ".": Kind.GROUND,
    START: Kind.GROUND,
    "P": Kind.PLEASANT,
    "U": Kind.UNPLEASANT,
    "#": Kind.OBSTACLE,
    "F": Kind.FOOD,
    "T": Kind.TOXIC,
    "~": Kind.WATER,
    "B": Kind.FOUL,
}

WALKABLE: frozenset[Kind] = frozenset({Kind.GROUND, Kind.PLEASANT, Kind.UNPLEASANT})

EFFECTS: dict[Kind, frozenset[Belief]] = {
    Kind.GROUND: frozenset(),
    Kind.PLEASANT: frozenset({Belief.PLEASES}),
    Kind.UNPLEASANT: frozenset({Belief.HURTS}),
    Kind.OBSTACLE: frozenset({Belief.BLOCKS}),
    Kind.FOOD: frozenset({Belief.BLOCKS, Belief.FEEDS}),
    Kind.TOXIC: frozenset({Belief.BLOCKS, Belief.HURTS, Belief.FEEDS}),
    Kind.WATER: frozenset({Belief.BLOCKS, Belief.QUENCHES}),
    Kind.FOUL: frozenset({Belief.BLOCKS, Belief.HURTS, Belief.QUENCHES}),
}

EDIBLE = frozenset({Kind.FOOD, Kind.TOXIC})
DRINKABLE = frozenset({Kind.WATER, Kind.FOUL})
SICKENING = frozenset({Kind.TOXIC, Kind.FOUL})


def require_keys[K, V](value: dict[K, V], keys: type[StrEnum]) -> dict[K, V]:
    """Raise unless `value` has an entry for every member of `keys`."""
    missing = [k.value for k in keys if k not in value]
    if missing:
        raise ValueError(f"missing keys: {', '.join(missing)}")
    return value


def derive_rng(seed: int, *key: int | str) -> random.Random:
    """A generator keyed by seed and context; string seeding is stable across processes."""
    return random.Random(":".join(map(str, (seed, *key))))


def key(at: Position) -> str:
    return f"{at[0]},{at[1]}"


def position_of(key: str) -> Position:
    east, north = key.split(",")
    return int(east), int(north)


def moved_to(at: Position, direction: Direction) -> Position:
    east, north = STEP[direction]
    return at[0] + east, at[1] + north


def neighbours(at: Position) -> tuple[Position, ...]:
    return tuple(moved_to(at, direction) for direction in Direction)


class BodyParams(Frozen):
    hunger_rate: Unit = 0.012
    thirst_rate: Unit = 0.018
    food_portion: Unit = 0.30
    water_portion: Unit = 0.30
    ache_per_excess: Annotated[float, Field(ge=0)] = 1.5
    ache_fade: Unit = 0.25
    sick_dose: Unit = 0.5
    sick_fade: Unit = 0.08
    sick_delays: tuple[int, ...] = (2, 3, 4)
    unease_dose: Unit = 0.5
    unease_fade: Unit = 0.5
    pleasure_dose: Unit = 0.5
    pleasure_fade: Unit = 0.5
    # the chance that each neighbouring thing is noticed
    smell_chance: Unit = 1.0
    damp_chance: Unit = 1.0

    @field_validator("sick_delays")
    @classmethod
    def _ascending(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("needs at least one delay")
        if value[0] < 1:
            raise ValueError("each delay must be at least 1")
        if any(a >= b for a, b in pairwise(value)):
            raise ValueError("delays must be strictly ascending")
        return value


def _default_start_body() -> dict[Channel, float]:
    return {channel: 0.0 for channel in Channel} | {Channel.HUNGER: 0.2, Channel.THIRST: 0.2}


def _start(layout: tuple[str, ...]) -> tuple[int, int]:
    """The column and row of the start, both from 0."""
    for r, row in enumerate(layout):
        if START in row:
            return row.index(START), r
    raise ValueError(f"the layout has no start '{START}'")


def _kind_in(layout: tuple[str, ...], at: Position) -> Kind:
    c0, r0 = _start(layout)
    c, r = at[0] + c0, r0 - at[1]
    if 0 <= r < len(layout) and 0 <= c < len(layout[r]):
        return LAYOUT_CHARS[layout[r][c]]
    return Kind.OBSTACLE


class WorldConfig(Frozen):
    # rows, the northernmost first
    layout: tuple[str, ...]
    steps: int = Field(default=120, ge=1)
    body: BodyParams = BodyParams()
    start_body: dict[Channel, Unit] = Field(default_factory=_default_start_body)

    @field_validator("layout")
    @classmethod
    def _valid_layout(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("the layout needs at least one row")
        width = len(value[0])
        starts: list[tuple[int, int]] = []
        for r, row in enumerate(value, start=1):
            if len(row) != width:
                raise ValueError(f"row {r} has {len(row)} characters, row 1 has {width}")
            for c, char in enumerate(row, start=1):
                if char not in LAYOUT_CHARS:
                    raise ValueError(f"row {r}, column {c}: {char!r} is not a layout character")
                if char == START:
                    starts.append((r, c))
        if not starts:
            raise ValueError(f"the layout has no start '{START}'")
        if len(starts) > 1:
            r, c = starts[1]
            raise ValueError(f"row {r}, column {c}: a second start '{START}'")
        if not any(_kind_in(value, at) in WALKABLE for at in neighbours((0, 0))):
            r, c = starts[0]
            raise ValueError(f"row {r}, column {c}: the start has no walkable neighbour")
        return value

    @field_validator("start_body")
    @classmethod
    def _every_channel(cls, value: dict[Channel, float]) -> dict[Channel, float]:
        return require_keys(value, Channel)


def kind_at(config: WorldConfig, at: Position) -> Kind:
    """The kind of tile at `at`; every position outside the layout is an obstacle."""
    return _kind_in(config.layout, at)


def truth_at(config: WorldConfig, at: Position) -> dict[Belief, float]:
    effects = EFFECTS[kind_at(config, at)]
    return {belief: 1.0 if belief in effects else 0.0 for belief in Belief}


def layout_positions(config: WorldConfig) -> tuple[Position, ...]:
    """Every tile of the layout, row by row from the north-west."""
    c0, r0 = _start(config.layout)
    return tuple((c - c0, r0 - r) for r, row in enumerate(config.layout) for c in range(len(row)))


def frame(config: WorldConfig) -> tuple[int, int, int, int]:
    """West, south, east and north of the layout with its ring of obstacles."""
    c0, r0 = _start(config.layout)
    width, height = len(config.layout[0]), len(config.layout)
    return -c0 - 1, r0 - height, width - c0, r0 + 1


class Signal(Frozen):
    moved: bool | None
    body: dict[Channel, Unit]
    smell: Annotated[int, Field(ge=0, le=4)]
    damp: Annotated[int, Field(ge=0, le=4)]
    position: Position


class StepInfo(Frozen):
    step: int
    target: Position
    target_kind: Kind
    sickness_arrived: bool


class Outcome(Frozen):
    signal: Signal
    info: StepInfo


class WorldState(Frozen):
    seed: int
    step: int
    position: Position
    body: dict[Channel, Unit]
    pending_sickness: tuple[int, ...]


class WorldDone(RuntimeError):
    """A step was asked of a world whose day is over."""


class World(Protocol):
    @property
    def done(self) -> bool: ...

    def observe(self) -> Signal: ...

    def step(self, direction: Direction) -> Outcome: ...

    def snapshot(self) -> WorldState: ...


class GridWorld:
    """One creature on one layout. Holds nothing beyond what `WorldState` snapshots."""

    def __init__(self, config: WorldConfig, seed: int) -> None:
        self._config = config
        self._seed = seed
        self._step = 0
        self._position: Position = (0, 0)
        self._body: dict[Channel, float] = dict(config.start_body)
        self._pending: tuple[int, ...] = ()

    @classmethod
    def restore(cls, config: WorldConfig, state: WorldState) -> "GridWorld":
        world = cls(config, state.seed)
        world._step = state.step
        world._position = state.position
        world._body = dict(state.body)
        world._pending = state.pending_sickness
        return world

    @property
    def done(self) -> bool:
        return self._step >= self._config.steps

    def observe(self) -> Signal:
        return self._signal(None, 0)

    def snapshot(self) -> WorldState:
        return WorldState(
            seed=self._seed,
            step=self._step,
            position=self._position,
            body=dict(self._body),
            pending_sickness=self._pending,
        )

    def step(self, direction: Direction) -> Outcome:
        if self._step == self._config.steps:
            raise WorldDone(f"the world ends after {self._config.steps} steps")
        self._step += 1
        step = self._step
        target = moved_to(self._position, direction)
        kind = kind_at(self._config, target)
        p = self._config.body
        b = self._body

        b[Channel.HUNGER] += p.hunger_rate
        b[Channel.THIRST] += p.thirst_rate
        b[Channel.ACHE] *= 1 - p.ache_fade
        b[Channel.SICKNESS] *= 1 - p.sick_fade
        b[Channel.UNEASE] *= 1 - p.unease_fade
        b[Channel.PLEASURE] *= 1 - p.pleasure_fade

        moved = kind in WALKABLE
        if moved:
            self._position = target
            if kind == Kind.PLEASANT:
                b[Channel.PLEASURE] += p.pleasure_dose
            elif kind == Kind.UNPLEASANT:
                b[Channel.UNEASE] += p.unease_dose
        elif kind in EDIBLE:
            excess = max(0.0, p.food_portion - b[Channel.HUNGER])
            b[Channel.HUNGER] = max(0.0, b[Channel.HUNGER] - p.food_portion)
            b[Channel.ACHE] += p.ache_per_excess * excess
        elif kind in DRINKABLE:
            b[Channel.THIRST] = max(0.0, b[Channel.THIRST] - p.water_portion)
        if kind in SICKENING:
            delay = derive_rng(self._seed, "sick", step).choice(p.sick_delays)
            self._pending = (*self._pending, step + delay)

        due = [s for s in self._pending if s <= step]
        b[Channel.SICKNESS] += p.sick_dose * len(due)
        self._pending = tuple(s for s in self._pending if s > step)

        for channel in Channel:
            b[channel] = min(1.0, max(0.0, b[channel]))

        info = StepInfo(step=step, target=target, target_kind=kind, sickness_arrived=bool(due))
        return Outcome(signal=self._signal(moved, step), info=info)

    def _signal(self, moved: bool | None, step: int) -> Signal:
        """The body and position now, with what the nose and skin notice nearby."""
        p = self._config.body
        kinds = [kind_at(self._config, at) for at in neighbours(self._position)]
        rng = derive_rng(self._seed, "nose", step)
        smell = sum(rng.random() < p.smell_chance and kind in EDIBLE for kind in kinds)
        damp = sum(rng.random() < p.damp_chance and kind in DRINKABLE for kind in kinds)
        return Signal(
            moved=moved,
            body=dict(self._body),
            smell=smell,
            damp=damp,
            position=self._position,
        )
