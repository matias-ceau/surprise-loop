"""The careful reasoner: what a creature should believe of each tile, given what it felt."""

import math

import pytest
from creatures import run_config

from surprise_loop.config import LensParams, ReasonerParams
from surprise_loop.lens import sense
from surprise_loop.reasoner import (
    initial_reasoner,
    marginals,
    reason,
    reasoner_beliefs,
    shares_at,
)
from surprise_loop.state import FeltRecord, ReasonerState, Touch
from surprise_loop.world import (
    WALKABLE,
    Belief,
    BodyParams,
    Channel,
    Direction,
    GridWorld,
    Kind,
    Position,
    Signal,
    derive_rng,
    key,
)

EQUAL = dict.fromkeys(Kind, 1 / len(Kind))
QUIET = BodyParams(smell_chance=0, damp_chance=0)
P = ReasonerParams()
S = P.slip

# for the smell and damp tests: no tile is presumed to be food, toxic, water or foul
PLAIN = dict.fromkeys(Kind, 0.0) | {Kind.GROUND: 0.5, Kind.OBSTACLE: 0.5}
A: Position = (5, 5)
B: Position = (6, 5)


def felt(
    at: Position = (0, 0), moved: bool | None = None, smell: int = 0, damp: int = 0, **levels: int
) -> FeltRecord:
    return FeltRecord(
        moved=moved,
        levels=dict.fromkeys(Channel, 0) | {Channel(c): v for c, v in levels.items()},
        smell=smell,
        damp=damp,
        tone=0,
        stir=0,
        at=at,
    )


def shares(**values: float) -> dict[Kind, float]:
    return dict.fromkeys(Kind, 0.0) | {Kind(k): v for k, v in values.items()}


def state(
    presumes: dict[Kind, float] = EQUAL,
    tiles: dict[Position, dict[Kind, float]] | None = None,
    touched: tuple[Touch, ...] = (),
    onsets: tuple[int, ...] = (),
) -> ReasonerState:
    return ReasonerState(
        presumes=presumes,
        tiles={key(at): s for at, s in (tiles or {}).items()},
        touched=touched,
        onsets=onsets,
    )


def moved_north(
    s: ReasonerState,
    step: int = 1,
    body: BodyParams = QUIET,
    p: ReasonerParams = P,
    **after: int,
) -> tuple[ReasonerState, tuple[Position, ...]]:
    """A step north from (0, -1) onto (0, 0)."""
    return reason(
        s,
        step,
        Direction.NORTH,
        felt(at=(0, -1)),
        felt(at=(0, 0), moved=True, **after),
        body,
        p,
    )


def blocked_east(
    s: ReasonerState, before: dict[str, int], step: int = 1, **after: int
) -> ReasonerState:
    """A step east from (0, 0), blocked by (1, 0)."""
    new, _ = reason(
        s,
        step,
        Direction.EAST,
        felt(**before),
        felt(moved=False, **(before | after)),
        QUIET,
        P,
    )
    return new


def harmful(s: dict[Kind, float]) -> float:
    return s[Kind.TOXIC] + s[Kind.FOUL]


# marginals


def test_marginals_of_equal_shares() -> None:
    m = marginals(EQUAL)
    assert m[Belief.BLOCKS] == pytest.approx(5 / 8)
    assert m[Belief.HURTS] == pytest.approx(3 / 8)
    assert m[Belief.FEEDS] == pytest.approx(2 / 8)
    assert m[Belief.QUENCHES] == pytest.approx(2 / 8)
    assert m[Belief.PLEASES] == pytest.approx(1 / 8)


def test_marginals_of_certain_toxic() -> None:
    m = marginals(shares(toxic=1.0))
    assert m == {
        Belief.BLOCKS: 1.0,
        Belief.HURTS: 1.0,
        Belief.FEEDS: 1.0,
        Belief.QUENCHES: 0.0,
        Belief.PLEASES: 0.0,
    }


def test_an_untouched_tile_is_what_is_presumed() -> None:
    s = initial_reasoner(EQUAL)
    assert s.tiles == {}
    assert s.touched == ()
    assert s.onsets == ()
    assert shares_at(s, (3, -2)) == EQUAL
    assert reasoner_beliefs(s, (3, -2)) == marginals(EQUAL)


# steps 1 and 2: the target


def test_moved_with_nothing_felt() -> None:
    new, _ = moved_north(initial_reasoner(EQUAL))
    t = shares_at(new, (0, 0))
    for kind in Kind:
        if kind not in WALKABLE:
            assert t[kind] == 0
    assert reasoner_beliefs(new, (0, 0))[Belief.BLOCKS] == 0
    assert t[Kind.GROUND] == pytest.approx(0.95 * 0.95 / (0.95 * 0.95 + 2 * 0.05 * 0.95))


def test_moved_and_pleasure_rose() -> None:
    new, _ = moved_north(initial_reasoner(EQUAL), pleasure=1)
    t = shares_at(new, (0, 0))
    assert t[Kind.PLEASANT] == pytest.approx(0.9025 / 0.9525)
    assert t[Kind.GROUND] == pytest.approx(0.0475 / 0.9525)
    assert t[Kind.UNPLEASANT] == pytest.approx(0.0025 / 0.9525)


def test_blocked_hungry_and_thirsty_nothing_felt() -> None:
    new = blocked_east(initial_reasoner(EQUAL), {"hunger": 2, "thirst": 2})
    t = shares_at(new, (1, 0))
    assert t[Kind.OBSTACLE] == pytest.approx(0.95 / 1.15)
    for kind in (Kind.FOOD, Kind.TOXIC, Kind.WATER, Kind.FOUL):
        assert t[kind] == pytest.approx(0.05 / 1.15)
    for kind in WALKABLE:
        assert t[kind] == 0
    assert reasoner_beliefs(new, (1, 0))[Belief.BLOCKS] == pytest.approx(1)


def test_blocked_hungry_not_thirsty_nothing_felt() -> None:
    new = blocked_east(initial_reasoner(EQUAL), {"hunger": 2})
    t = shares_at(new, (1, 0))
    for kind in (Kind.OBSTACLE, Kind.WATER, Kind.FOUL):
        assert t[kind] == pytest.approx(0.95 / 2.95)
    assert reasoner_beliefs(new, (1, 0))[Belief.QUENCHES] == pytest.approx(1.9 / 2.95)


def test_blocked_hungry_and_hunger_fell() -> None:
    new = blocked_east(initial_reasoner(EQUAL), {"hunger": 2}, hunger=1)
    t = shares_at(new, (1, 0))
    assert t[Kind.FOOD] == pytest.approx(t[Kind.TOXIC])
    assert reasoner_beliefs(new, (1, 0))[Belief.FEEDS] == pytest.approx(1.9 / 2.05)


def test_blocked_full_and_ache_rose() -> None:
    new = blocked_east(initial_reasoner(EQUAL), {"hunger": 0}, ache=1)
    assert reasoner_beliefs(new, (1, 0))[Belief.FEEDS] == pytest.approx(1.9 / 2.05)


def test_blocked_slightly_hungry_uses_ache_when_slight() -> None:
    new = blocked_east(initial_reasoner(EQUAL), {"hunger": 1})
    # each weight: P(fed) as observed, P(quenched) as observed, P(ached) as observed
    food = S * (1 - S) * (1 - P.ache_when_slight)
    obstacle = (1 - S) * (1 - S) * (1 - S)
    water = (1 - S) * (1 - S) * (1 - S)
    total = 2 * food + obstacle + 2 * water
    t = shares_at(new, (1, 0))
    assert t[Kind.FOOD] == pytest.approx(food / total)
    assert t[Kind.TOXIC] == pytest.approx(food / total)
    assert t[Kind.OBSTACLE] == pytest.approx(obstacle / total)


def test_no_walkable_share_left_gives_equal_walkable_shares() -> None:
    s = state(presumes=shares(obstacle=0.5, food=0.5))
    # with slip one half every event is as likely either way, so step 2 leaves step 1's shares
    new, _ = moved_north(s, p=ReasonerParams(slip=0.5))
    t = shares_at(new, (0, 0))
    for kind in Kind:
        assert t[kind] == pytest.approx(1 / 3 if kind in WALKABLE else 0)
    # with the default slip, step 2 then weighs the three as for any step that moved
    new, _ = moved_north(s)
    assert shares_at(new, (0, 0))[Kind.GROUND] == pytest.approx(
        0.95 * 0.95 / (0.95 * 0.95 + 2 * 0.05 * 0.95)
    )


# steps 3 to 5: touches, sickness and clearing


def test_sickness_is_blamed_on_the_touches_in_the_delays() -> None:
    s = state(
        tiles={
            A: shares(toxic=0.3, foul=0.2, obstacle=0.5),
            B: shares(toxic=0.05, foul=0.2, ground=0.75),
        },
        touched=(Touch(step=10, at=A), Touch(step=11, at=B)),
    )
    new, _ = moved_north(s, step=13, sickness=1)
    a, b = shares_at(new, A), shares_at(new, B)
    assert harmful(a) == pytest.approx(2 / 3 + 1 / 3 * 0.5)
    assert harmful(b) == pytest.approx(1 / 3 + 2 / 3 * 0.25)
    assert a[Kind.TOXIC] / a[Kind.FOUL] == pytest.approx(0.3 / 0.2)
    assert b[Kind.TOXIC] / b[Kind.FOUL] == pytest.approx(0.05 / 0.2)
    assert math.fsum(a.values()) == pytest.approx(1)
    assert math.fsum(b.values()) == pytest.approx(1)
    assert new.onsets == (13,)


def test_sickness_with_no_candidate_changes_nothing() -> None:
    before = shares(toxic=0.3, foul=0.2, obstacle=0.5)
    s = state(tiles={A: before}, touched=(Touch(step=12, at=A),))
    new, changed = moved_north(s, step=13, sickness=1)
    assert shares_at(new, A) == before
    assert changed == ((0, 0),)
    assert new.onsets == (13,)


def test_a_touch_with_no_sickness_in_its_window_is_cleared() -> None:
    s = state(tiles={A: shares(food=0.5, toxic=0.5)}, touched=(Touch(step=10, at=A),))
    new, _ = moved_north(s, step=14)
    assert shares_at(new, A)[Kind.TOXIC] == pytest.approx(0.025 / 0.525)
    assert new.touched == (Touch(step=10, at=A),)
    later, _ = moved_north(new, step=15)
    assert later.touched == ()


@pytest.mark.parametrize("onsets", [(12,), (13,)])
def test_a_touch_with_an_onset_in_its_window_is_not_cleared(onsets: tuple[int, ...]) -> None:
    s = state(
        tiles={A: shares(food=0.5, toxic=0.5)}, touched=(Touch(step=10, at=A),), onsets=onsets
    )
    new, _ = moved_north(s, step=14)
    assert shares_at(new, A)[Kind.TOXIC] == pytest.approx(0.5)


def test_an_onset_in_this_step_saves_its_tile_from_clearing() -> None:
    s = state(tiles={A: shares(food=0.5, toxic=0.5)}, touched=(Touch(step=10, at=A),))
    new, _ = moved_north(s, step=14, sickness=1)
    # blamed on A alone: r is 1, so its harmful share becomes 1, and it is not cleared after
    assert shares_at(new, A)[Kind.TOXIC] == pytest.approx(1)


def test_touched_and_onsets_never_hold_anything_older_than_the_window() -> None:
    window = max(QUIET.sick_delays)
    s = initial_reasoner(EQUAL)
    sickness = 0
    for step in range(1, 30):
        rises = step % 3 == 0
        after = sickness + 1 if rises else sickness
        s = blocked_east(s, {"sickness": sickness}, step=step, sickness=min(after, 4))
        sickness = 0 if after >= 4 else after
        assert s.touched
        assert all(t.step >= step - window for t in s.touched)
        assert all(o >= step - window for o in s.onsets)
        assert [t.step for t in s.touched] == sorted(t.step for t in s.touched)
    assert s.onsets


# step 6: smell and damp


def smelt(
    tiles: dict[Position, dict[Kind, float]], smell: int = 0, damp: int = 0, **body: float
) -> tuple[ReasonerState, tuple[Position, ...]]:
    return moved_north(
        state(presumes=PLAIN, tiles=tiles),
        body=BodyParams(**({"smell_chance": 0, "damp_chance": 0} | body)),
        smell=smell,
        damp=damp,
    )


def test_no_smell_when_every_smell_is_noticed_rules_food_out() -> None:
    new, _ = smelt(
        {(0, 1): shares(food=0.5, ground=0.5), (1, 0): shares(toxic=0.5, obstacle=0.5)},
        smell_chance=1,
    )
    assert shares_at(new, (0, 1)) == pytest.approx(shares(ground=1))
    assert shares_at(new, (1, 0)) == pytest.approx(shares(obstacle=1))


def test_one_smell_when_every_smell_is_noticed() -> None:
    new, _ = smelt(
        {(0, 1): shares(food=0.2, ground=0.8), (1, 0): shares(toxic=0.6, obstacle=0.4)},
        smell=1,
        smell_chance=1,
    )
    north, east = shares_at(new, (0, 1)), shares_at(new, (1, 0))
    assert north[Kind.FOOD] == pytest.approx(0.08 / 0.56)
    assert north[Kind.GROUND] == pytest.approx(0.48 / 0.56)
    assert east[Kind.TOXIC] == pytest.approx(0.48 / 0.56)
    assert east[Kind.OBSTACLE] == pytest.approx(0.08 / 0.56)


def test_two_smells_when_every_smell_is_noticed() -> None:
    new, _ = smelt(
        {(0, 1): shares(food=0.2, ground=0.8), (1, 0): shares(toxic=0.6, obstacle=0.4)},
        smell=2,
        smell_chance=1,
    )
    assert shares_at(new, (0, 1)) == pytest.approx(shares(food=1))
    assert shares_at(new, (1, 0)) == pytest.approx(shares(toxic=1))


def test_a_smell_no_neighbour_can_hold_changes_nothing() -> None:
    new, changed = smelt({}, smell=1, smell_chance=1)
    assert changed == ((0, 0),)
    assert set(new.tiles) == {key((0, 0))}


def test_no_smell_when_half_of_smells_are_noticed() -> None:
    new, _ = smelt(
        {(0, 1): shares(food=0.5, ground=0.5), (1, 0): shares(food=0.5, obstacle=0.5)},
        smell_chance=0.5,
    )
    assert shares_at(new, (0, 1))[Kind.FOOD] == pytest.approx(1 / 3)
    assert shares_at(new, (1, 0))[Kind.FOOD] == pytest.approx(1 / 3)


def test_damp_works_on_water_and_foul_and_leaves_food_alone() -> None:
    new, _ = smelt(
        {
            (0, 1): shares(water=0.2, food=0.3, ground=0.5),
            (1, 0): shares(foul=0.6, obstacle=0.4),
            (-1, 0): shares(food=0.5, ground=0.5),
        },
        damp=1,
        damp_chance=1,
    )
    north, east = shares_at(new, (0, 1)), shares_at(new, (1, 0))
    assert north[Kind.WATER] == pytest.approx(0.08 / 0.56)
    assert north[Kind.FOOD] / north[Kind.GROUND] == pytest.approx(0.3 / 0.5)
    assert east[Kind.FOUL] == pytest.approx(0.48 / 0.56)
    assert shares_at(new, (-1, 0)) == pytest.approx(shares(food=0.5, ground=0.5))
    assert reasoner_beliefs(new, (-1, 0))[Belief.FEEDS] == pytest.approx(0.5)


def test_damp_with_every_damp_noticed_rules_water_out() -> None:
    new, _ = smelt({(0, 1): shares(water=0.5, foul=0.25, ground=0.25)}, damp_chance=1)
    assert shares_at(new, (0, 1)) == pytest.approx(shares(ground=1))


# the whole


def test_the_returned_positions_are_those_that_moved_and_the_input_is_unchanged() -> None:
    s = state(
        presumes=PLAIN,
        tiles={(0, 1): shares(food=0.5, ground=0.5), (1, 0): shares(toxic=0.5, obstacle=0.5)},
    )
    dumped = s.model_dump()
    new, changed = moved_north(s, body=BodyParams(smell_chance=1, damp_chance=0))
    assert s.model_dump() == dumped
    assert changed == ((0, 0), (0, 1), (1, 0))
    for at in [(0, -1), (-1, 0), (0, 2), A]:
        assert reasoner_beliefs(new, at) == reasoner_beliefs(s, at)


def test_shares_sum_to_one_over_300_random_steps() -> None:
    config = run_config()
    world_config = config.world.model_copy(update={"steps": 300})
    world = GridWorld(world_config, seed=3)
    temperament = config.agent.temperament
    rng = derive_rng(3, "test-moves")
    lens_rng = derive_rng(3, "test-lens")

    def record(signal: Signal) -> FeltRecord:
        levels, smell, damp = sense(signal, temperament, lens_rng, LensParams())
        return FeltRecord(
            moved=signal.moved,
            levels=levels,
            smell=smell,
            damp=damp,
            tone=2,
            stir=1,
            at=signal.position,
        )

    s = initial_reasoner(temperament.presumes)
    before = record(world.observe())
    directions = list(Direction)
    for step in range(1, 301):
        direction = rng.choice(directions)
        after = record(world.step(direction).signal)
        s, _ = reason(s, step, direction, before, after, world_config.body, config.params.reasoner)
        before = after
        for tile in s.tiles.values():
            assert abs(math.fsum(tile.values()) - 1) <= 1e-9
    assert world.done
    assert s.tiles
