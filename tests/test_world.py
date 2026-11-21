"""The grid world and the body: vocabulary, layout, one step, and snapshots."""

import pytest

from surprise_loop.world import (
    EFFECTS,
    WALKABLE,
    Belief,
    BodyParams,
    Channel,
    Direction,
    GridWorld,
    Kind,
    WorldConfig,
    WorldDone,
    derive_rng,
    frame,
    key,
    kind_at,
    layout_positions,
    neighbours,
    position_of,
    truth_at,
)

DEFAULT_LAYOUT = (
    "...#..~..",
    ".F.#...T.",
    ".....P...",
    "##..@..U.",
    "...T..F..",
    ".~...#..B",
    "....F#...",
)

STILL = BodyParams(hunger_rate=0, thirst_rate=0, ache_fade=0, sick_fade=0)


def body(**levels: float) -> dict[Channel, float]:
    return {channel: levels.get(channel.value, 0.0) for channel in Channel}


def world(layout: tuple[str, ...], seed: int = 0, **config: object) -> GridWorld:
    return GridWorld(WorldConfig(layout=layout, **config), seed)


# vocabulary


def test_effects_and_walkable() -> None:
    assert EFFECTS[Kind.TOXIC] == {Belief.BLOCKS, Belief.HURTS, Belief.FEEDS}
    assert frozenset({Kind.GROUND, Kind.PLEASANT, Kind.UNPLEASANT}) == WALKABLE


def test_key_and_position_of() -> None:
    assert key((2, -1)) == "2,-1"
    assert position_of("2,-1") == (2, -1)


def test_neighbours_in_compass_order() -> None:
    assert neighbours((0, 0)) == ((0, 1), (1, 0), (0, -1), (-1, 0))


# layout


SMALL = WorldConfig(layout=("#F.", ".@~", "P.U"))


def test_kind_at() -> None:
    assert kind_at(SMALL, (0, 1)) == Kind.FOOD
    assert kind_at(SMALL, (1, 0)) == Kind.WATER
    assert kind_at(SMALL, (-1, -1)) == Kind.PLEASANT
    assert kind_at(SMALL, (-1, 1)) == Kind.OBSTACLE
    assert kind_at(SMALL, (5, 5)) == Kind.OBSTACLE


def test_layout_positions_and_frame() -> None:
    positions = layout_positions(SMALL)
    assert len(positions) == 9
    assert positions[0] == (-1, 1)
    assert frame(SMALL) == (-2, -2, 2, 2)


def test_truth_at() -> None:
    truth = truth_at(SMALL, (0, 1))
    assert truth == {
        Belief.BLOCKS: 1.0,
        Belief.HURTS: 0.0,
        Belief.FEEDS: 1.0,
        Belief.QUENCHES: 0.0,
        Belief.PLEASES: 0.0,
    }


@pytest.mark.parametrize(
    ("layout", "place"),
    [
        ((), "row"),
        ((".@.", ".."), "row 2"),
        (("x@",), "row 1, column 1"),
        ((".", "."), "@"),
        (("@.@",), "row 1, column 3"),
        (("#@#",), "row 1, column 2"),
    ],
)
def test_layout_errors(layout: tuple[str, ...], place: str) -> None:
    with pytest.raises(ValueError, match=place) as caught:
        WorldConfig(layout=layout)
    assert "layout" in str(caught.value)


# moving


@pytest.mark.parametrize("ground", [".", "P", "U"])
def test_stepping_onto_ground_moves(ground: str) -> None:
    w = world((f".@{ground}",))
    outcome = w.step(Direction.EAST)
    assert outcome.signal.moved is True
    assert outcome.signal.position == (1, 0)


@pytest.mark.parametrize("solid", ["#", "F", "T", "~", "B"])
def test_stepping_into_solid_does_not_move(solid: str) -> None:
    w = world((f".@{solid}",))
    outcome = w.step(Direction.EAST)
    assert outcome.signal.moved is False
    assert outcome.signal.position == (0, 0)


def test_off_the_layout_is_an_obstacle() -> None:
    w = world(("@.",))
    outcome = w.step(Direction.WEST)
    assert outcome.signal.moved is False
    assert outcome.signal.position == (0, 0)
    assert outcome.info.target_kind == Kind.OBSTACLE


# eating and drinking


def test_food_eases_hunger() -> None:
    w = world((".@F",), body=STILL, start_body=body(hunger=0.5))
    signal = w.step(Direction.EAST).signal
    assert signal.body[Channel.HUNGER] == pytest.approx(0.2)
    assert signal.body[Channel.ACHE] == 0


def test_eating_when_full_aches() -> None:
    w = world((".@F",), body=STILL, start_body=body(hunger=0.1))
    signal = w.step(Direction.EAST).signal
    assert signal.body[Channel.HUNGER] == 0
    assert signal.body[Channel.ACHE] == pytest.approx(1.5 * 0.2)


def test_water_eases_thirst() -> None:
    w = world((".@~",), body=STILL, start_body=body(thirst=0.5))
    assert w.step(Direction.EAST).signal.body[Channel.THIRST] == pytest.approx(0.2)


def test_drinking_when_quenched_changes_nothing_else() -> None:
    w = world((".@~",), body=STILL, start_body=body(thirst=0.1))
    assert w.step(Direction.EAST).signal.body == body()


# drift and doses


def test_drift() -> None:
    w = world((".@.",), start_body=body(ache=0.4))
    levels = w.step(Direction.EAST).signal.body
    assert levels[Channel.HUNGER] == pytest.approx(0.012)
    assert levels[Channel.THIRST] == pytest.approx(0.018)
    assert levels[Channel.ACHE] == pytest.approx(0.3)


@pytest.mark.parametrize(("tile", "channel"), [("P", Channel.PLEASURE), ("U", Channel.UNEASE)])
def test_pleasant_and_unpleasant_ground(tile: str, channel: Channel) -> None:
    w = world((f".@{tile}.",), start_body=body())
    assert w.step(Direction.EAST).signal.body[channel] == pytest.approx(0.5)
    assert w.step(Direction.EAST).signal.body[channel] == pytest.approx(0.25)


def test_pushing_while_on_pleasant_ground_adds_no_pleasure() -> None:
    w = world(("@P#",), start_body=body())
    w.step(Direction.EAST)
    assert w.step(Direction.EAST).signal.body[Channel.PLEASURE] == pytest.approx(0.25)


# sickness


@pytest.mark.parametrize(("tile", "channel"), [("T", Channel.HUNGER), ("B", Channel.THIRST)])
def test_sickness_arrives_late(tile: str, channel: Channel) -> None:
    w = world((f"{tile}@.",), body=STILL, start_body=body(**{channel.value: 0.5}))
    first = w.step(Direction.WEST)
    assert first.signal.body[channel] == pytest.approx(0.2)
    (due,) = w.snapshot().pending_sickness
    assert 1 + 2 <= due <= 1 + 4
    assert first.signal.body[Channel.SICKNESS] == 0
    assert first.info.sickness_arrived is False
    for step in range(2, due + 2):
        outcome = w.step(Direction.EAST)
        arrived = step == due
        assert outcome.info.step == step
        assert outcome.info.sickness_arrived is arrived
        assert outcome.signal.body[Channel.SICKNESS] == (0.5 if step >= due else 0)
    assert w.snapshot().pending_sickness == ()


def delay(seed: int) -> int:
    w = world(("T@.",), seed=seed)
    w.step(Direction.WEST)
    (due,) = w.snapshot().pending_sickness
    return due - 1


def test_sickness_delays_vary_by_seed_and_repeat() -> None:
    assert {delay(seed) for seed in range(30)} == {2, 3, 4}
    assert all(delay(seed) == delay(seed) for seed in range(30))


# smell and damp


def test_smell_and_damp_count_neighbours() -> None:
    w = world(("#~.", "@.F", ".T."))
    signal = w.step(Direction.EAST).signal
    assert signal.smell == 2
    assert signal.damp == 1


def test_smell_chance_zero() -> None:
    for seed in range(10):
        w = world(("#~.", "@.F", ".T."), seed=seed, body=BodyParams(smell_chance=0))
        assert w.step(Direction.EAST).signal.smell == 0


def smell_at(seed: int) -> int:
    w = world(("@.F",), seed=seed, body=BodyParams(smell_chance=0.5))
    return w.step(Direction.EAST).signal.smell


def test_smell_chance_half() -> None:
    assert {smell_at(seed) for seed in range(40)} == {0, 1}
    assert all(smell_at(seed) == smell_at(seed) for seed in range(40))


# observing, ending, snapshots


def test_observe_before_any_step() -> None:
    start = body(hunger=0.3, thirst=0.1)
    signal = world((".@.",), start_body=start).observe()
    assert signal.moved is None
    assert signal.position == (0, 0)
    assert signal.body == start


def test_default_start_body() -> None:
    assert world((".@.",)).observe().body == body(hunger=0.2, thirst=0.2)


def test_done_after_steps() -> None:
    w = world((".@.",), steps=3)
    for _ in range(3):
        assert not w.done
        w.step(Direction.EAST)
    assert w.done
    with pytest.raises(WorldDone):
        w.step(Direction.EAST)


def test_snapshot_and_restore_continue_identically() -> None:
    config = WorldConfig(layout=("T@P", ".~F"))
    directions = [Direction.WEST, Direction.EAST, Direction.SOUTH, Direction.WEST]
    directions += [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST] * 3
    straight = GridWorld(config, 7)
    expected = [straight.step(d) for d in directions]

    first = GridWorld(config, 7)
    for d in directions[:2]:
        first.step(d)
    state = first.snapshot()
    assert state.pending_sickness
    resumed = GridWorld.restore(config, state)
    assert [resumed.step(d) for d in directions[2:]] == expected[2:]
    assert resumed.snapshot() == straight.snapshot()


def test_channels_stay_in_range() -> None:
    w = world(DEFAULT_LAYOUT, steps=500)
    rng = derive_rng(0, "walk")
    while not w.done:
        signal = w.step(rng.choice(list(Direction))).signal
        assert all(0 <= level <= 1 for level in signal.body.values())
        assert 0 <= signal.smell <= 4
        assert 0 <= signal.damp <= 4
