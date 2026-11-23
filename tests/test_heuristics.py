"""The rule chooser, its expectation, the blind chooser, the measures and the starting state."""

import math

import pytest
from creatures import run_config

from surprise_loop.config import RuleParams, initial_agent_state
from surprise_loop.phases.heuristics import blind_decision, measures, rule_choice, rule_expectation
from surprise_loop.reasoner import initial_reasoner, marginals, reasoner_beliefs
from surprise_loop.state import AgentState, CreatureMap, FeltRecord, Sureness
from surprise_loop.world import (
    Belief,
    Channel,
    Direction,
    GridWorld,
    Kind,
    Position,
    derive_rng,
    key,
    layout_positions,
    truth_at,
)

P = RuleParams()
ALL = tuple(Direction)


def felt(at: Position = (0, 0), **levels: int) -> FeltRecord:
    return FeltRecord(
        moved=None,
        levels=dict.fromkeys(Channel, 0) | {Channel(c): v for c, v in levels.items()},
        smell=0,
        damp=0,
        tone=0,
        stir=0,
        at=at,
    )


def beliefs(**values: float) -> dict[Belief, float]:
    return dict.fromkeys(Belief, 0.0) | {Belief(b): v for b, v in values.items()}


def untried(agent: str) -> dict[Belief, float]:
    return marginals(run_config(agent).agent.temperament.presumes)


def a_map(agent: str = "plain", **tiles: dict[Belief, float]) -> CreatureMap:
    """A map with `agent`'s untried beliefs; tile keywords are written like `t3_0` for (3, 0)."""
    return CreatureMap(untried=untried(agent), tiles={_key(name): b for name, b in tiles.items()})


def _key(name: str) -> str:
    x, y = name[1:].replace("m", "-").split("_")
    return f"{x},{y}"


def choose(f: FeltRecord, m: CreatureMap, tries: dict[str, int], seed: int = 1):
    return rule_choice(f, m, tries, P, derive_rng(seed, "rule", 1))


# the starting state


def test_initial_agent_state_for_the_plain_creature() -> None:
    config = run_config("plain")
    agent = initial_agent_state(config, GridWorld(config.world, config.seed).observe())
    assert agent.felt.at == (0, 0)
    assert agent.felt.moved is None
    assert agent.felt.levels[Channel.HUNGER] == 1
    assert agent.felt.levels[Channel.THIRST] == 1
    assert agent.felt.stir == 0
    assert agent.stir == 0.0
    assert agent.own_map.untried == marginals(config.agent.temperament.presumes)
    assert math.isclose(agent.own_map.untried[Belief.BLOCKS], 0.29)
    assert agent.reasoner == initial_reasoner(config.agent.temperament.presumes)
    assert agent.own_map.tiles == {"0,0": reasoner_beliefs(agent.reasoner, (0, 0))}
    assert agent.own_map.at((0, 0))[Belief.BLOCKS] == 0
    assert reasoner_beliefs(agent.reasoner, (0, 0))[Belief.BLOCKS] == 0
    assert agent.tries == {}
    assert agent.memory.episodes == []
    assert agent.tick == 0


# rule_choice

FOOD = beliefs(blocks=1.0, feeds=0.9)


def test_a_hungry_creature_heads_for_tried_food() -> None:
    rc = choose(felt(hunger=3), a_map(t3_0=FOOD), {"3,0": 1})
    assert rc.best == (Direction.EAST,)
    assert rc.choice == Direction.EAST


def test_food_believed_to_hurt_is_no_goal_until_the_need_is_desperate() -> None:
    m = a_map(t3_0=beliefs(blocks=1.0, feeds=0.9, hurts=0.6))
    assert choose(felt(hunger=2), m, {"3,0": 1}).best == ALL
    assert choose(felt(hunger=3), m, {"3,0": 1}).best == (Direction.EAST,)


WATER = beliefs(blocks=1.0, quenches=0.9)


def test_believed_water_serves_a_need_before_it_is_tried() -> None:
    assert choose(felt(thirst=2), a_map(t2_0=WATER), {}).best == (Direction.EAST,)


def test_believed_water_that_may_hurt_waits_while_the_need_is_not_desperate() -> None:
    m = a_map(t2_0=beliefs(blocks=1.0, quenches=1.0, hurts=0.75))
    assert choose(felt(thirst=2), m, {}).best == ALL


def test_a_desperate_need_heads_for_believed_water_that_may_hurt() -> None:
    m = a_map(t2_0=beliefs(blocks=1.0, quenches=1.0, hurts=0.75))
    assert choose(felt(thirst=3), m, {}).best == (Direction.EAST,)


def test_a_curious_creature_tries_the_new_before_the_known() -> None:
    tries = {"-1,0": 1}
    curious = choose(felt(), a_map("plain", tm1_0=beliefs()), tries)
    assert curious.best == (Direction.NORTH, Direction.EAST, Direction.SOUTH)
    assert choose(felt(), a_map("wary", tm1_0=beliefs()), tries).best == (Direction.WEST,)


def test_no_need_and_nothing_tried_explores_every_way() -> None:
    a = choose(felt(), a_map(), {}, seed=7)
    b = choose(felt(), a_map(), {}, seed=7)
    assert a.best == ALL
    assert a.choice in ALL
    assert a == b


def test_a_wary_creature_with_no_need_goes_back_to_what_it_knows() -> None:
    rc = choose(felt(), a_map("wary", tm1_0=beliefs()), {"-1,0": 1})
    assert rc.best == (Direction.WEST,)


def test_a_wary_creature_with_nothing_tried_falls_back_to_untried_tiles() -> None:
    assert untried("wary")[Belief.HURTS] < P.avoid_at
    assert choose(felt(), a_map("wary"), {}).best == ALL


def test_a_believed_block_is_walked_round() -> None:
    m = a_map(t1_0=beliefs(blocks=0.9), t2_0=FOOD)
    rc = choose(felt(hunger=3), m, {"2,0": 1})
    assert rc.best == (Direction.NORTH, Direction.SOUTH)


def test_hunger_wins_a_tie_and_thirst_wins_when_louder() -> None:
    water = beliefs(blocks=1.0, quenches=0.9)
    m = a_map(t2_0=FOOD, tm2_0=water)
    tries = {"2,0": 1, "-2,0": 1}
    assert choose(felt(hunger=3, thirst=3), m, tries).best == (Direction.EAST,)
    assert choose(felt(hunger=2, thirst=3), m, tries).best == (Direction.WEST,)


def test_a_curious_creature_walks_past_an_untried_tile_it_believes_harmful() -> None:
    tried_ground = {"t0_1": beliefs(), "t0_m1": beliefs(), "tm1_0": beliefs()}
    m = a_map(t1_0=beliefs(hurts=0.9), **tried_ground)
    rc = choose(felt(), m, {"0,1": 1, "0,-1": 1, "-1,0": 1})
    assert rc.best == (Direction.NORTH, Direction.SOUTH, Direction.WEST)


def test_a_loud_need_with_nothing_safe_to_try_heads_for_harm() -> None:
    m = CreatureMap(untried=beliefs(hurts=0.9), tiles={"-1,0": beliefs()})
    rc = choose(felt(hunger=3), m, {"-1,0": 1})
    assert rc.best == (Direction.NORTH, Direction.EAST, Direction.SOUTH)


@pytest.mark.parametrize(
    "untried_beliefs",
    [
        beliefs(hurts=0.6, feeds=0.3, quenches=0.3, pleases=0.3),  # curious
        beliefs(hurts=0.9),  # not curious
    ],
)
def test_with_no_need_and_only_harm_untried_it_goes_back_to_what_it_knows(
    untried_beliefs: dict[Belief, float],
) -> None:
    m = CreatureMap(untried=untried_beliefs, tiles={"-1,0": beliefs()})
    assert choose(felt(), m, {"-1,0": 1}).best == (Direction.WEST,)


@pytest.mark.parametrize(
    "untried_beliefs",
    [
        beliefs(hurts=0.6, feeds=0.3, quenches=0.3, pleases=0.3),  # curious
        beliefs(hurts=0.9),  # not curious
    ],
)
def test_with_no_need_it_stands_on_known_ground_and_does_not_push_at_things(
    untried_beliefs: dict[Belief, float],
) -> None:
    food = beliefs(blocks=1.0, feeds=0.9)
    m = CreatureMap(untried=untried_beliefs, tiles={"1,0": food, "0,1": beliefs()})
    # the tried, safe food east is no goal; the tried, safe ground north is
    assert choose(felt(), m, {"1,0": 3, "0,1": 1}).best == (Direction.NORTH,)


def test_with_no_need_and_only_a_tried_obstacle_known_it_explores() -> None:
    m = a_map("wary", t1_0=beliefs(blocks=1.0))
    assert choose(felt(), m, {"1,0": 2}).best == (Direction.NORTH, Direction.SOUTH, Direction.WEST)


# rule_expectation


def test_expecting_to_eat() -> None:
    m = a_map(t1_0=beliefs(blocks=1.0, feeds=0.9))
    e = rule_expectation(Direction.EAST, felt(hunger=2), m, {"1,0": 2}, P)
    assert not e.moved
    assert e.levels[Channel.HUNGER] == 1
    assert e.levels[Channel.ACHE] == 0
    assert e.sure == Sureness.SURE
    assert rule_expectation(Direction.EAST, felt(hunger=2), m, {"1,0": 1}, P).sure == (
        Sureness.FAIRLY_SURE
    )
    assert rule_expectation(Direction.EAST, felt(hunger=2), m, {}, P).sure == Sureness.GUESSING


def test_expecting_to_eat_when_full_expects_an_ache() -> None:
    m = a_map(t1_0=beliefs(blocks=1.0, feeds=0.9))
    e = rule_expectation(Direction.EAST, felt(hunger=0, ache=1), m, {}, P)
    assert e.levels[Channel.HUNGER] == 0
    assert e.levels[Channel.ACHE] == 2
    top = rule_expectation(Direction.EAST, felt(ache=4), m, {}, P)
    assert top.levels[Channel.ACHE] == 4


def test_expecting_to_drink() -> None:
    m = a_map(t1_0=beliefs(blocks=1.0, quenches=0.9))
    e = rule_expectation(Direction.EAST, felt(thirst=2), m, {}, P)
    assert not e.moved
    assert e.levels[Channel.THIRST] == 1


def test_expecting_pleasant_ground_and_the_smell_beside_it() -> None:
    m = a_map(t1_0=beliefs(pleases=0.9), t2_0=FOOD, t1_1=FOOD, t1_m1=FOOD)
    e = rule_expectation(Direction.EAST, felt(pleasure=1), m, {}, P)
    assert e.moved
    assert e.levels[Channel.PLEASURE] == 2
    assert e.smell == 2  # three believed food beside the new spot, at most two on the ladder
    assert e.damp == 0
    high = rule_expectation(Direction.EAST, felt(pleasure=3), m, {}, P)
    assert high.levels[Channel.PLEASURE] == 3


def test_expecting_ground_believed_to_hurt() -> None:
    m = a_map(t1_0=beliefs(hurts=0.9))
    e = rule_expectation(Direction.EAST, felt(), m, {}, P)
    assert e.moved
    assert e.levels[Channel.UNEASE] == 2


# blind_decision


def test_the_blind_chooser_reads_only_the_felt_record() -> None:
    f = felt(hunger=2, thirst=1)
    d = blind_decision(f, derive_rng(3, "blind", 1))
    assert d == blind_decision(f, derive_rng(3, "blind", 1))
    assert d.expectation.moved
    assert d.expectation.levels == f.levels
    assert (d.expectation.smell, d.expectation.damp) == (f.smell, f.damp)
    assert d.expectation.sure == Sureness.GUESSING
    assert d.why == ""


# measures


def _agent(own_map: CreatureMap, tries: dict[str, int]) -> AgentState:
    config = run_config("plain")
    agent = initial_agent_state(config, GridWorld(config.world, config.seed).observe())
    return agent.model_copy(update={"own_map": own_map, "tries": tries})


def _error(a, b, tiles) -> float:
    if not tiles:
        return 0.0
    return sum(abs(a(t)[x] - b(t)[x]) for t in tiles for x in Belief) / (len(tiles) * len(Belief))


def test_measures_on_two_tried_tiles() -> None:
    world = run_config("plain").world
    harmful, fine = (1, 0), (-1, 0)  # both truly ground
    own = a_map(t1_0=beliefs(hurts=0.9), tm1_0=beliefs())
    agent = _agent(own, {key(harmful): 2, key(fine): 1, "0,1": 0})
    m = measures(agent, world, run_config("plain").params.metrics)

    tried = [harmful, fine]
    everywhere = list(layout_positions(world))

    def truth(t: Position) -> dict[Belief, float]:
        return truth_at(world, t)

    def reasoner(t: Position) -> dict[Belief, float]:
        return reasoner_beliefs(agent.reasoner, t)

    assert m.tiles_tried == 2
    assert m.false_harm == 1
    assert m.false_safe == 0
    assert m.reasoner_false_harm == 0
    assert m.reasoner_false_safe == 0
    assert m.wrong == (key(harmful),)
    assert math.isclose(m.own_error, 0.9 / 10)
    assert math.isclose(m.start_error, _error(lambda t: own.untried, truth, tried))
    assert math.isclose(m.reasoner_error, _error(reasoner, truth, tried))
    assert math.isclose(m.gap, _error(own.at, reasoner, tried))
    assert math.isclose(m.own_error_all, _error(own.at, truth, everywhere))
    assert math.isclose(m.reasoner_error_all, _error(reasoner, truth, everywhere))
    assert math.isclose(m.gap_all, _error(own.at, reasoner, everywhere))


def pinned_agent() -> AgentState:
    """Three tried tiles (ground, food, toxic) and an untried one, in both maps."""
    own = a_map(
        t1_0=beliefs(hurts=0.9),
        t2_m1=beliefs(blocks=1.0, feeds=0.7, hurts=0.3),
        tm1_m1=beliefs(blocks=1.0, feeds=0.8),
        t0_1=beliefs(pleases=0.6),
    )
    agent = _agent(own, {"1,0": 1, "2,-1": 3, "-1,-1": 1})
    reasoner = agent.reasoner.model_copy(
        update={
            "tiles": {
                "2,-1": dict.fromkeys(Kind, 0.0) | {Kind.FOOD: 0.75, Kind.TOXIC: 0.25},
                "-1,-1": dict.fromkeys(Kind, 0.0) | {Kind.FOOD: 0.5, Kind.TOXIC: 0.5},
                "0,1": dict.fromkeys(Kind, 0.0) | {Kind.GROUND: 0.6, Kind.PLEASANT: 0.4},
            }
        }
    )
    return agent.model_copy(update={"reasoner": reasoner})


# measures of pinned_agent, recorded before measures was rewritten to read each tile once
PINNED = {
    "tiles_tried": 3,
    "start_error": 0.3426666666666666,
    "own_error": 0.18000000000000002,
    "reasoner_error": 0.09133333333333332,
    "gap": 0.158,
    "own_error_all": 0.16882539682539682,
    "reasoner_error_all": 0.16396825396825396,
    "gap_all": 0.008158730158730159,
    "false_harm": 1,
    "false_safe": 1,
    "reasoner_false_harm": 0,
    "reasoner_false_safe": 0,
    "wrong": ("-1,-1", "1,0"),
}


def test_measures_pinned() -> None:
    config = run_config("plain")
    m = measures(pinned_agent(), config.world, config.params.metrics)
    assert m.model_dump() == PINNED


def test_measures_with_nothing_tried() -> None:
    config = run_config("plain")
    agent = _agent(a_map(), {})
    m = measures(agent, config.world, config.params.metrics)
    assert m.tiles_tried == 0
    assert (m.start_error, m.own_error, m.reasoner_error, m.gap) == (0.0, 0.0, 0.0, 0.0)
    assert (m.false_harm, m.false_safe, m.reasoner_false_harm, m.reasoner_false_safe) == (0,) * 4
    assert m.wrong == ()
    assert m.own_error_all > 0
