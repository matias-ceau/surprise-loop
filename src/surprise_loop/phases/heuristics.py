"""The rules the stand-ins and the code phases share: the rule chooser and what it expects, the
blind chooser, and the measures of a creature's map against the truth.

The choosers read only what the creature has: its felt record, its own map and its tries.
`measures` is the one place on the creature's side that reads the truth, and nothing it returns
is ever read by a chooser.
"""

import math
import random
from collections.abc import Callable, Sequence

from surprise_loop.config import MetricsParams, RuleParams
from surprise_loop.feeling import TOP_STEP
from surprise_loop.lens import near_step
from surprise_loop.reasoner import reasoner_beliefs
from surprise_loop.state import (
    AgentState,
    CreatureMap,
    Decision,
    Expectation,
    FeltRecord,
    Measures,
    RuleChoice,
    Sureness,
)
from surprise_loop.world import (
    Belief,
    Channel,
    Direction,
    Position,
    WorldConfig,
    key,
    layout_positions,
    moved_to,
    neighbours,
    position_of,
    truth_at,
)

# what a need is served by
SERVES: dict[Channel, Belief] = {Channel.HUNGER: Belief.FEEDS, Channel.THIRST: Belief.QUENCHES}


def require[T](value: T | None, name: str, phase: str) -> T:
    """`value`, or a ValueError naming the workspace field `phase` needs but lacks."""
    if value is None:
        raise ValueError(f"{phase} needs ws.{name}, which is missing")
    return value


def _need(felt: FeltRecord, p: RuleParams) -> Channel | None:
    """Hunger or thirst, whichever is felt louder (hunger on a tie), if it is loud enough."""
    hunger, thirst = felt.levels[Channel.HUNGER], felt.levels[Channel.THIRST]
    need = Channel.HUNGER if hunger >= thirst else Channel.THIRST
    return need if felt.levels[need] >= p.need_at else None


def _search(
    start: Position,
    goal: Callable[[Position], bool],
    passable: Callable[[Position], bool],
    reach: int,
) -> tuple[Direction, ...]:
    """The first direction of every shortest path from `start` to a goal, breadth-first. A goal
    ends a path whether or not it is passable; only passable tiles are walked through. Only
    tiles within `reach` of (0, 0) along each axis are visited; `start` is never a goal."""

    def open_to(t: Position) -> bool:
        return abs(t[0]) <= reach and abs(t[1]) <= reach and (goal(t) or passable(t))

    # each tile of the present layer, with the first directions of its shortest paths
    layer: dict[Position, set[Direction]] = {}
    for direction in Direction:
        if open_to(t := moved_to(start, direction)):
            layer.setdefault(t, set()).add(direction)
    seen = {start, *layer}
    while layer:
        found = [t for t in layer if goal(t)]
        if found:
            return tuple(d for d in Direction if any(d in layer[t] for t in found))
        following: dict[Position, set[Direction]] = {}
        for t, firsts in layer.items():
            for n in neighbours(t):
                if n not in seen and open_to(n):
                    following.setdefault(n, set()).update(firsts)
        seen.update(following)
        layer = following
    return ()


def rule_choice(
    felt: FeltRecord,
    own_map: CreatureMap,
    tries: dict[str, int],
    p: RuleParams,
    rng: random.Random,
) -> RuleChoice:
    """The ten-line rule: walk the shortest believed-safe way to the first goal set that has a goal
    in reach (what serves the need, caution yielding sooner when the need is desperate; with no
    need, what is worth trying or already known); failing every set, anywhere believed safe."""
    here = felt.at

    def safe(t: Position) -> bool:
        return own_map.at(t)[Belief.HURTS] < p.avoid_at

    def passable(t: Position) -> bool:
        return own_map.at(t)[Belief.BLOCKS] < p.believe_at and safe(t)

    def tried(t: Position) -> bool:
        return tries.get(key(t), 0) >= 1

    def untried(t: Position) -> bool:
        return t != here and not tried(t)

    def untried_safe(t: Position) -> bool:
        return untried(t) and safe(t)

    def tried_safe(t: Position) -> bool:
        return t != here and tried(t) and safe(t)

    need = _need(felt, p)
    u = own_map.untried
    goal_sets: tuple[Callable[[Position], bool], ...]
    if need is not None:
        serves = SERVES[need]

        def serving(t: Position) -> bool:
            return t != here and own_map.at(t)[serves] >= p.serve_at

        def serving_safe(t: Position) -> bool:
            return serving(t) and safe(t)

        if felt.levels[need] >= p.desperate_at:
            goal_sets = (serving_safe, serving, untried_safe, untried)
        else:
            goal_sets = (serving_safe, untried_safe, serving, untried)
    elif u[Belief.FEEDS] + u[Belief.QUENCHES] + u[Belief.PLEASES] > u[Belief.HURTS]:
        goal_sets = (untried_safe, tried_safe)
    else:
        goal_sets = (tried_safe, untried_safe)

    best: tuple[Direction, ...] = ()
    for goal in goal_sets:
        if best := _search(here, goal, passable, p.reach):
            break
    if not best:
        best = tuple(d for d in Direction if safe(moved_to(here, d))) or tuple(Direction)
    return RuleChoice(best=best, choice=rng.choice(best))


def _near(own_map: CreatureMap, at: Position, belief: Belief, p: RuleParams) -> int:
    """The ladder step of the neighbours of `at` believed to have `belief`."""
    return near_step(sum(own_map.at(n)[belief] >= p.believe_at for n in neighbours(at)))


def rule_expectation(
    direction: Direction,
    felt: FeltRecord,
    own_map: CreatureMap,
    tries: dict[str, int],
    p: RuleParams,
) -> Expectation:
    """What the rule expects to feel after stepping `direction`, from its own map alone."""
    target = moved_to(felt.at, direction)
    b = own_map.at(target)
    moved = b[Belief.BLOCKS] < p.believe_at
    levels = dict(felt.levels)
    if not moved and b[Belief.FEEDS] >= p.believe_at:
        if levels[Channel.HUNGER] >= 1:
            levels[Channel.HUNGER] -= 1
        else:
            levels[Channel.ACHE] = min(TOP_STEP, levels[Channel.ACHE] + 1)
    if not moved and b[Belief.QUENCHES] >= p.believe_at and levels[Channel.THIRST] >= 1:
        levels[Channel.THIRST] -= 1
    if moved and b[Belief.PLEASES] >= p.believe_at:
        levels[Channel.PLEASURE] = max(p.felt_step, levels[Channel.PLEASURE])
    if moved and b[Belief.HURTS] >= p.believe_at:
        levels[Channel.UNEASE] = max(p.felt_step, levels[Channel.UNEASE])
    stand = target if moved else felt.at
    count = tries.get(key(target), 0)
    if count >= p.sure_tries:
        sure = Sureness.SURE
    elif count >= p.fairly_sure_tries:
        sure = Sureness.FAIRLY_SURE
    else:
        sure = Sureness.GUESSING
    return Expectation(
        moved=moved,
        levels=levels,
        smell=_near(own_map, stand, Belief.FEEDS, p),
        damp=_near(own_map, stand, Belief.QUENCHES, p),
        sure=sure,
    )


def blind_decision(felt: FeltRecord, rng: random.Random) -> Decision:
    """A chooser that reads nothing: any direction, expecting to move and feel no change."""
    expectation = Expectation(
        moved=True,
        levels=dict(felt.levels),
        smell=felt.smell,
        damp=felt.damp,
        sure=Sureness.GUESSING,
    )
    return Decision(direction=rng.choice(list(Direction)), expectation=expectation, why="")


# the beliefs of one map, by tile
MapBeliefs = dict[Position, dict[Belief, float]]


def _error(a: MapBeliefs, b: MapBeliefs, tiles: Sequence[Position]) -> float:
    """The mean absolute difference of two maps over `tiles` and the five beliefs; 0.0 for none."""
    diffs = [abs(a[t][x] - b[t][x]) for t in tiles for x in Belief]
    return math.fsum(diffs) / len(diffs) if diffs else 0.0


def measures(agent: AgentState, world: WorldConfig, p: MetricsParams) -> Measures:
    """How far the creature's own map and the reasoner's are from the truth, and from each other.
    Each tile's beliefs and truth are read once."""
    tried = sorted(position_of(k) for k, n in agent.tries.items() if n >= 1)
    everywhere = layout_positions(world)
    tiles = set(tried).union(everywhere)
    own = {t: agent.own_map.at(t) for t in tiles}
    reasoner = {t: reasoner_beliefs(agent.reasoner, t) for t in tiles}
    truth = {t: truth_at(world, t) for t in tiles}
    start = dict.fromkeys(tried, agent.own_map.untried)

    def false_harm(believed: MapBeliefs) -> int:
        return sum(
            believed[t][Belief.HURTS] >= p.wrong_side and truth[t][Belief.HURTS] == 0 for t in tried
        )

    def false_safe(believed: MapBeliefs) -> int:
        return sum(
            believed[t][Belief.HURTS] < p.wrong_side and truth[t][Belief.HURTS] == 1 for t in tried
        )

    def wrong_side(t: Position) -> bool:
        return any((own[t][x] >= p.wrong_side) != (truth[t][x] >= p.wrong_side) for x in Belief)

    return Measures(
        tiles_tried=len(tried),
        start_error=_error(start, truth, tried),
        own_error=_error(own, truth, tried),
        reasoner_error=_error(reasoner, truth, tried),
        gap=_error(own, reasoner, tried),
        own_error_all=_error(own, truth, everywhere),
        reasoner_error_all=_error(reasoner, truth, everywhere),
        gap_all=_error(own, reasoner, everywhere),
        false_harm=false_harm(own),
        false_safe=false_safe(own),
        reasoner_false_harm=false_harm(reasoner),
        reasoner_false_safe=false_safe(reasoner),
        wrong=tuple(sorted(key(t) for t in tried if wrong_side(t))),
    )
