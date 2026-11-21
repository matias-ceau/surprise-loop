"""The careful reasoner: what a creature should believe of each tile, given exactly what it felt.

It keeps, per tile, a share for each kind of tile and updates them after every step: from
whether the step moved, from what changed in the body at once, from sickness arriving some
steps after a touch (or not arriving), and from smell and damp. Its map is the yardstick the
creature's own map is compared with; nothing of it ever reaches a prompt.
"""

import math
from collections.abc import Callable
from itertools import product

from surprise_loop.config import ReasonerParams
from surprise_loop.state import FeltRecord, ReasonerState, Touch
from surprise_loop.world import (
    DRINKABLE,
    EDIBLE,
    EFFECTS,
    SICKENING,
    WALKABLE,
    Belief,
    BodyParams,
    Channel,
    Direction,
    Kind,
    Position,
    key,
    moved_to,
    neighbours,
    position_of,
)

# a belief that moved by no more than this has not moved
MOVED_BY = 1e-9

Shares = dict[Kind, float]


def initial_reasoner(presumes: dict[Kind, float]) -> ReasonerState:
    return ReasonerState(presumes=dict(presumes), tiles={}, touched=(), onsets=())


def shares_at(state: ReasonerState, at: Position) -> dict[Kind, float]:
    return dict(state.tiles.get(key(at), state.presumes))


def marginals(shares: dict[Kind, float]) -> dict[Belief, float]:
    """For each belief, the summed share of the kinds that have it."""
    return {b: math.fsum(shares[k] for k in Kind if b in EFFECTS[k]) for b in Belief}


def reasoner_beliefs(state: ReasonerState, at: Position) -> dict[Belief, float]:
    return marginals(shares_at(state, at))


def _normalised(shares: Shares) -> Shares:
    """The shares over their total; unchanged when the total is 0."""
    total = math.fsum(shares.values())
    if total == 0:
        return shares
    return {k: v / total for k, v in shares.items()}


def _scaled(shares: Shares, group: frozenset[Kind], total: float) -> Shares:
    """The group's kinds scaled together to `total`, the other kinds to the rest. A side whose
    present total is 0 is left as it is."""
    present = math.fsum(shares[k] for k in group)
    rest = math.fsum(v for k, v in shares.items() if k not in group)
    scaled = dict(shares)
    for k, v in shares.items():
        if k in group and present > 0:
            scaled[k] = v * total / present
        elif k not in group and rest > 0:
            scaled[k] = v * (1 - total) / rest
    return scaled


def _after_moving(shares: Shares, moved: bool) -> Shares:
    """Step 1: rule out the kinds the step's outcome rules out."""
    kept = WALKABLE if moved else frozenset(Kind) - WALKABLE
    left = {k: (v if k in kept else 0.0) for k, v in shares.items()}
    if math.fsum(left.values()) == 0:
        return {k: (1 / len(kept) if k in kept else 0.0) for k in Kind}
    return _normalised(left)


def _seen(happened: bool, chance: float) -> float:
    """The likelihood of an event's observed value, when its chance of happening is `chance`."""
    return chance if happened else 1 - chance


def _likelihoods(before: FeltRecord, after: FeltRecord, p: ReasonerParams) -> Shares:
    """Step 2: per kind, the likelihood of every relevant event as observed."""
    s = p.slip
    was, now = before.levels, after.levels
    if after.moved:
        pleased = now[Channel.PLEASURE] > was[Channel.PLEASURE]
        uneased = now[Channel.UNEASE] > was[Channel.UNEASE]
        return {
            k: _seen(pleased, 1 - s if k == Kind.PLEASANT else s)
            * _seen(uneased, 1 - s if k == Kind.UNPLEASANT else s)
            for k in Kind
        }
    fed = now[Channel.HUNGER] < was[Channel.HUNGER]
    quenched = now[Channel.THIRST] < was[Channel.THIRST]
    ached = now[Channel.ACHE] > was[Channel.ACHE]
    hunger = was[Channel.HUNGER]
    p_fed = 1 - s if hunger >= 1 else s
    p_quenched = 1 - s if was[Channel.THIRST] >= 1 else s
    if hunger == 0:
        p_ached = 1 - s
    elif hunger == 1:
        p_ached = p.ache_when_slight
    else:
        p_ached = s
    return {
        k: _seen(fed, p_fed if k in EDIBLE else s)
        * _seen(quenched, p_quenched if k in DRINKABLE else s)
        * _seen(ached, p_ached if k in EDIBLE else s)
        for k in Kind
    }


def _harm(shares: Shares) -> float:
    return math.fsum(shares[k] for k in SICKENING)


def _observation_chance(observed: int, holding: int, q: float) -> float:
    """The chance of `observed` (none, one, two or more) when `holding` neighbours hold the group
    and each is noticed with chance `q`."""
    none = (1 - q) ** holding
    one = holding * q * (1 - q) ** (holding - 1) if holding else 0.0
    return (none, one, max(0.0, 1 - none - one))[observed]


def _sensed(
    get: Callable[[Position], Shares],
    tiles: dict[str, Shares],
    around: tuple[Position, ...],
    group: frozenset[Kind],
    observed: int,
    q: float,
) -> None:
    """Step 6 for one sense: write the group's new share at each neighbour into `tiles`."""
    before = [get(at) for at in around]
    held = [math.fsum(s[k] for k in group) for s in before]
    weights = []
    for holds in product((False, True), repeat=len(around)):
        prior = math.prod(e if h else 1 - e for e, h in zip(held, holds, strict=True))
        weights.append((holds, prior * _observation_chance(observed, sum(holds), q)))
    total = math.fsum(w for _, w in weights)
    if total == 0:
        return
    for i, at in enumerate(around):
        share = math.fsum(w for holds, w in weights if holds[i]) / total
        scaled = _scaled(before[i], group, share)
        if scaled != before[i]:
            tiles[key(at)] = scaled


def reason(
    state: ReasonerState,
    step: int,
    direction: Direction,
    before: FeltRecord,
    after: FeltRecord,
    body: BodyParams,
    p: ReasonerParams,
) -> tuple[ReasonerState, tuple[Position, ...]]:
    """The reasoner's state after one step, and the positions whose beliefs moved, sorted."""
    s = p.slip
    target = moved_to(before.at, direction)
    window = max(body.sick_delays)
    tiles: dict[str, Shares] = {k: dict(v) for k, v in state.tiles.items()}

    def get(at: Position) -> Shares:
        return tiles.get(key(at), dict(state.presumes))

    # 1. moved
    moved = bool(after.moved)
    shares = _after_moving(get(target), moved)
    # 2. what changed at once
    likely = _likelihoods(before, after, p)
    tiles[key(target)] = _normalised({k: v * likely[k] for k, v in shares.items()})

    # 3. touches
    touched = list(state.touched)
    if not moved:
        touched.append(Touch(step=step, at=target))
    touched = [t for t in touched if t.step >= step - window]
    onsets = [o for o in state.onsets if o >= step - window]

    # 4. sickness arrived
    if after.levels[Channel.SICKNESS] > before.levels[Channel.SICKNESS]:
        onsets.append(step)
        weight: dict[Position, float] = {}
        for t in touched:
            if step - t.step in body.sick_delays:
                weight[t.at] = weight.get(t.at, 0.0) + _harm(get(t.at)) / len(body.sick_delays)
        total = math.fsum(weight.values())
        if total > 0:
            for at, w in weight.items():
                r = w / total
                h = _harm(get(at))
                tiles[key(at)] = _scaled(get(at), SICKENING, r + (1 - r) * h)

    # 5. cleared
    for t in touched:
        if step - t.step != window:
            continue
        if any(o - t.step in body.sick_delays for o in onsets):
            continue
        tiles[key(t.at)] = _normalised(
            {k: v * s if k in SICKENING else v for k, v in get(t.at).items()}
        )

    # 6. smell, then damp
    around = neighbours(after.at)
    _sensed(get, tiles, around, EDIBLE, after.smell, body.smell_chance)
    _sensed(get, tiles, around, DRINKABLE, after.damp, body.damp_chance)

    new = ReasonerState(
        presumes=dict(state.presumes),
        tiles={
            k: {kind: min(1.0, max(0.0, v)) for kind, v in shares.items()}
            for k, shares in tiles.items()
        },
        touched=tuple(touched),
        onsets=tuple(onsets),
    )
    return new, _moved_positions(state, new)


def _moved_positions(old: ReasonerState, new: ReasonerState) -> tuple[Position, ...]:
    moved = []
    for k in set(old.tiles) | set(new.tiles):
        at = position_of(k)
        was, now = reasoner_beliefs(old, at), reasoner_beliefs(new, at)
        if any(abs(now[b] - was[b]) > MOVED_BY for b in Belief):
            moved.append(at)
    return tuple(sorted(moved))
