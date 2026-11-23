"""The numbers the README quotes about the model-driven day that no command prints. Read-only.

Usage, from the repository, with the model runs unpacked into RUNS_DIR and the two no-model
experiments run into the same directory:

    uv run python docs/results/2026-11-22/readme_numbers.py RUNS_DIR
"""

import sys
from pathlib import Path
from statistics import mean, stdev

from surprise_loop.loop import RunResult, load_dumps, load_result
from surprise_loop.metrics import map_distance, summarize
from surprise_loop.world import Belief, Channel, Kind, key, kind_at, layout_positions, position_of

AGENTS, SEEDS = ("plain", "wary", "hopeful"), (1, 2)
EXPERIMENTS = {"model": "two_maps_llm", "stand-in": "two_maps", "blind": "two_maps_blind"}
# what a thing that stops the creature does for it, and which of those bring sickness
SERVES = {
    Kind.FOOD: Belief.FEEDS,
    Kind.TOXIC: Belief.FEEDS,
    Kind.WATER: Belief.QUENCHES,
    Kind.FOUL: Belief.QUENCHES,
}
SICKENS = {Kind.TOXIC: Channel.HUNGER, Kind.FOUL: Channel.THIRST}
WALKABLE = (Kind.GROUND, Kind.PLEASANT, Kind.UNPLEASANT)
# a belief at or above this counts as held
HELD = 0.5
# "strong" on the five-word ladder
STRONG = 3


def tried_in_layout(result: RunResult) -> int:
    """Tiles of the layout tried. The measure `tiles_tried` also counts the ring around it."""
    layout = {key(at) for at in layout_positions(result.config.world)}
    return sum(1 for at, tries in result.agent.tries.items() if tries > 0 and at in layout)


def layout_means(runs: Path, experiment: str) -> None:
    """Over every seed found: tiles tried inside the layout, and the hopeful - wary difference."""
    found: dict[str, dict[int, int]] = {}
    for run_dir in sorted(runs.glob(f"{experiment}-*-s*")):
        agent, seed = run_dir.name[len(experiment) + 1 :].rsplit("-s", 1)
        found.setdefault(agent, {})[int(seed)] = tried_in_layout(load_result(run_dir))
    for agent, by_seed in found.items():
        print(f"  {agent:8s} {mean(by_seed.values()):.2f} over {len(by_seed)} seeds")
    seeds = sorted(set(found.get("hopeful", {})) & set(found.get("wary", {})))
    if len(seeds) > 1:
        apart = [found["hopeful"][s] - found["wary"][s] for s in seeds]
        two_se = 2 * stdev(apart) / len(apart) ** 0.5
        print(f"  hopeful - wary: mean {mean(apart):.2f}, two standard errors {two_se:.2f}")


def per_run(kind: str, result: RunResult) -> str:
    m = summarize(result)
    closed = (m.start_error - m.own_error) / (m.start_error - m.reasoner_error)
    return (
        f"{kind:9s} {result.config.agent.name:8s} s{result.config.seed}"
        f"  tried {m.tiles_tried:2d} ({tried_in_layout(result):2d} in the layout)"
        f"  sick {m.sickness_arrivals}"
        f"  uncomfortable {m.uncomfortable_total:3d}"
        f"  start {m.start_error:.3f}  own {m.own_error:.3f}  reasoner {m.reasoner_error:.3f}"
        f"  closed {closed:.0%}  rule {m.rule_agreement:.2f}  chance {m.chance_agreement:.2f}"
        f"  false harm {m.false_harm}  stray {m.stray_changes}"
    )


def distances(results: dict[tuple[str, int], RunResult]) -> None:
    world = next(iter(results.values())).config.world
    between = [
        (results[a, s], results[b, s])
        for s in SEEDS
        for i, a in enumerate(AGENTS)
        for b in AGENTS[i + 1 :]
    ]
    within = [(results[a, SEEDS[0]], results[a, SEEDS[1]]) for a in AGENTS]
    for which in ("own", "reasoner"):
        for tiles in ("shared", "all"):
            row = []
            for pairs in (between, within):
                found = [map_distance(x.agent, y.agent, world, which, tiles) for x, y in pairs]
                row.append(mean(d for d in found if d is not None))
            print(f"  {which:8s} map, {tiles:6s} tiles: between {row[0]:.4f}  within {row[1]:.4f}")


def end_of_day(result: RunResult) -> dict[str, int]:
    """The creature's own map at the end of the day, read against the world."""
    world, own = result.config.world, result.agent.own_map
    count = dict.fromkeys(
        ("things", "things_credited", "sickening", "sickening_marked", "ground", "ground_credited"),
        0,
    )
    for at in (position_of(k) for k, n in result.agent.tries.items() if n > 0):
        kind, beliefs = kind_at(world, at), own.at(at)
        if kind in SERVES:
            count["things"] += 1
            count["things_credited"] += beliefs[SERVES[kind]] >= HELD
            if kind in SICKENS:
                count["sickening"] += 1
                count["sickening_marked"] += beliefs[Belief.HURTS] >= HELD
        elif kind in WALKABLE:
            count["ground"] += 1
            count["ground_credited"] += max(beliefs[Belief.FEEDS], beliefs[Belief.QUENCHES]) >= HELD
    return count


def sickening_pushes(run_dir: Path) -> dict[str, int]:
    """Pushes into toxic food or foul water, and what the creature's map said just before."""
    count = dict.fromkeys(("pushes", "believed_harmful", "need_strong", "knew_another"), 0)
    before = None
    for dump in load_dumps(run_dir):
        record = dump.record
        kind = record.outcome.info.target_kind
        if kind in SICKENS:
            count["pushes"] += 1
            if before is not None:
                target = before.own_map.at(record.outcome.info.target)
                if target[Belief.HURTS] >= HELD:
                    count["believed_harmful"] += 1
                    count["need_strong"] += record.felt_before.levels[SICKENS[kind]] >= STRONG
                    count["knew_another"] += any(
                        tile[SERVES[kind]] >= HELD and tile[Belief.HURTS] < HELD
                        for tile in before.own_map.tiles.values()
                    )
        before = dump.agent
    return count


def eased_pushes(run_dir: Path) -> dict[str, int]:
    """Pushes into food or water that eased the need, by the spot the creature then credited."""
    count = dict.fromkeys(("pushes", "eased", "thing", "own_tile", "neither"), 0)
    for dump in load_dumps(run_dir):
        record = dump.record
        kind = record.outcome.info.target_kind
        if kind not in SERVES:
            continue
        count["pushes"] += 1
        need = Channel.HUNGER if SERVES[kind] is Belief.FEEDS else Channel.THIRST
        if record.felt.levels[need] >= record.felt_before.levels[need]:
            continue
        count["eased"] += 1
        written = {tuple(c.at): c.beliefs[SERVES[kind]] for c in record.appraisal.changes}
        thing = written.get(tuple(record.outcome.info.target), 0.0) >= HELD
        own = written.get(tuple(record.felt.at), 0.0) >= HELD
        count["thing"] += thing
        count["own_tile"] += own
        count["neither"] += not thing and not own
    return count


def expectations(results: list[RunResult]) -> None:
    """How often each expected word was the word felt, against expecting the word felt before."""
    records = [r for result in results for r in result.records]
    for channel in Channel:
        hit = mean(r.comparison.misses[channel] == 0 for r in records)
        same = mean(r.felt_before.levels[channel] == r.felt.levels[channel] for r in records)
        print(f"  {channel.value:9s} expected right {hit:.3f}   no change right {same:.3f}")
    for name in ("smell", "damp"):
        hit = mean(getattr(r.comparison, f"{name}_miss") == 0 for r in records)
        same = mean(getattr(r.felt_before, name) == getattr(r.felt, name) for r in records)
        print(f"  {name:9s} expected right {hit:.3f}   no change right {same:.3f}")
    print(f"  moved     expected right {mean(not r.comparison.moved_miss for r in records):.3f}")


def total(rows: list[dict[str, int]]) -> dict[str, int]:
    return {name: sum(row[name] for row in rows) for name in rows[0]}


def main(runs: Path) -> None:
    for kind, experiment in EXPERIMENTS.items():
        dirs = {(a, s): runs / f"{experiment}-{a}-s{s}" for a in AGENTS for s in SEEDS}
        results = {at: load_result(d) for at, d in dirs.items()}
        print(f"\n== {kind} ({experiment}), seeds {SEEDS}")
        for result in results.values():
            print(per_run(kind, result))
        print("distances between the final maps (mean absolute difference of a belief):")
        distances(results)
        print("tiles of the layout tried:")
        layout_means(runs, experiment)
        if kind == "blind":
            continue
        print("end of day:", total([end_of_day(r) for r in results.values()]))
        print("sickening pushes:", total([sickening_pushes(d) for d in dirs.values()]))
        print("pushes that eased a need:", total([eased_pushes(d) for d in dirs.values()]))
        print("expectations:")
        expectations(list(results.values()))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
