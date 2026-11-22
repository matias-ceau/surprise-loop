"""The film: two stored runs on the same layout, played side by side as three maps.

The page gets what it draws and nothing more: the world's frame with the truth of every tile, and
for each run its name, the starting beliefs of both maps, where it starts and how it feels there,
and one entry per tick with where it went and what it felt and expected (feelings as words), how
the two compared, its one line of reading, the changes it made to its own map, the tiles whose
beliefs the careful reasoner moved, and the tried tiles where each map is on the wrong side of the
truth (worked out here, by the rule the measures use). The page replays the changes over the
starting maps, so it can show either map at any tick. Numbers are rounded to `DIGITS` decimals.
"""

from pathlib import Path
from typing import Any

from surprise_loop.agents import expected_words, felt_words, how_it_went
from surprise_loop.config import RunConfig, initial_agent_state
from surprise_loop.loop import RunError, RunResult, is_complete, load_dumps, read_config
from surprise_loop.phases.heuristics import tried_tiles, wrong_tiles
from surprise_loop.phrases import FEEL_WORDS, NEAR_WORDS, STIR_WORDS, SURPRISE_WORDS, TONE_WORDS
from surprise_loop.reasoner import marginals, reasoner_beliefs
from surprise_loop.state import FeltRecord, TickDump
from surprise_loop.viewer import export_page
from surprise_loop.world import (
    Belief,
    Channel,
    GridWorld,
    WorldConfig,
    frame,
    key,
    kind_at,
    position_of,
    truth_at,
)

DIGITS = 4  # decimals kept in the embedded numbers
TEMPLATE = "film.html"
PLACEHOLDER = "@@FILM_DATA@@"


def _r(x: float) -> float:
    return round(x, DIGITS)


def _beliefs(beliefs: dict[Belief, float]) -> dict[str, float]:
    return {b.value: _r(beliefs[b]) for b in Belief}


def _felt(felt: FeltRecord) -> dict[str, Any]:
    """The felt record in words, with its tone and stir apart."""
    words = felt_words(felt)
    tone, stir = words.pop("tone"), words.pop("stir")
    return {"felt": words, "tone": tone, "stir": stir}


def _world(config: WorldConfig) -> dict[str, Any]:
    west, south, east, north = frame(config)
    tiles = {}
    for y in range(north, south - 1, -1):
        for x in range(west, east + 1):
            tiles[key((x, y))] = {
                "kind": kind_at(config, (x, y)).value,
                "beliefs": _beliefs(truth_at(config, (x, y))),
            }
    return {"frame": [west, south, east, north], "tiles": tiles}


def _tick(dump: TickDump, config: RunConfig) -> dict[str, Any]:
    r, reasoner = dump.record, dump.agent.reasoner
    return {
        "at": list(r.felt.at),
        "direction": r.decision.direction.value,
        "target": list(r.outcome.info.target),
        **_felt(r.felt),
        "expected": expected_words(r.decision.expectation),
        **how_it_went(r.comparison, config.params.monitor),
        "reading": r.appraisal.reading,
        "changes": [
            {"at": list(change.at), "beliefs": _beliefs(change.beliefs)}
            for change in r.appraisal.changes
        ],
        "reasoner": [
            {"at": list(at), "beliefs": _beliefs(reasoner_beliefs(reasoner, at))}
            for at in r.reasoner_changed
        ],
        "wrong": {
            "own": list(r.measures.wrong),
            "reasoner": list(
                wrong_tiles(
                    lambda at: reasoner_beliefs(reasoner, at),
                    tried_tiles(dump.agent.tries),
                    config.world,
                    config.params.metrics.wrong_side,
                )
            ),
        },
    }


def _run(config: RunConfig, dumps: list[TickDump]) -> dict[str, Any]:
    start = dumps[0].record.felt_before
    metrics = config.params.metrics
    # the maps before the first tick, which no dump holds
    first = initial_agent_state(config, GridWorld(config.world, config.seed).observe())
    return {
        "name": config.run_id,
        "agent": config.agent.name,
        "untried": {
            "own": _beliefs(first.own_map.untried),
            "reasoner": _beliefs(marginals(first.reasoner.presumes)),
        },
        "start_tiles": {
            "own": {k: _beliefs(b) for k, b in first.own_map.tiles.items()},
            "reasoner": {
                k: _beliefs(reasoner_beliefs(first.reasoner, position_of(k)))
                for k in first.reasoner.tiles
            },
        },
        "marks": {"wrong_side": _r(metrics.wrong_side), "rewritten_by": _r(metrics.rewritten_by)},
        "start": {"at": list(start.at), **_felt(start)},
        "ticks": [_tick(d, config) for d in dumps],
    }


def _read(run_dir: Path) -> tuple[RunConfig, list[TickDump]]:
    config, dumps = read_config(run_dir), load_dumps(run_dir)
    if not dumps:
        raise RunError(f"{run_dir}: no tick yet")
    last = dumps[-1]
    result = RunResult(
        config=config, records=[d.record for d in dumps], agent=last.agent, world=last.world
    )
    if not is_complete(result):
        raise RunError(f"{run_dir}: not complete: {len(dumps)} ticks; resume it first")
    return config, dumps


def film_data(run_a: Path, run_b: Path) -> dict[str, Any]:
    """What the film plays; RunError when the two runs are the same directory, one has no tick or
    is not complete, they are in different worlds (the layout or anything else of the world
    config), have different metrics params (the page's outline and ring), or have different
    numbers of ticks."""
    if Path(run_a).resolve() == Path(run_b).resolve():
        raise RunError(f"{run_a}: the same run twice")
    (config_a, dumps_a), (config_b, dumps_b) = _read(run_a), _read(run_b)
    if config_a.world.layout != config_b.world.layout:
        raise RunError(f"{run_a} and {run_b} do not belong together: their layouts differ")
    if config_a.world != config_b.world:
        raise RunError(f"{run_a} and {run_b} do not belong together: their worlds differ")
    if config_a.params.metrics != config_b.params.metrics:
        raise RunError(f"{run_a} and {run_b} do not belong together: their metrics params differ")
    if len(dumps_a) != len(dumps_b):
        raise RunError(
            f"{run_a} and {run_b} do not belong together: {len(dumps_a)} and {len(dumps_b)} ticks"
        )
    return {
        "world": _world(config_a.world),
        "ladders": {
            "feel": list(FEEL_WORDS),
            "near": list(NEAR_WORDS),
            "tone": list(TONE_WORDS),
            "stir": list(STIR_WORDS),
            "surprise": list(SURPRISE_WORDS),
        },
        "channels": [c.value for c in Channel],
        "runs": [_run(config_a, dumps_a), _run(config_b, dumps_b)],
    }


def film_html(run_a: Path, run_b: Path) -> str:
    """The film of two runs as one self-contained HTML page."""
    return export_page(TEMPLATE, PLACEHOLDER, film_data(run_a, run_b))
