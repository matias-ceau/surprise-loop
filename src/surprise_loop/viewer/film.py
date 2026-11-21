"""The film: two stored runs on the same layout, played side by side as three maps.

The page gets what it draws and nothing more: the world's frame with the truth of every tile, and
for each run its name, the starting beliefs of both maps, where it starts and how it feels there,
and one entry per tick with where it went and what it felt and expected (feelings as words), how
the two compared, its one line of reading, the changes it made to its own map and the tiles whose
beliefs the careful reasoner moved. The page replays the changes over the starting maps, so it can
show either map at any tick. Numbers are rounded to `DIGITS` decimals.
"""

from pathlib import Path
from typing import Any

from surprise_loop.agents import _expected_words, _felt_words, _how_it_went
from surprise_loop.config import MonitorParams, RunConfig
from surprise_loop.loop import RunError, load_dumps, read_config
from surprise_loop.phrases import FEEL_WORDS, NEAR_WORDS, STIR_WORDS, SURPRISE_WORDS, TONE_WORDS
from surprise_loop.reasoner import marginals, reasoner_beliefs
from surprise_loop.state import FeltRecord, TickDump
from surprise_loop.viewer import export_page
from surprise_loop.world import Belief, Channel, WorldConfig, frame, key, kind_at, truth_at

DIGITS = 4  # decimals kept in the embedded numbers
TEMPLATE = "film.html"
PLACEHOLDER = "@@FILM_DATA@@"


def _r(x: float) -> float:
    return round(x, DIGITS)


def _beliefs(beliefs: dict[Belief, float]) -> dict[str, float]:
    return {b.value: _r(beliefs[b]) for b in Belief}


def _felt(felt: FeltRecord) -> dict[str, Any]:
    """The felt record in words, with its tone and stir apart."""
    words = _felt_words(felt)
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


def _tick(dump: TickDump, p: MonitorParams) -> dict[str, Any]:
    r, c = dump.record, dump.record.comparison
    return {
        "tick": r.tick,
        "at": list(r.felt.at),
        "direction": r.decision.direction.value,
        "target": list(r.outcome.info.target),
        **_felt(r.felt),
        "expected": _expected_words(r.decision.expectation),
        "misses": {
            "moved": c.moved_miss,
            **{ch.value: c.misses[ch] for ch in Channel},
            "smell": c.smell_miss,
            "damp": c.damp_miss,
        },
        **_how_it_went(c, p),
        "reading": r.appraisal.reading,
        "changes": [
            {"at": list(change.at), "beliefs": _beliefs(change.beliefs)}
            for change in r.appraisal.changes
        ],
        "reasoner": [
            {"at": list(at), "beliefs": _beliefs(reasoner_beliefs(dump.agent.reasoner, at))}
            for at in r.reasoner_changed
        ],
        "wrong": list(r.measures.wrong),
    }


def _run(config: RunConfig, dumps: list[TickDump]) -> dict[str, Any]:
    start = dumps[0].record.felt_before
    metrics = config.params.metrics
    return {
        "name": config.run_id,
        "agent": config.agent.name,
        "untried": {
            "own": _beliefs(dumps[0].agent.own_map.untried),
            "reasoner": _beliefs(marginals(dumps[0].agent.reasoner.presumes)),
        },
        "marks": {"wrong_side": _r(metrics.wrong_side), "rewritten_by": _r(metrics.rewritten_by)},
        "start": {"at": list(start.at), **_felt(start)},
        "ticks": [_tick(d, config.params.monitor) for d in dumps],
    }


def _read(run_dir: Path) -> tuple[RunConfig, list[TickDump]]:
    config, dumps = read_config(run_dir), load_dumps(run_dir)
    if not dumps:
        raise RunError(f"{run_dir}: no tick yet")
    return config, dumps


def film_data(run_a: Path, run_b: Path) -> dict[str, Any]:
    """What the film plays; RunError when the two runs are the same directory, are on different
    layouts, have different metrics params (the page's outline and ring), have different numbers
    of ticks, or one has no tick."""
    if Path(run_a).resolve() == Path(run_b).resolve():
        raise RunError(f"{run_a}: the same run twice")
    (config_a, dumps_a), (config_b, dumps_b) = _read(run_a), _read(run_b)
    if config_a.world.layout != config_b.world.layout:
        raise RunError(f"{run_a} and {run_b} do not belong together: their layouts differ")
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
