"""The film: two stored runs of the same day played side by side, paired by world step.

One world step is one pass (a tick of two passes gives two steps). For each step and body the page
gets what it draws and nothing more: where and when, the felt scene and its text, the reading,
the action and what was expected, the outcome, the comparison, belief and feeling after, the
world's truth for the measured strip, and the override. The episode list a tick file repeats is
never read into the page. Numbers are rounded to `DIGITS` decimals.
"""

from typing import Any

from surprise_loop.feeling import felt
from surprise_loop.loop import RunError, RunResult
from surprise_loop.metrics import flatten
from surprise_loop.phrases import LADDER_WORDS
from surprise_loop.state import Band, FeltScene, PassRecord, Temperament
from surprise_loop.viewer import export_page
from surprise_loop.world import QUESTION_OF, Channel, Levels, Place

DIGITS = 4  # decimals kept in the embedded numbers
PLACEHOLDER = "@@SCENE_DATA@@"


def _r(x: float) -> float:
    return round(x, DIGITS)


def _levels(levels: Levels) -> dict[str, float]:
    return {c.value: _r(levels.of(c)) for c in Channel}


def expected_felt(bands: dict[Channel, Band], t: Temperament) -> dict[Channel, float | None]:
    """The felt strength at the middle of each band; for a band open on one side, at its finite
    edge; None for a band open on both sides (nothing expected for that channel)."""
    middle: dict[Channel, float | None] = {}
    for c in Channel:
        band = bands[c]
        edges = [x for x in (band.lo, band.hi) if x is not None]
        middle[c] = sum(edges) / len(edges) if edges else None
    # a channel without an expectation is read at the setpoint, then reported as None
    levels = Levels(**{c.value: t.setpoint.of(c) if v is None else v for c, v in middle.items()})
    strengths = felt(levels, t)
    return {c: None if middle[c] is None else strengths[c] for c in Channel}


def _scene(scene: FeltScene) -> dict[str, Any]:
    return {
        "place": scene.place.value,
        "opening": scene.opening,
        "text": scene.text(),
        "facts": [
            {
                "name": f.name.value,
                "weight": _r(f.weight),
                "value": _r(f.value),
                "sign": f.sign,
                "grade": f.grade,
                "noticed": f.noticed,
                "phrase": f.phrase,
            }
            for f in scene.facts
        ],
        "body": [
            {"channel": b.channel.value, "grade": b.grade, "felt": _r(b.felt), "phrase": b.phrase}
            for b in scene.body
        ],
        "surprise": scene.surprise,
        "felt": {c.value: _r(x) for c, x in scene.felt.items()},
    }


def _step(p: PassRecord, reading: str, arousal_before: float, t: Temperament) -> dict[str, Any]:
    decision, comparison, info = p.decision, p.comparison, p.outcome.info
    expectation = decision.expectation
    words = None
    if expectation.words is not None:
        words = {c.value: LADDER_WORDS[c][w] for c, w in expectation.words.items()}
    question = QUESTION_OF[decision.action]
    return {
        "place": p.scene.place.value,
        "place_after": info.place.value,
        "second_pass": p.pass_index == 1,
        "scene": _scene(p.scene),
        "reading": reading,
        "action": decision.action.value,
        "event": p.outcome.observation.event.value,
        "expected": {
            "words": words,
            "felt": {
                c.value: None if x is None else _r(x)
                for c, x in expected_felt(expectation.bands, t).items()
            },
            "bands": {
                c.value: [
                    None if b.lo is None else _r(b.lo),
                    None if b.hi is None else _r(b.hi),
                ]
                for c, b in expectation.bands.items()
            },
            "world": expectation.world,
            "question": None if question is None else question.value,
            "sureness": expectation.sureness.value,
        },
        "comparison": {
            "body": {c.value: _r(x) for c, x in comparison.body.items()},
            "body_felt": {c.value: _r(x) for c, x in comparison.body_felt.items()},
            "world": comparison.world,
            "better": _r(comparison.better),
            "size": _r(comparison.size),
        },
        "belief": _levels(p.belief_after),
        "feeling": {
            "valence": _r(p.feeling_after.valence),
            "arousal": _r(p.feeling_after.arousal),
        },
        "arousal_before": _r(arousal_before),
        "override": None
        if p.override is None
        else {
            "channel": p.override.channel.value,
            "actions": [a.value for a in p.override.actions],
        },
        "truth": {
            "air_c": _r(info.air_c),
            "sun": _r(info.sun),
            "log_shaded": info.log_shaded,
            "stream_full": info.stream_full,
            "stumbled": info.stumbled,
            "collapsed": info.collapsed,
            "body": {
                "heat": _r(info.body.heat),
                "water": _r(info.body.water),
                "energy": _r(info.body.energy),
            },
        },
    }


def _steps(result: RunResult) -> list[dict[str, Any]]:
    """One entry per world step of a run, in world-step order."""
    spec = result.config.agent
    readings = [r.appraisal.reading for r in result.records for _ in r.passes]
    out, arousal = [], spec.arousal
    for p, reading in zip(flatten(result.records), readings, strict=True):
        out.append(_step(p, reading, arousal, spec.temperament))
        arousal = p.feeling_after.arousal
    return out


def _check_pair(a: RunResult, b: RunResult) -> int:
    """The number of world steps the two runs share; RunError when they do not belong together."""
    ida, idb = a.config.run_id, b.config.run_id
    if a.config.seed != b.config.seed:
        raise RunError(
            f"{ida} and {idb} do not belong together: seeds {a.config.seed} and {b.config.seed}"
        )
    if a.config.world != b.config.world:
        raise RunError(f"{ida} and {idb} do not belong together: their world configs differ")
    na, nb = len(flatten(a.records)), len(flatten(b.records))
    if na != nb:
        raise RunError(f"{ida} and {idb} do not belong together: {na} and {nb} world steps")
    if na == 0:
        raise RunError(f"{ida} and {idb} have no world steps yet")
    return na


def _start_minute(result: RunResult) -> int:
    return result.records[0].observation.minute


def scene_data(a: RunResult, b: RunResult) -> dict[str, Any]:
    """What the film plays: both runs' steps, paired by index, and the measured strip's world."""
    _check_pair(a, b)
    bodies = [_steps(a), _steps(b)]
    minutes_after = [p.outcome.info.minute for p in flatten(a.records)]
    minutes = [_start_minute(a), *minutes_after[:-1]]
    world = a.config.world
    return {
        "agents": [
            {
                "name": r.config.agent.name,
                "setpoint": r.config.agent.temperament.setpoint.model_dump(),
                "start_body": {
                    "heat": _r(r.config.agent.body.heat),
                    "water": _r(r.config.agent.body.water),
                    "energy": _r(r.config.agent.body.energy),
                },
                "start_place": r.config.start_place.value,
            }
            for r in (a, b)
        ],
        "world": {
            "paces": world.paces,
            "tick_minutes": world.tick_minutes,
            "places": [p.value for p in Place],
        },
        "steps": [
            {"minute": m, "minute_after": m_after, "bodies": [sa, sb]}
            for m, m_after, sa, sb in zip(minutes, minutes_after, *bodies, strict=True)
        ],
    }


def scene_html(a: RunResult, b: RunResult) -> str:
    """The film of two runs as one self-contained HTML page; RunError when the runs have different
    seeds, world configs or numbers of world steps."""
    return export_page("scene.html", PLACEHOLDER, scene_data(a, b))
