"""The lens page: the meadow at 13:00 as each of a config's two bodies lives it, with sliders.

The lens is computed here, over a grid of beliefs and arousals, plus one exact entry per preset and
body; the page only looks the results up, so the lens exists in one language. Grids and presets
are presentation constants: they choose what to show, not how a body behaves.
"""

import itertools
import json
from importlib import resources
from typing import Any

from surprise_loop.config import AgentSpec, ConfigError, ExperimentConfig, LensParams
from surprise_loop.lens import felt_scene
from surprise_loop.phrases import OPENING
from surprise_loop.state import FeltScene
from surprise_loop.world import Fact, Levels, Place, frozen_noon_facts

PLACE = Place.MEADOW
HEAT_GRID: tuple[float, ...] = tuple(round(-0.5 + 0.1 * i, 1) for i in range(18))
WATER_GRID: tuple[float, ...] = (0.25, 0.40, 0.55, 0.70, 0.85)
ENERGY_GRID: tuple[float, ...] = (0.25, 0.40, 0.55, 0.70, 0.85)
AROUSAL_GRID: tuple[float, ...] = (0.2, 0.4, 0.7)

# the ordinary pair of the plan (first agent, second agent): two heat grades apart, calm otherwise
ORDINARY_PAIR: tuple[tuple[Levels, float], tuple[Levels, float]] = (
    (Levels(heat=0.40, water=0.78, energy=0.66), 0.4),
    (Levels(heat=0.25, water=0.80, energy=0.70), 0.2),
)
# one belief on the grid for both bodies, so the temperaments are the only difference
SAME_BODY: tuple[Levels, float] = (Levels(heat=0.3, water=0.70, energy=0.55), 0.4)

DIGITS = 4  # decimals kept in the embedded numbers
PLACEHOLDER = "@@LENS_DATA@@"


def _entry(scene: FeltScene) -> dict[str, Any]:
    """What the page needs of one felt scene, and no more."""
    return {
        "text": scene.text(),
        "facts": [
            {
                "name": f.name.value,
                "weight": round(f.weight, DIGITS),
                "value": round(f.value, DIGITS),
                "sign": f.sign,
                "grade": f.grade,
                "noticed": f.noticed,
                "phrase": f.phrase,
            }
            for f in scene.facts
        ],
        "body": [
            {
                "channel": line.channel.value,
                "grade": line.grade,
                "felt": round(line.felt, DIGITS),
                "phrase": line.phrase,
            }
            for line in scene.body
        ],
        "felt": {c.value: round(x, DIGITS) for c, x in scene.felt.items()},
        "surprise": scene.surprise,
    }


def _lens(
    agent: AgentSpec,
    facts: list[Fact],
    belief: Levels,
    arousal: float,
    p: LensParams,
    *,
    voiced: bool = False,
) -> dict[str, Any]:
    scene = felt_scene(
        PLACE,
        facts,
        belief,
        agent.temperament,
        arousal,
        None,
        agent.voiced if voiced else None,
        0,
        p,
    )
    return _entry(scene)


def _body(
    agent: AgentSpec,
    facts: list[Fact],
    belief: Levels,
    arousal: float,
    p: LensParams,
    *,
    voiced: bool = False,
) -> dict[str, Any]:
    return {
        "belief": belief.model_dump(),
        "arousal": arousal,
        "entry": _lens(agent, facts, belief, arousal, p, voiced=voiced),
    }


def lens_data(config: ExperimentConfig) -> dict[str, Any]:
    """The grid and the presets for the config's two agents, as the page reads them."""
    if len(config.agents) != 2:
        raise ConfigError(f"the lens page needs two agents, the config has {len(config.agents)}")
    facts = frozen_noon_facts()
    p = config.params.lens
    points = list(itertools.product(HEAT_GRID, WATER_GRID, ENERGY_GRID, AROUSAL_GRID))
    agents = [
        {
            "name": agent.name,
            "setpoint": agent.temperament.setpoint.model_dump(),
            # heat, water, energy and arousal vary in that order, arousal fastest
            "entries": [
                _lens(agent, facts, Levels(heat=h, water=w, energy=e), a, p)
                for h, w, e, a in points
            ],
        }
        for agent in config.agents
    ]
    # the worked example is the config's start: its belief, arousal and voiced surprise
    worked = [_body(a, facts, a.belief, a.arousal, p, voiced=True) for a in config.agents]
    ordinary = [
        _body(a, facts, belief, arousal, p)
        for a, (belief, arousal) in zip(config.agents, ORDINARY_PAIR, strict=True)
    ]
    same = [_body(a, facts, *SAME_BODY, p) for a in config.agents]
    return {
        "place": {"name": PLACE.value, "opening": OPENING[PLACE]},
        "grid": {
            "heat": list(HEAT_GRID),
            "water": list(WATER_GRID),
            "energy": list(ENERGY_GRID),
            "arousal": list(AROUSAL_GRID),
        },
        "agents": agents,
        "presets": [
            {"name": "worked example", "bodies": worked},
            {"name": "ordinary pair", "bodies": ordinary},
            {"name": "same body, two temperaments", "bodies": same},
        ],
    }


def lens_page_html(config: ExperimentConfig) -> str:
    """One self-contained HTML page: the template with the lens data embedded as JSON."""
    template = (
        resources.files("surprise_loop.viewer")
        .joinpath("lens_page.html")
        .read_text(encoding="utf-8")
    )
    payload = json.dumps(lens_data(config), ensure_ascii=False, separators=(",", ":"))
    # nothing inside the script element may close it or open a comment
    payload = payload.replace("<", "\\u003c")
    return template.replace(PLACEHOLDER, payload, 1)
