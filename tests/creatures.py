"""What later tests share: the two-maps layout, its experiment file, and one run's config."""

from pathlib import Path
from typing import Any

from surprise_loop.config import ExperimentConfig, RunConfig, load_config

TWO_MAPS = Path(__file__).resolve().parents[1] / "experiments" / "two_maps.yaml"

LAYOUT = (
    "...#..~..",
    ".F.#...T.",
    ".....P...",
    "##..@..U.",
    "...T..F..",
    ".~...#..B",
    "....F#...",
)


def load_two_maps() -> ExperimentConfig:
    return load_config(TWO_MAPS)


def run_config(agent: str = "plain", seed: int = 1, **changes: Any) -> RunConfig:
    """The run of `agent` on `seed` in the two-maps experiment, with `changes` applied."""
    for run in load_two_maps().runs():
        if run.agent.name == agent and run.seed == seed:
            return run.model_copy(update=changes)
    raise KeyError(f"no run for agent {agent!r} on seed {seed}")
