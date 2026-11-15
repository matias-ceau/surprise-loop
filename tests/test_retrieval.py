"""Salience-based retrieval (build step 7): config, default-run invariance and an end-to-end run.

Default-run invariance is checked in-test: the default config against an explicit
`retrieval: last_k` writes byte-identical files (no stored digests, which would be brittle across
platforms). tests/test_behaviour.py and the resume/replay tests pin the default behaviour itself.
"""

import asyncio
import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest
from pydantic import ValidationError

from surprise_loop.cli import main
from surprise_loop.config import MemoryParams, RunConfig, load_config
from surprise_loop.loop import load_dumps, run

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT / "experiments"


def select(file: str, agent: str, seed: int) -> RunConfig:
    runs = load_config(EXPERIMENTS / file).runs()
    return next(r for r in runs if r.agent.name == agent and r.seed == seed)


def with_retrieval(config: RunConfig, retrieval: str) -> RunConfig:
    params = config.params
    memory = params.memory.model_copy(update={"retrieval": retrieval})
    return config.model_copy(update={"params": params.model_copy(update={"memory": memory})})


def tick_files(run_dir: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(run_dir.glob("tick_*.json"))}


# --- config ---------------------------------------------------------------------------------


def test_retrieval_defaults_to_last_k() -> None:
    assert MemoryParams().retrieval == "last_k"


def test_retrieval_rejects_an_unknown_method() -> None:
    with pytest.raises(ValidationError):
        MemoryParams(retrieval="decay")  # type: ignore[arg-type]


def test_trait_grid_salient_is_trait_grid_with_salient_k() -> None:
    grid = load_config(EXPERIMENTS / "trait_grid.yaml")
    salient = load_config(EXPERIMENTS / "trait_grid_salient.yaml")
    assert grid.params.memory.retrieval == "last_k"
    assert salient.name == "trait_grid_salient"
    assert salient.params.memory.retrieval == "salient_k"
    params = salient.params.model_copy(
        update={"memory": salient.params.memory.model_copy(update={"retrieval": "last_k"})}
    )
    assert salient.model_copy(update={"name": grid.name, "params": params}) == grid


# --- the default run is unchanged -------------------------------------------------------------


def test_default_config_and_explicit_last_k_write_identical_files(tmp_path: Path) -> None:
    default = select("trait_grid.yaml", "surprise_learning_rate_low", 2)
    explicit = with_retrieval(default, "last_k")
    asyncio.run(run(default, tmp_path / "default"))
    asyncio.run(run(explicit, tmp_path / "explicit"))

    files = tick_files(tmp_path / "default")
    assert files
    assert files == tick_files(tmp_path / "explicit")
    assert (tmp_path / "default" / "config.json").read_bytes() == (
        tmp_path / "explicit" / "config.json"
    ).read_bytes()


# --- salient_k end to end -----------------------------------------------------------------------


def cli(*argv: str) -> str:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(list(argv))
    assert code == 0, err.getvalue()
    return out.getvalue()


def test_salient_k_run_through_the_cli_replays(tmp_path: Path) -> None:
    config_path = EXPERIMENTS / "trait_grid_salient.yaml"
    cli("run", str(config_path), "--agent", "baseline", "--seed", "1", "--runs-dir", str(tmp_path))
    run_dir = tmp_path / "trait_grid_salient-baseline-s1"

    dumps = load_dumps(run_dir)
    assert dumps and dumps[-1].world.step == 400
    assert '"retrieval":"salient_k"' in (run_dir / "config.json").read_text()
    assert f"{len(dumps)} ticks verified" in cli("replay", str(run_dir))

    # the same agent and seed under last_k behaves differently
    last_k = with_retrieval(select("trait_grid_salient.yaml", "baseline", 1), "last_k")
    asyncio.run(run(last_k, tmp_path / "last_k"))
    assert tick_files(run_dir) != tick_files(tmp_path / "last_k")
