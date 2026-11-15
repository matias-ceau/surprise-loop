"""Salience-based retrieval (build step 7): config, default-run invariance and an end-to-end run.

Default-run invariance is checked two ways:
- against SHA-256 digests of the tick dumps (and config.json) of two runs made at commit 768609a,
  before `memory.retrieval` existed, so any change to the default path's bytes fails here;
- in-test, the default config against an explicit `retrieval: last_k`, which does not depend on
  stored digests (and so would survive a platform whose float output differed from the stored one).
"""

import asyncio
import hashlib
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

# (config file, agent, seed) -> digests of the run's files at commit 768609a (default config).
# `ticks` hashes every tick_*.json name and bytes in order; `config` hashes config.json.
REFERENCE = {
    ("stub_smoke.yaml", "baseline", 1): {
        "ticks": "eabb12e51a405e9da2c7c7760bb804b3527ccf0b0f2cfe5b546807eec29cd1ab",
        "config": "3550beadbecea6eb321b9e56427ba5997c3d09dd85c5df5a4639e741cffd3198",
    },
    ("trait_grid.yaml", "mood_inertia_high", 3): {
        "ticks": "4d4b013db6df5a8958c967b5829782a8b705175013c8d0e4cfa48ac8c51d9d67",
        "config": "957cbc681319c9acff78fa1d7f875e3e944a892e80ab2431516dd197a6187adf",
    },
}
# the one byte change to config.json: the new key, serialized after retrieval_k
NEW_CONFIG_KEY = b'"retrieval":"last_k",'


def select(file: str, agent: str, seed: int) -> RunConfig:
    runs = load_config(EXPERIMENTS / file).runs()
    return next(r for r in runs if r.agent.name == agent and r.seed == seed)


def with_retrieval(config: RunConfig, retrieval: str) -> RunConfig:
    params = config.params
    memory = params.memory.model_copy(update={"retrieval": retrieval})
    return config.model_copy(update={"params": params.model_copy(update={"memory": memory})})


def ticks_digest(run_dir: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(run_dir.glob("tick_*.json")):
        h.update(path.name.encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


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


@pytest.mark.parametrize("key", list(REFERENCE), ids=lambda k: f"{k[0]}-{k[1]}-s{k[2]}")
def test_default_run_dumps_match_the_pre_retrieval_reference(
    key: tuple[str, str, int], tmp_path: Path
) -> None:
    config = select(*key)
    assert config.params.memory.retrieval == "last_k"
    run_dir = tmp_path / config.run_id
    asyncio.run(run(config, run_dir))

    assert ticks_digest(run_dir) == REFERENCE[key]["ticks"]
    stored = (run_dir / "config.json").read_bytes()
    assert stored.count(NEW_CONFIG_KEY) == 1
    old = hashlib.sha256(stored.replace(NEW_CONFIG_KEY, b"")).hexdigest()
    assert old == REFERENCE[key]["config"]


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
