"""Tick files, resume and replay."""

import asyncio
import errno
import os
import pickle
from collections.abc import Callable
from pathlib import Path

import pytest
from creatures import run_config

from surprise_loop.config import RunConfig, initial_agent_state
from surprise_loop.loop import (
    ReplayMismatch,
    RunError,
    RunResult,
    is_complete,
    load_dumps,
    load_result,
    replay,
    run,
    write_dump,
)
from surprise_loop.state import TickDump
from surprise_loop.world import Belief, GridWorld

CUT = 40


def make_config(
    agent: str = "plain", ticks: int | None = None, seed: int = 1, run_id: str = "persist"
) -> RunConfig:
    """The two-maps run of `agent` with the given tick cap, seed and run id."""
    return run_config(agent, seed, ticks=ticks, run_id=run_id)


def every_file(run_dir: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(run_dir.iterdir())}


def run_files(run_dir: Path) -> dict[str, bytes]:
    """The bytes of `config.json` and every `tick_*.json`."""
    names = ["config.json", *sorted(p.name for p in run_dir.glob("tick_*.json"))]
    return {name: (run_dir / name).read_bytes() for name in names}


def edit_dump(run_dir: Path, tick: int, edit: Callable[[TickDump], TickDump]) -> None:
    path = run_dir / f"tick_{tick:06d}.json"
    path.write_text(edit(TickDump.model_validate_json(path.read_text())).model_dump_json())


# --- dumps --------------------------------------------------------------------------------------


def test_run_writes_the_config_and_one_dump_per_tick(tmp_path: Path) -> None:
    config = make_config(ticks=12)
    run_dir = tmp_path / "nested" / "run"

    result = asyncio.run(run(config, run_dir))

    names = sorted(p.name for p in run_dir.iterdir())
    assert names == ["config.json", *[f"tick_{t:06d}.json" for t in range(1, 13)]]
    assert RunConfig.model_validate_json((run_dir / "config.json").read_text()) == config
    dumps = load_dumps(run_dir)
    assert [d.tick for d in dumps] == list(range(1, 13))
    assert [d.record for d in dumps] == result.records
    assert all(d.run_id == config.run_id for d in dumps)
    assert all(d.agent.tick == d.record.tick == d.tick for d in dumps)
    assert dumps[-1].agent == result.agent
    assert dumps[-1].world == result.world


def test_dump_file_is_the_compact_model_json(tmp_path: Path) -> None:
    asyncio.run(run(make_config(ticks=2), tmp_path))
    dump = load_dumps(tmp_path)[1]

    assert (tmp_path / "tick_000002.json").read_text() == dump.model_dump_json()


def test_write_dump_is_atomic_and_leaves_no_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    asyncio.run(run(make_config(ticks=1), source))
    dump = load_dumps(source)[0]
    target = tmp_path / "target"
    target.mkdir()
    renames: list[tuple[Path, Path, bytes, bool]] = []
    real_replace = os.replace

    def spy_replace(src: str | Path, dst: str | Path) -> None:
        src, dst = Path(src), Path(dst)
        # At the moment of the rename the temp file is complete and the target absent.
        renames.append((src, dst, src.read_bytes(), dst.exists()))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy_replace)

    path = write_dump(target, dump)

    assert path == target / "tick_000001.json"
    assert len(renames) == 1
    src, dst, content, existed = renames[0]
    assert src.parent == target and src != dst and not src.name.endswith(".json")
    assert dst == path and not existed
    assert content == dump.model_dump_json().encode()
    assert [p.name for p in target.iterdir()] == ["tick_000001.json"]
    assert path.read_bytes() == (source / "tick_000001.json").read_bytes()


def test_config_json_is_written_atomically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    renamed: list[str] = []
    real_replace = os.replace

    def spy_replace(src: str | Path, dst: str | Path) -> None:
        renamed.append(Path(dst).name)
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy_replace)

    asyncio.run(run(make_config(ticks=2), tmp_path))

    assert renamed == ["config.json", "tick_000001.json", "tick_000002.json"]


def test_load_dumps_sorts_by_tick_and_ignores_other_files(tmp_path: Path) -> None:
    asyncio.run(run(make_config(ticks=12), tmp_path))
    (tmp_path / "tick_000013.json.tmp").write_text("{partial")
    (tmp_path / "notes.txt").write_text("hello")

    assert [d.tick for d in load_dumps(tmp_path)] == list(range(1, 13))


def test_load_dumps_rejects_a_gap_naming_the_missing_tick(tmp_path: Path) -> None:
    asyncio.run(run(make_config(ticks=6), tmp_path))
    (tmp_path / "tick_000004.json").unlink()

    with pytest.raises(RunError, match="missing tick 4"):
        load_dumps(tmp_path)


def test_load_dumps_rejects_a_sequence_not_starting_at_one(tmp_path: Path) -> None:
    asyncio.run(run(make_config(ticks=3), tmp_path))
    (tmp_path / "tick_000001.json").unlink()

    with pytest.raises(RunError, match="missing tick 1"):
        load_dumps(tmp_path)


def test_load_dumps_rejects_an_unreadable_file(tmp_path: Path) -> None:
    asyncio.run(run(make_config(ticks=4), tmp_path))
    (tmp_path / "tick_000003.json").write_text('{"run_id": "persist", "tick": 3')

    with pytest.raises(RunError, match="tick_000003.json"):
        load_dumps(tmp_path)


def test_load_dumps_rejects_a_file_whose_name_does_not_match_its_tick(tmp_path: Path) -> None:
    asyncio.run(run(make_config(ticks=2), tmp_path))
    (tmp_path / "tick_000002.json").rename(tmp_path / "tick_2.json")

    with pytest.raises(RunError, match="tick_2.json"):
        load_dumps(tmp_path)


# --- start and resume ---------------------------------------------------------------------------


def test_non_empty_directory_without_resume_raises(tmp_path: Path) -> None:
    (tmp_path / "something.txt").write_text("x")

    with pytest.raises(RunError, match=str(tmp_path)):
        asyncio.run(run(make_config(ticks=3), tmp_path))

    assert [p.name for p in tmp_path.iterdir()] == ["something.txt"]


def test_existing_run_without_resume_raises(tmp_path: Path) -> None:
    asyncio.run(run(make_config(ticks=3), tmp_path))
    before = every_file(tmp_path)

    with pytest.raises(RunError, match="--resume"):
        asyncio.run(run(make_config(ticks=3), tmp_path))

    assert every_file(tmp_path) == before


@pytest.mark.parametrize("exists", [True, False])
def test_resume_on_empty_or_missing_directory_starts_fresh(tmp_path: Path, exists: bool) -> None:
    run_dir = tmp_path / "run"
    if exists:
        run_dir.mkdir()
    reference = tmp_path / "reference"
    asyncio.run(run(make_config(ticks=5), reference))

    result = asyncio.run(run(make_config(ticks=5), run_dir, resume=True))

    assert len(result.records) == 5
    assert run_files(run_dir) == run_files(reference)


@pytest.mark.parametrize("agent", ["plain", "wary"])
def test_a_stopped_and_resumed_run_is_byte_identical(tmp_path: Path, agent: str) -> None:
    config = make_config(agent)
    whole, split = tmp_path / "whole", tmp_path / "split"

    uninterrupted = asyncio.run(run(config, whole))
    first = asyncio.run(run(config, split, stop_after_tick=CUT))
    assert len(first.records) == CUT
    assert len(list(split.glob("tick_*.json"))) == CUT
    resumed = asyncio.run(run(config, split, resume=True))

    assert every_file(split) == every_file(whole)
    assert resumed == uninterrupted
    assert [r.tick for r in resumed.records] == list(range(1, config.world.steps + 1))


def test_resume_ignores_a_stray_temp_file(tmp_path: Path) -> None:
    config = make_config(ticks=CUT + 5)
    whole, split = tmp_path / "whole", tmp_path / "split"
    uninterrupted = asyncio.run(run(config, whole))
    asyncio.run(run(config, split, stop_after_tick=CUT))
    (split / f"tick_{CUT + 1:06d}.json.tmp").write_text('{"run_id": "persist", "ti')

    resumed = asyncio.run(run(config, split, resume=True))

    assert resumed == uninterrupted
    assert run_files(split) == run_files(whole)


def test_resume_with_a_changed_config_raises(tmp_path: Path) -> None:
    asyncio.run(run(make_config(), tmp_path, stop_after_tick=5))
    before = every_file(tmp_path)

    with pytest.raises(RunError, match="config"):
        asyncio.run(run(make_config(seed=2), tmp_path, resume=True))

    assert every_file(tmp_path) == before


def test_resume_without_config_json_raises(tmp_path: Path) -> None:
    asyncio.run(run(make_config(), tmp_path, stop_after_tick=5))
    (tmp_path / "config.json").unlink()

    with pytest.raises(RunError, match="config.json"):
        asyncio.run(run(make_config(), tmp_path, resume=True))


def test_resume_with_a_missing_tick_file_raises_naming_the_tick(tmp_path: Path) -> None:
    asyncio.run(run(make_config(), tmp_path, stop_after_tick=5))
    (tmp_path / "tick_000002.json").unlink()

    with pytest.raises(RunError, match="missing tick 2"):
        asyncio.run(run(make_config(), tmp_path, resume=True))


def test_resume_of_a_finished_run_returns_it_unchanged(tmp_path: Path) -> None:
    config = make_config(ticks=8)
    done = asyncio.run(run(config, tmp_path))
    before = every_file(tmp_path)

    again = asyncio.run(run(config, tmp_path, resume=True))

    assert again == done
    assert every_file(tmp_path) == before


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d.model_copy(update={"run_id": "other"}),
        lambda d: d.model_copy(
            update={
                "agent": d.agent.model_copy(
                    update={"temperament": d.agent.temperament.model_copy(update={"blur": 0.5})}
                )
            }
        ),
    ],
    ids=["run_id", "temperament"],
)
def test_resume_with_dumps_of_another_run_raises(
    tmp_path: Path, edit: Callable[[TickDump], TickDump]
) -> None:
    asyncio.run(run(make_config(), tmp_path, stop_after_tick=5))
    edit_dump(tmp_path, 5, edit)
    before = every_file(tmp_path)

    with pytest.raises(RunError, match="do not belong"):
        asyncio.run(run(make_config(), tmp_path, resume=True))

    assert every_file(tmp_path) == before


def test_run_dir_that_is_a_file_raises(tmp_path: Path) -> None:
    (tmp_path / "file").write_text("x")
    with pytest.raises(RunError, match="not a directory"):
        asyncio.run(run(make_config(), tmp_path / "file"))


# --- load_result --------------------------------------------------------------------------------


def test_load_result_equals_the_in_memory_result(tmp_path: Path) -> None:
    result = asyncio.run(run(make_config(ticks=20), tmp_path))

    loaded = load_result(tmp_path)

    assert loaded == result
    assert is_complete(loaded)


def test_load_result_of_a_run_with_no_ticks(tmp_path: Path) -> None:
    config = make_config(ticks=5)
    asyncio.run(run(config, tmp_path, stop_after_tick=0))

    loaded = load_result(tmp_path)

    world = GridWorld(config.world, config.seed)
    assert loaded == RunResult(
        config=config,
        records=[],
        agent=initial_agent_state(config, world.observe()),
        world=world.snapshot(),
    )
    assert not is_complete(loaded)


def test_load_result_without_config_json_raises(tmp_path: Path) -> None:
    with pytest.raises(RunError, match="config.json"):
        load_result(tmp_path)


# --- replay -------------------------------------------------------------------------------------


def test_replay_verifies_every_tick_and_writes_nothing(tmp_path: Path) -> None:
    asyncio.run(run(make_config(ticks=25), tmp_path))
    before = every_file(tmp_path)

    assert asyncio.run(replay(tmp_path)) == 25
    assert every_file(tmp_path) == before


def test_replay_names_the_tick_whose_own_map_was_edited(tmp_path: Path) -> None:
    asyncio.run(run(make_config(ticks=25), tmp_path))

    def shift_untried(dump: TickDump) -> TickDump:
        own = dump.agent.own_map
        untried = own.untried | {Belief.HURTS: 1.0 - own.untried[Belief.HURTS]}
        own_map = own.model_copy(update={"untried": untried})
        return dump.model_copy(update={"agent": dump.agent.model_copy(update={"own_map": own_map})})

    edit_dump(tmp_path, 9, shift_untried)

    with pytest.raises(ReplayMismatch, match=r"tick 9\b") as caught:
        asyncio.run(replay(tmp_path))
    assert caught.value.tick == 9
    assert str(tmp_path) in str(caught.value)
    assert isinstance(caught.value, RunError)


def test_replay_feeds_back_the_recorded_decision(tmp_path: Path) -> None:
    asyncio.run(run(make_config(ticks=10), tmp_path))

    def other_why(dump: TickDump) -> TickDump:
        decision = dump.record.decision.model_copy(update={"why": "edited"})
        record = dump.record.model_copy(update={"decision": decision})
        return dump.model_copy(update={"record": record})

    edit_dump(tmp_path, 5, other_why)

    # The replay returns the edited decision, so the rebuilt record carries it: the edit passes.
    assert asyncio.run(replay(tmp_path)) == 10


def test_replay_runs_monitor_as_code(tmp_path: Path) -> None:
    asyncio.run(run(make_config(ticks=10), tmp_path))

    def other_comparison(dump: TickDump) -> TickDump:
        comparison = dump.record.comparison.model_copy(update={"worse": 9.0})
        record = dump.record.model_copy(update={"comparison": comparison})
        return dump.model_copy(update={"record": record})

    edit_dump(tmp_path, 6, other_comparison)

    # The recorded comparison is not fed back in: Monitor recomputes it and the record differs.
    with pytest.raises(ReplayMismatch, match="record") as caught:
        asyncio.run(replay(tmp_path))
    assert caught.value.tick == 6


def test_replay_mismatch_pickles() -> None:
    error = ReplayMismatch(Path("runs/x"), 7, "mismatch in agent")
    copy = pickle.loads(pickle.dumps(error))
    assert type(copy) is ReplayMismatch
    assert (copy.tick, copy.run_dir, str(copy)) == (7, Path("runs/x"), str(error))


# --- write errors -------------------------------------------------------------------------------


def fail_replace_of(name: str, error: OSError) -> Callable[[str | Path, str | Path], None]:
    """`os.replace` that raises `error` when renaming into a file called `name`."""
    real_replace = os.replace

    def replace(src: str | Path, dst: str | Path) -> None:
        if Path(dst).name == name:
            raise error
        real_replace(src, dst)

    return replace


@pytest.mark.parametrize(
    ("name", "error", "reason"),
    [
        (
            "tick_000003.json",
            OSError(errno.ENOSPC, "No space left on device"),
            "No space left on device",
        ),
        ("config.json", OSError(errno.EACCES, "Permission denied"), "Permission denied"),
        ("tick_000001.json", OSError("disk on fire"), "disk on fire"),  # no strerror
    ],
)
def test_a_run_file_that_cannot_be_written_is_a_run_error_naming_it(
    name: str, error: OSError, reason: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "replace", fail_replace_of(name, error))
    run_dir = tmp_path / "run"
    with pytest.raises(RunError) as info:
        asyncio.run(run(make_config(ticks=5), run_dir))
    message = str(info.value)
    assert message == f"{run_dir / name}: cannot write: {reason}"
    assert "\n" not in message
