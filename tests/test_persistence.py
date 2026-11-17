"""Tick files, resume and replay."""

import asyncio
import errno
import os
import pickle
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from surprise_loop.config import (
    OverrideParams,
    Params,
    RunConfig,
    Runtime,
    initial_agent_state,
    load_config,
)
from surprise_loop.loop import (
    ReplayMismatch,
    RunError,
    RunResult,
    _Phases,
    _replay,
    _run,
    load_dumps,
    load_result,
    replay,
    run,
    write_dump,
)
from surprise_loop.phases import reconcile
from surprise_loop.state import Band, Override, TickDump, Workspace
from surprise_loop.world import Action, Channel, ClearingWorld, Levels

Phase = Callable[[Workspace, Runtime], Awaitable[Workspace]]

ROOT = Path(__file__).resolve().parents[1]
AFTERNOON = ROOT / "experiments" / "clearing_afternoon.yaml"

# The real override rule off: every two-pass tick in the tests on OVERRIDE_PHASES comes from the
# injected override.
QUIET = Params(override=OverrideParams(enabled=False))
# Low enough that the real override rule fires often on the afternoon run.
URGENT = Params(override=OverrideParams(enabled=True, urgent_felt=0.5, worse_margin=0.0))
INJECTED = Override(channel=Channel.ENERGY, actions=[Action.REST], detail="injected")
CUT = 15
# with URGENT the afternoon's first fifteen ticks take two passes, so cut inside them
OVERRIDE_CUT = 10


def make_config(
    ticks: int | None = None, seed: int = 11, run_id: str = "persist", params: Params = QUIET
) -> RunConfig:
    """Ilo's afternoon run with the given tick cap, seed, run id and params."""
    return (
        load_config(AFTERNOON)
        .runs()[0]
        .model_copy(update={"ticks": ticks, "seed": seed, "run_id": run_id, "params": params})
    )


def every_file(run_dir: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(run_dir.iterdir())}


def run_files(run_dir: Path) -> dict[str, bytes]:
    """The bytes of `config.json` and every `tick_*.json`."""
    names = ["config.json", *sorted(p.name for p in run_dir.glob("tick_*.json"))]
    return {name: (run_dir / name).read_bytes() for name in names}


def override_every_other_tick() -> Phase:
    """The real Reconcile, plus the injected override on pass 0 of every even tick."""

    async def reconcile_with_override(ws: Workspace, rt: Runtime) -> Workspace:
        ws = await reconcile(ws, rt)
        if ws.pass_index != 0 or (ws.agent.tick + 1) % 2:
            return ws
        last = ws.passes[-1].model_copy(update={"override": INJECTED})
        return ws.model_copy(update={"override": INJECTED, "passes": (*ws.passes[:-1], last)})

    return reconcile_with_override


OVERRIDE_PHASES = _Phases(reconcile=override_every_other_tick())


def edit_dump(run_dir: Path, tick: int, edit: Callable[[TickDump], TickDump]) -> None:
    path = run_dir / f"tick_{tick:06d}.json"
    path.write_text(edit(TickDump.model_validate_json(path.read_text())).model_dump_json())


# --- dumps --------------------------------------------------------------------------------------


def test_run_writes_one_dump_per_tick_and_the_config(tmp_path: Path) -> None:
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
    assert all(d.agent.tick == d.tick for d in dumps)
    assert dumps[-1].agent == result.agent
    assert dumps[-1].world == result.world


def test_afternoon_run_writes_every_tick_and_stops_when_the_world_is_done(tmp_path: Path) -> None:
    config = load_config(AFTERNOON).runs()[0]

    result = asyncio.run(run(config, tmp_path))

    assert ClearingWorld.restore(config.world, result.world).done
    n = len(result.records)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["config.json", *[f"tick_{t:06d}.json" for t in range(1, n + 1)]]
    assert result.agent.tick == n
    assert result.world.step == sum(len(r.passes) for r in result.records)


def test_dump_file_is_the_compact_model_json(tmp_path: Path) -> None:
    asyncio.run(run(make_config(ticks=2), tmp_path))
    dump = load_dumps(tmp_path)[1]

    assert (tmp_path / "tick_000002.json").read_text() == dump.model_dump_json()


def test_every_pass_and_episode_in_a_tick_file_carries_its_number(tmp_path: Path) -> None:
    asyncio.run(run(make_config(params=URGENT), tmp_path))
    dumps = load_dumps(tmp_path)

    assert any(len(d.record.passes) == 2 for d in dumps)
    for dump in dumps:
        assert dump.record.tick == dump.agent.tick == dump.tick
        assert {p.tick for p in dump.record.passes} == {dump.tick}
        stored = dump.agent.memory.episodes[-len(dump.record.passes) :]
        assert [e.tick for e in stored] == [dump.tick] * len(dump.record.passes)
        assert [e.step for e in stored] == [p.outcome.info.step for p in dump.record.passes]


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


def test_load_dumps_rejects_a_gap(tmp_path: Path) -> None:
    asyncio.run(run(make_config(ticks=6), tmp_path))
    (tmp_path / "tick_000004.json").unlink()

    with pytest.raises(RunError, match="tick"):
        load_dumps(tmp_path)


def test_load_dumps_rejects_a_sequence_not_starting_at_one(tmp_path: Path) -> None:
    asyncio.run(run(make_config(ticks=3), tmp_path))
    (tmp_path / "tick_000001.json").unlink()

    with pytest.raises(RunError):
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


def test_empty_existing_directory_starts_fresh(tmp_path: Path) -> None:
    result = asyncio.run(run(make_config(ticks=3), tmp_path))

    assert len(result.records) == 3
    assert (tmp_path / "tick_000003.json").exists()


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


def test_interrupted_and_resumed_run_with_real_overrides_is_byte_identical(tmp_path: Path) -> None:
    config = make_config(params=URGENT)
    whole, split = tmp_path / "whole", tmp_path / "split"

    uninterrupted = asyncio.run(run(config, whole))
    first = asyncio.run(run(config, split, stop_after_tick=OVERRIDE_CUT))
    assert len(first.records) == OVERRIDE_CUT
    assert len(list(split.glob("tick_*.json"))) == OVERRIDE_CUT
    resumed = asyncio.run(run(config, split, resume=True))

    # The override fired, and on both sides of the cut.
    two_pass = [r.tick for r in uninterrupted.records if len(r.passes) == 2]
    assert any(t <= OVERRIDE_CUT for t in two_pass)
    assert any(t > OVERRIDE_CUT for t in two_pass)
    overrides = [r.passes[0].override for r in uninterrupted.records if r.tick in two_pass]
    assert None not in overrides
    assert run_files(split) == run_files(whole)
    assert every_file(split) == every_file(whole)
    assert resumed == uninterrupted
    assert [r.tick for r in resumed.records] == list(range(1, len(uninterrupted.records) + 1))
    assert len(resumed.records) > OVERRIDE_CUT
    # The run went to the end of the day: every world step was taken.
    w = config.world
    steps = (w.end_minute - w.start_minute) // w.tick_minutes
    assert sum(len(r.passes) for r in resumed.records) == steps


@pytest.mark.parametrize("stop", [17, 18])  # after a one-pass and after a two-pass tick
def test_resume_of_a_two_pass_run_is_byte_identical(tmp_path: Path, stop: int) -> None:
    config = make_config()
    whole, split = tmp_path / "whole", tmp_path / "split"

    uninterrupted = asyncio.run(_run(config, whole, phases=OVERRIDE_PHASES))
    asyncio.run(_run(config, split, stop_after_tick=stop, phases=OVERRIDE_PHASES))
    resumed = asyncio.run(_run(config, split, resume=True, phases=OVERRIDE_PHASES))

    assert any(len(r.passes) == 2 for r in uninterrupted.records)
    assert len(uninterrupted.records) > stop
    assert run_files(split) == run_files(whole)
    assert resumed == uninterrupted


def test_resume_ignores_a_stray_temp_file(tmp_path: Path) -> None:
    config = make_config()
    whole, split = tmp_path / "whole", tmp_path / "split"
    uninterrupted = asyncio.run(run(config, whole))
    asyncio.run(run(config, split, stop_after_tick=CUT))
    (split / f"tick_{CUT + 1:06d}.json.tmp").write_text('{"run_id": "persist", "ti')

    resumed = asyncio.run(run(config, split, resume=True))

    assert len(resumed.records) == len(uninterrupted.records) > CUT
    assert run_files(split) == run_files(whole)


def test_resume_with_a_changed_config_raises(tmp_path: Path) -> None:
    asyncio.run(run(make_config(), tmp_path, stop_after_tick=5))
    before = every_file(tmp_path)

    with pytest.raises(RunError, match="config"):
        asyncio.run(run(make_config(seed=12), tmp_path, resume=True))

    assert every_file(tmp_path) == before


def test_resume_without_config_json_raises(tmp_path: Path) -> None:
    asyncio.run(run(make_config(), tmp_path, stop_after_tick=5))
    (tmp_path / "config.json").unlink()

    with pytest.raises(RunError, match="config.json"):
        asyncio.run(run(make_config(), tmp_path, resume=True))


def test_resume_with_a_gap_in_the_dumps_raises(tmp_path: Path) -> None:
    asyncio.run(run(make_config(), tmp_path, stop_after_tick=5))
    (tmp_path / "tick_000002.json").unlink()

    with pytest.raises(RunError):
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
                    update={"temperament": d.agent.temperament.model_copy(update={"baseline": 0.5})}
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

    assert load_result(tmp_path) == result


def test_two_pass_ticks_survive_dump_and_load(tmp_path: Path) -> None:
    result = asyncio.run(_run(make_config(ticks=10), tmp_path, phases=OVERRIDE_PHASES))

    loaded = load_result(tmp_path)

    assert [len(r.passes) for r in loaded.records] == [1, 2] * 5
    assert loaded.records[1].passes[0].override == INJECTED
    assert loaded == result


def test_load_result_of_a_run_with_no_ticks(tmp_path: Path) -> None:
    config = make_config(ticks=5)
    asyncio.run(run(config, tmp_path, stop_after_tick=0))

    loaded = load_result(tmp_path)

    world = ClearingWorld(config.world, config.seed, config.agent.body, config.start_place)
    assert loaded == RunResult(
        config=config, records=[], agent=initial_agent_state(config), world=world.snapshot()
    )


def test_load_result_without_config_json_raises(tmp_path: Path) -> None:
    with pytest.raises(RunError, match="config.json"):
        load_result(tmp_path)


# --- replay -------------------------------------------------------------------------------------


def test_replay_verifies_every_tick_and_writes_nothing(tmp_path: Path) -> None:
    asyncio.run(run(make_config(ticks=25), tmp_path))
    before = every_file(tmp_path)

    assert asyncio.run(replay(tmp_path)) == 25
    assert every_file(tmp_path) == before


def test_replay_names_the_tick_whose_belief_was_edited(tmp_path: Path) -> None:
    asyncio.run(run(make_config(ticks=25), tmp_path))

    def shift_belief(dump: TickDump) -> TickDump:
        b = dump.agent.belief
        belief = Levels(heat=b.heat + 0.1, water=b.water, energy=b.energy)
        return dump.model_copy(update={"agent": dump.agent.model_copy(update={"belief": belief})})

    edit_dump(tmp_path, 9, shift_belief)

    with pytest.raises(ReplayMismatch, match=r"tick 9\b") as caught:
        asyncio.run(replay(tmp_path))
    assert caught.value.tick == 9
    assert str(tmp_path) in str(caught.value)
    assert isinstance(caught.value, RunError)


def test_replay_passes_on_a_run_with_two_pass_ticks(tmp_path: Path) -> None:
    result = asyncio.run(_run(make_config(ticks=20), tmp_path, phases=OVERRIDE_PHASES))
    assert any(len(r.passes) == 2 for r in result.records)

    assert asyncio.run(_replay(tmp_path, OVERRIDE_PHASES)) == 20


def test_replay_passes_on_a_run_with_real_overrides(tmp_path: Path) -> None:
    result = asyncio.run(run(make_config(params=URGENT), tmp_path))
    assert any(len(r.passes) == 2 for r in result.records)

    assert asyncio.run(replay(tmp_path)) == len(result.records)


def test_replay_of_a_two_pass_run_without_the_override_fails(tmp_path: Path) -> None:
    asyncio.run(_run(make_config(ticks=20), tmp_path, phases=OVERRIDE_PHASES))

    with pytest.raises(ReplayMismatch) as caught:
        asyncio.run(replay(tmp_path))
    assert caught.value.tick == 2


def test_replay_fails_when_a_stored_pass_is_altered(tmp_path: Path) -> None:
    asyncio.run(_run(make_config(ticks=10), tmp_path, phases=OVERRIDE_PHASES))

    def alter(dump: TickDump) -> TickDump:
        second = dump.record.passes[1]
        comparison = second.comparison.model_copy(update={"size": second.comparison.size + 1})
        passes = [dump.record.passes[0], second.model_copy(update={"comparison": comparison})]
        record = dump.record.model_copy(update={"passes": passes})
        return dump.model_copy(update={"record": record})

    edit_dump(tmp_path, 6, alter)

    with pytest.raises(ReplayMismatch) as caught:
        asyncio.run(_replay(tmp_path, OVERRIDE_PHASES))
    assert caught.value.tick == 6


def test_replay_fails_when_the_record_lacks_a_pass_the_replay_needs(tmp_path: Path) -> None:
    asyncio.run(_run(make_config(ticks=10), tmp_path, phases=OVERRIDE_PHASES))

    def drop_second_pass(dump: TickDump) -> TickDump:
        record = dump.record.model_copy(update={"passes": dump.record.passes[:1]})
        return dump.model_copy(update={"record": record})

    edit_dump(tmp_path, 4, drop_second_pass)

    with pytest.raises(ReplayMismatch, match=r"tick 4\b") as caught:
        asyncio.run(_replay(tmp_path, OVERRIDE_PHASES))
    assert caught.value.tick == 4


def test_replay_fails_when_a_recorded_decision_is_altered(tmp_path: Path) -> None:
    asyncio.run(run(make_config(ticks=10), tmp_path))

    def other_expectation(dump: TickDump) -> TickDump:
        first = dump.record.passes[0]
        expectation = first.decision.expectation
        bands = {c: Band(lo=None, hi=None) for c in expectation.bands}
        decision = first.decision.model_copy(
            update={"expectation": expectation.model_copy(update={"bands": bands})}
        )
        record = dump.record.model_copy(
            update={"passes": [first.model_copy(update={"decision": decision})]}
        )
        return dump.model_copy(update={"record": record})

    edit_dump(tmp_path, 5, other_expectation)

    # The replay feeds the edited decision back in, so the comparison differs.
    with pytest.raises(ReplayMismatch) as caught:
        asyncio.run(replay(tmp_path))
    assert caught.value.tick == 5


def test_replay_runs_evaluate_as_code(tmp_path: Path) -> None:
    asyncio.run(run(make_config(ticks=10), tmp_path))
    dumps = load_dumps(tmp_path)
    judged = next(d.tick for d in dumps if d.record.passes[0].feedback.score is not None)

    def other_feedback(dump: TickDump) -> TickDump:
        first = dump.record.passes[0]
        feedback = first.feedback.model_copy(update={"line": "edited"})
        record = dump.record.model_copy(
            update={"passes": [first.model_copy(update={"feedback": feedback})]}
        )
        return dump.model_copy(update={"record": record})

    edit_dump(tmp_path, judged, other_feedback)

    # The recorded feedback is not fed back in: Evaluate recomputes it and the record differs.
    with pytest.raises(ReplayMismatch, match="record") as caught:
        asyncio.run(replay(tmp_path))
    assert caught.value.tick == judged


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
