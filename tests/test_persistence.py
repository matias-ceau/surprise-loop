import asyncio
import os
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from surprise_loop.config import (
    AgentSpec,
    Backends,
    Params,
    RunConfig,
    Runtime,
    initial_agent_state,
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
from surprise_loop.state import Conflict, GoalName, Mood, TickDump, Traits, Workspace
from surprise_loop.world import PhaseKind, PhaseSpec, SelfAdminWorld, WorldConfig

Phase = Callable[[Workspace, Runtime], Awaitable[Workspace]]

TRAITS = Traits(reward_sensitivity=1.2, surprise_learning_rate=0.4, mood_inertia=0.3)
PRIORITIES = {GoalName.SEEK_REWARD: 1.0, GoalName.EXPLORE: 0.5, GoalName.CONSERVE: 0.5}


def make_config(ticks: int | None = None, seed: int = 11, run_id: str = "persist") -> RunConfig:
    return RunConfig(
        run_id=run_id,
        seed=seed,
        ticks=ticks,
        world=WorldConfig(
            active_lever="random",
            reward_magnitude=1.0,
            schedule=[
                PhaseSpec(kind=PhaseKind.FIXED_RATIO, steps=30, ratio=2),
                PhaseSpec(kind=PhaseKind.PROGRESSIVE_RATIO, steps=30, pr_start=1, pr_increment=2),
                PhaseSpec(kind=PhaseKind.EXTINCTION, steps=30),
            ],
        ),
        agent=AgentSpec(
            name="a", traits=TRAITS, mood=Mood(valence=0.0, arousal=0.0), goal_priorities=PRIORITIES
        ),
        params=Params(),
        backends=Backends(),
        model=None,
    )


def every_file(run_dir: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(run_dir.iterdir())}


def run_files(run_dir: Path) -> dict[str, bytes]:
    """The bytes of `config.json` and every `tick_*.json`."""
    names = ["config.json", *sorted(p.name for p in run_dir.glob("tick_*.json"))]
    return {name: (run_dir / name).read_bytes() for name in names}


def conflict_every_other_tick() -> Phase:
    """The real Reconcile, plus a conflict on pass 0 of every even tick."""

    async def reconcile_with_conflict(ws: Workspace, rt: Runtime) -> Workspace:
        ws = await reconcile(ws, rt)
        if ws.pass_index != 0 or (ws.agent.tick + 1) % 2:
            return ws
        assert ws.decision is not None
        conflict = Conflict(kind="habit_vs_goal", action=ws.decision.action, detail="injected")
        last = ws.passes[-1].model_copy(update={"conflict": conflict})
        return ws.model_copy(update={"conflict": conflict, "passes": (*ws.passes[:-1], last)})

    return reconcile_with_conflict


CONFLICT_PHASES = _Phases(reconcile=conflict_every_other_tick())


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

    with pytest.raises(RunError):
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


def test_interrupted_and_resumed_run_is_byte_identical(tmp_path: Path) -> None:
    config = make_config()
    whole, split = tmp_path / "whole", tmp_path / "split"

    uninterrupted = asyncio.run(run(config, whole))
    first = asyncio.run(run(config, split, stop_after_tick=40))
    assert len(first.records) == 40
    assert len(list(split.glob("tick_*.json"))) == 40
    resumed = asyncio.run(run(config, split, resume=True))

    assert run_files(split) == run_files(whole)
    assert every_file(split) == every_file(whole)
    assert resumed == uninterrupted
    assert [r.tick for r in resumed.records] == list(range(1, len(uninterrupted.records) + 1))
    assert len(resumed.records) == 90


def test_resume_of_a_two_pass_run_is_byte_identical(tmp_path: Path) -> None:
    config = make_config()
    whole, split = tmp_path / "whole", tmp_path / "split"

    uninterrupted = asyncio.run(_run(config, whole, phases=CONFLICT_PHASES))
    asyncio.run(_run(config, split, stop_after_tick=17, phases=CONFLICT_PHASES))
    resumed = asyncio.run(_run(config, split, resume=True, phases=CONFLICT_PHASES))

    assert any(len(r.passes) == 2 for r in uninterrupted.records)
    assert run_files(split) == run_files(whole)
    assert resumed == uninterrupted


def test_resume_ignores_a_stray_temp_file(tmp_path: Path) -> None:
    config = make_config()
    whole, split = tmp_path / "whole", tmp_path / "split"
    asyncio.run(run(config, whole))
    asyncio.run(run(config, split, stop_after_tick=40))
    (split / "tick_000041.json.tmp").write_text('{"run_id": "persist", "ti')

    resumed = asyncio.run(run(config, split, resume=True))

    assert len(resumed.records) == 90
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


# --- load_result --------------------------------------------------------------------------------


def test_load_result_equals_the_in_memory_result(tmp_path: Path) -> None:
    result = asyncio.run(run(make_config(ticks=20), tmp_path))

    assert load_result(tmp_path) == result


def test_two_pass_ticks_survive_dump_and_load(tmp_path: Path) -> None:
    result = asyncio.run(_run(make_config(ticks=10), tmp_path, phases=CONFLICT_PHASES))

    loaded = load_result(tmp_path)

    assert [len(r.passes) for r in loaded.records] == [1, 2] * 5
    assert loaded.records[1].passes[0].conflict is not None
    assert loaded == result


def test_load_result_of_a_run_with_no_ticks(tmp_path: Path) -> None:
    config = make_config(ticks=5)
    asyncio.run(run(config, tmp_path, stop_after_tick=0))

    loaded = load_result(tmp_path)

    assert loaded == RunResult(
        config=config,
        records=[],
        agent=initial_agent_state(config),
        world=SelfAdminWorld(config.world, config.seed).snapshot(),
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


def test_replay_names_the_tick_whose_mood_was_edited(tmp_path: Path) -> None:
    asyncio.run(run(make_config(ticks=25), tmp_path))

    def bump_mood(dump: TickDump) -> TickDump:
        mood = Mood(valence=-dump.agent.mood.valence / 2 + 0.5, arousal=dump.agent.mood.arousal)
        return dump.model_copy(update={"agent": dump.agent.model_copy(update={"mood": mood})})

    edit_dump(tmp_path, 9, bump_mood)

    with pytest.raises(ReplayMismatch, match=r"tick 9\b") as caught:
        asyncio.run(replay(tmp_path))
    assert caught.value.tick == 9
    assert str(tmp_path) in str(caught.value)
    assert isinstance(caught.value, RunError)


def test_replay_passes_on_a_run_with_two_pass_ticks(tmp_path: Path) -> None:
    result = asyncio.run(_run(make_config(ticks=20), tmp_path, phases=CONFLICT_PHASES))
    assert any(len(r.passes) == 2 for r in result.records)

    assert asyncio.run(_replay(tmp_path, CONFLICT_PHASES)) == 20


def test_replay_of_a_two_pass_run_without_the_conflict_rule_fails(tmp_path: Path) -> None:
    asyncio.run(_run(make_config(ticks=20), tmp_path, phases=CONFLICT_PHASES))

    with pytest.raises(ReplayMismatch) as caught:
        asyncio.run(replay(tmp_path))
    assert caught.value.tick == 2


def test_replay_fails_when_a_stored_pass_is_altered(tmp_path: Path) -> None:
    asyncio.run(_run(make_config(ticks=10), tmp_path, phases=CONFLICT_PHASES))

    def alter(dump: TickDump) -> TickDump:
        second = dump.record.passes[1]
        error = second.prediction_error.model_copy(
            update={"signed": second.prediction_error.signed + 1}
        )
        passes = [dump.record.passes[0], second.model_copy(update={"prediction_error": error})]
        record = dump.record.model_copy(update={"passes": passes})
        return dump.model_copy(update={"record": record})

    edit_dump(tmp_path, 6, alter)

    with pytest.raises(ReplayMismatch) as caught:
        asyncio.run(_replay(tmp_path, CONFLICT_PHASES))
    assert caught.value.tick == 6


def test_replay_fails_when_the_record_lacks_a_pass_the_replay_needs(tmp_path: Path) -> None:
    asyncio.run(_run(make_config(ticks=10), tmp_path, phases=CONFLICT_PHASES))

    def drop_second_pass(dump: TickDump) -> TickDump:
        record = dump.record.model_copy(update={"passes": dump.record.passes[:1]})
        return dump.model_copy(update={"record": record})

    edit_dump(tmp_path, 4, drop_second_pass)

    with pytest.raises(ReplayMismatch, match=r"tick 4\b") as caught:
        asyncio.run(_replay(tmp_path, CONFLICT_PHASES))
    assert caught.value.tick == 4


def test_replay_fails_when_a_recorded_decision_is_altered(tmp_path: Path) -> None:
    asyncio.run(run(make_config(ticks=10), tmp_path))

    def other_prediction(dump: TickDump) -> TickDump:
        first = dump.record.passes[0]
        prediction = first.decision.prediction.model_copy(
            update={"expected_reward": first.decision.prediction.expected_reward + 0.25}
        )
        decision = first.decision.model_copy(update={"prediction": prediction})
        record = dump.record.model_copy(
            update={"passes": [first.model_copy(update={"decision": decision})]}
        )
        return dump.model_copy(update={"record": record})

    edit_dump(tmp_path, 5, other_prediction)

    # The replay feeds the edited decision back in, so the prediction error differs.
    with pytest.raises(ReplayMismatch) as caught:
        asyncio.run(replay(tmp_path))
    assert caught.value.tick == 5
