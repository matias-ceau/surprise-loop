"""`surprise-loop experiment`: concurrent runs, skip and resume, failures, and the CSV tables."""

import asyncio
import csv
import errno
import io
import math
import os
import shutil
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from surprise_loop import cli
from surprise_loop.config import load_config
from surprise_loop.loop import RunResult, is_complete, load_result, run
from surprise_loop.metrics import (
    STABILITY_METRICS,
    TRACE_COLUMNS,
    RunMetrics,
    experiment_tables,
    flatten,
    format_cell,
    icc1,
    results_csv,
    stability_csv,
    summarize,
    trace_csv,
)
from surprise_loop.world import Channel

ROOT = Path(__file__).resolve().parent.parent
AFTERNOON = ROOT / "experiments" / "clearing_afternoon.yaml"
CONFIG = (
    AFTERNOON.read_text()
    .replace("name: clearing_afternoon", "name: exp")
    .replace("seeds: [1]", "seeds: [1, 2]")
)
RUN_IDS = ["exp-ilo-s1", "exp-ilo-s2", "exp-tavi-s1", "exp-tavi-s2"]
CSV_FILES = ["results.csv", "stability.csv", "trace.csv"]
TICKS = 42


def write_config(tmp_path: Path, text: str = CONFIG, **replacements: str) -> Path:
    for old, new in replacements.items():
        assert old in text
        text = text.replace(old, new)
    path = tmp_path / "exp.yaml"
    path.write_text(text)
    return path


def experiment(
    config: Path, runs_dir: Path, capsys: pytest.CaptureFixture[str], *extra: str
) -> tuple[int, str]:
    """Run the command; return its exit code and stderr (stdout must stay empty)."""
    code = cli.main(["experiment", str(config), "--runs-dir", str(runs_dir), *extra])
    out, err = capsys.readouterr()
    assert out == ""
    return code, err


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def header(path: Path) -> list[str]:
    return path.read_text().splitlines()[0].split(",")


def csv_bytes(runs_dir: Path) -> dict[str, bytes]:
    return {name: (runs_dir / "exp" / name).read_bytes() for name in CSV_FILES}


def dump_bytes(runs_dir: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(runs_dir)): p.read_bytes()
        for run_id in RUN_IDS
        for p in sorted((runs_dir / run_id).iterdir())
    }


def mtimes(runs_dir: Path) -> dict[str, int]:
    return {
        str(p.relative_to(runs_dir)): p.stat().st_mtime_ns
        for run_id in RUN_IDS
        for p in sorted((runs_dir / run_id).iterdir())
    }


def error_lines(err: str) -> list[str]:
    return [line for line in err.splitlines() if line.startswith("error: ")]


@pytest.fixture(scope="module")
def reference(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One uninterrupted --jobs 1 run of CONFIG; tests compare against its files."""
    tmp = tmp_path_factory.mktemp("reference")
    runs_dir = tmp / "runs"
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli.main(["experiment", str(write_config(tmp)), "--runs-dir", str(runs_dir)])
    assert code == 0, err.getvalue()
    assert out.getvalue() == ""
    return runs_dir


# --- files and columns ----------------------------------------------------------------------


def test_writes_one_run_dir_per_agent_and_seed_and_the_three_csvs(reference: Path) -> None:
    assert sorted(p.name for p in reference.iterdir()) == sorted([*RUN_IDS, "exp"])
    assert sorted(p.name for p in (reference / "exp").iterdir()) == CSV_FILES
    for run_id in RUN_IDS:
        result = load_result(reference / run_id)
        assert len(result.records) == TICKS and is_complete(result)


def test_results_csv_has_one_row_per_run_in_config_order(reference: Path) -> None:
    path = reference / "exp" / "results.csv"
    assert header(path) == ["agent", "seed", *RunMetrics.model_fields]
    rows = read_csv(path)
    assert [(r["agent"], r["seed"]) for r in rows] == [
        ("ilo", "1"),
        ("ilo", "2"),
        ("tavi", "1"),
        ("tavi", "2"),
    ]
    for row, run_id in zip(rows, RUN_IDS, strict=True):
        metrics = summarize(load_result(reference / run_id))
        for field, value in metrics.model_dump().items():
            assert row[field] == format_cell(value), (run_id, field)
        assert row["truncated"] == "False"
        assert row["ticks"] == str(TICKS)


def test_trace_csv_has_one_row_per_world_step(reference: Path) -> None:
    path = reference / "exp" / "trace.csv"
    assert header(path) == [
        "agent",
        "seed",
        "step",
        "minute",
        "place",
        "action",
        "event",
        "error_heat",
        "error_water",
        "error_energy",
        "world",
        "better",
        "size",
        "belief_heat",
        "belief_water",
        "belief_energy",
        "body_heat",
        "body_water",
        "body_energy",
        "valence",
        "arousal",
    ]
    assert header(path) == list(TRACE_COLUMNS)
    rows = read_csv(path)
    results = {(r.split("-")[1], r[-1]): load_result(reference / r) for r in RUN_IDS}
    assert len(rows) == sum(len(flatten(r.records)) for r in results.values())
    for (agent, seed), result in results.items():
        mine = [r for r in rows if (r["agent"], r["seed"]) == (agent, seed)]
        passes = flatten(result.records)
        assert [int(r["step"]) for r in mine] == list(range(1, len(passes) + 1))
        for row, p in zip(mine, passes, strict=True):
            assert row["minute"] == str(p.outcome.info.minute)
            assert row["place"] == p.scene.place.value
            assert row["action"] == p.decision.action.value
            assert row["event"] == p.outcome.observation.event.value
            assert row["world"] == ""  # the stubs expect nothing of the world
            for c in Channel:
                assert row[f"error_{c.value}"] == repr(p.comparison.body[c])
                assert row[f"belief_{c.value}"] == repr(p.belief_after.of(c))
                assert row[f"body_{c.value}"] == repr(getattr(p.outcome.info.body, c.value))
            assert row["size"] == repr(p.comparison.size)
            assert row["arousal"] == repr(p.feeling_after.arousal)


def test_stability_csv_has_one_row_per_behavioural_metric(reference: Path) -> None:
    path = reference / "exp" / "stability.csv"
    assert header(path) == ["metric", "icc1", "n_agents", "n_seeds"]
    rows = read_csv(path)
    assert [r["metric"] for r in rows] == list(STABILITY_METRICS)
    assert all((r["n_agents"], r["n_seeds"]) == ("2", "2") for r in rows)
    for r in rows:
        assert r["icc1"] == "nan" or math.isfinite(float(r["icc1"]))


def test_stability_metrics_are_the_behavioural_fields() -> None:
    assert set(STABILITY_METRICS) == set(RunMetrics.model_fields) - {"steps", "ticks", "truncated"}
    assert list(STABILITY_METRICS) == [f for f in RunMetrics.model_fields if f in STABILITY_METRICS]


def test_one_seed_writes_no_stability_csv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path, **{"seeds: [1, 2]": "seeds: [1]", "ticks: null": "ticks: 3"})
    runs_dir = tmp_path / "runs"
    code, err = experiment(config, runs_dir, capsys)
    assert code == 0, err
    assert sorted(p.name for p in (runs_dir / "exp").iterdir()) == ["results.csv", "trace.csv"]


def test_a_rerun_with_one_seed_removes_an_earlier_stability_csv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    two = write_config(tmp_path, **{"ticks: null": "ticks: 3"})
    assert experiment(two, runs_dir, capsys)[0] == 0
    assert (runs_dir / "exp" / "stability.csv").is_file()
    one = write_config(tmp_path, **{"ticks: null": "ticks: 3", "seeds: [1, 2]": "seeds: [1]"})
    code, err = experiment(one, runs_dir, capsys)
    assert code == 0, err
    assert sorted(p.name for p in (runs_dir / "exp").iterdir()) == ["results.csv", "trace.csv"]
    assert len(read_csv(runs_dir / "exp" / "results.csv")) == 2


# --- determinism ----------------------------------------------------------------------------


def test_rerun_gives_byte_identical_csvs_and_does_not_rerun_complete_runs(
    reference: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs_dir = tmp_path / "runs"
    shutil.copytree(reference, runs_dir)
    before = mtimes(runs_dir)
    for name in CSV_FILES:
        (runs_dir / "exp" / name).unlink()

    async def must_not_run(*args: object, **kwargs: object) -> RunResult:
        raise AssertionError("a complete run was run again")

    monkeypatch.setattr(cli, "run", must_not_run)
    code, err = experiment(write_config(tmp_path), runs_dir, capsys)

    assert code == 0, err
    assert csv_bytes(runs_dir) == csv_bytes(reference)
    assert mtimes(runs_dir) == before
    assert err.count("complete, not rerun") == 4


def test_jobs_4_equals_jobs_1(
    reference: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    code, err = experiment(write_config(tmp_path), runs_dir, capsys, "--jobs", "4")
    assert code == 0, err
    assert csv_bytes(runs_dir) == csv_bytes(reference)
    assert dump_bytes(runs_dir) == dump_bytes(reference)


def test_runs_interleave_under_jobs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """With --jobs 4 all four runs are in flight at once; with --jobs 1 only one is."""
    peak = {"now": 0, "max": 0}
    real_run = cli.run

    async def counting(*args: object, **kwargs: object) -> RunResult:
        peak["now"] += 1
        peak["max"] = max(peak["max"], peak["now"])
        try:
            return await real_run(*args, **kwargs)  # type: ignore[arg-type]
        finally:
            peak["now"] -= 1

    config = write_config(tmp_path, **{"ticks: null": "ticks: 5"})
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cli, "run", counting)
        assert experiment(config, tmp_path / "a", capsys, "--jobs", "4")[0] == 0
        assert peak["max"] == 4
        peak["max"] = 0
        assert experiment(config, tmp_path / "b", capsys, "--jobs", "1")[0] == 0
        assert peak["max"] == 1


# --- resume ---------------------------------------------------------------------------------


def test_partial_run_is_resumed_to_the_uninterrupted_result(
    reference: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path)
    runs_dir = tmp_path / "runs"
    partial = next(r for r in load_config(config).runs() if r.run_id == "exp-tavi-s2")
    asyncio.run(run(partial, runs_dir / partial.run_id, stop_after_tick=11))
    assert len(list((runs_dir / partial.run_id).glob("tick_*.json"))) == 11

    code, err = experiment(config, runs_dir, capsys, "--jobs", "2")

    assert code == 0, err
    assert "exp-tavi-s2: resumed at tick 11" in err
    assert csv_bytes(runs_dir) == csv_bytes(reference)
    assert dump_bytes(runs_dir) == dump_bytes(reference)


def test_run_dir_with_a_different_config_fails_before_any_run(
    reference: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    shutil.copytree(reference / "exp-tavi-s1", runs_dir / "exp-tavi-s1")
    changed = write_config(tmp_path, **{"baseline: 0.1": "baseline: 0.2"})

    code, err = experiment(changed, runs_dir, capsys)

    assert code == 2
    assert error_lines(err) == err.splitlines()
    assert len(err.splitlines()) == 1
    assert "exp-tavi-s1" in err and "differs" in err
    assert sorted(p.name for p in runs_dir.iterdir()) == ["exp-tavi-s1"]


def test_run_capped_by_ticks_is_complete_and_not_rerun(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path, **{"ticks: null": "ticks: 5"})
    runs_dir = tmp_path / "runs"
    code, err = experiment(config, runs_dir, capsys)
    assert code == 0, err
    for run_id in RUN_IDS:
        result = load_result(runs_dir / run_id)
        assert len(result.records) == 5 and is_complete(result)
    first, before = csv_bytes(runs_dir), mtimes(runs_dir)

    async def must_not_run(*args: object, **kwargs: object) -> RunResult:
        raise AssertionError("a run complete by its tick cap was run again")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cli, "run", must_not_run)
        code, err = experiment(config, runs_dir, capsys, "--jobs", "4")

    assert code == 0, err
    assert err.count("complete, not rerun") == 4
    assert csv_bytes(runs_dir) == first
    assert mtimes(runs_dir) == before
    assert all(r["truncated"] == "True" for r in read_csv(runs_dir / "exp" / "results.csv"))


# --- failures -------------------------------------------------------------------------------


def test_a_failing_run_lets_the_others_finish_exits_1_and_writes_no_csvs(
    reference: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path)
    runs_dir = tmp_path / "runs"
    real_run = cli.run

    async def failing(config, run_dir, **kwargs):  # type: ignore[no-untyped-def]
        if config.run_id == "exp-ilo-s2":
            await real_run(config, run_dir, stop_after_tick=7, **kwargs)
            raise RuntimeError("boom\nsecond line")
        return await real_run(config, run_dir, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cli, "run", failing)
        code, err = experiment(config, runs_dir, capsys, "--jobs", "2")

    assert code == 1
    assert "Traceback" not in err
    assert error_lines(err) == ["error: exp-ilo-s2 failed: RuntimeError: boom; second line"]
    assert not (runs_dir / "exp").exists()
    for run_id in set(RUN_IDS) - {"exp-ilo-s2"}:
        assert len(load_result(runs_dir / run_id).records) == TICKS
    assert len(load_result(runs_dir / "exp-ilo-s2").records) == 7

    # Rerunning resumes the failed run and gives the same files as a run that never failed.
    code, err = experiment(config, runs_dir, capsys)
    assert code == 0, err
    assert "exp-ilo-s2: resumed at tick 7" in err
    assert csv_bytes(runs_dir) == csv_bytes(reference)
    assert dump_bytes(runs_dir) == dump_bytes(reference)


def test_a_failing_run_removes_tables_left_by_an_earlier_invocation(
    reference: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    shutil.copytree(reference, runs_dir)
    shutil.rmtree(runs_dir / "exp-tavi-s1")
    (runs_dir / "exp" / "notes.txt").write_text("not a table")

    async def failing(config, run_dir, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cli, "run", failing)
        code, err = experiment(write_config(tmp_path), runs_dir, capsys)

    assert code == 1
    assert error_lines(err) == ["error: exp-tavi-s1 failed: RuntimeError: boom"]
    assert sorted(p.name for p in (runs_dir / "exp").iterdir()) == ["notes.txt"]


def test_a_model_phase_not_built_yet_fails_each_run_with_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(
        tmp_path, **{"deliberate: stub": "deliberate: llm", "model: null": "model: test"}
    )
    code, err = experiment(config, tmp_path / "runs", capsys, "--jobs", "2")
    assert code == 1
    assert "Traceback" not in err
    errors = error_lines(err)
    assert len(errors) == 4
    for line, run_id in zip(sorted(errors), sorted(RUN_IDS), strict=True):
        assert line.startswith(f"error: {run_id} failed: ConfigError: deliberate")
    assert not (tmp_path / "runs" / "exp").exists()


def test_bad_config_fails_with_one_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, err = experiment(tmp_path / "nope.yaml", tmp_path / "runs", capsys)
    assert code == 2
    assert len(err.splitlines()) == 1 and err.startswith("error: ")
    assert not (tmp_path / "runs").exists()


def test_duplicate_seeds_fail_with_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path, **{"seeds: [1, 2]": "seeds: [1, 1]"})
    code, err = experiment(config, tmp_path / "runs", capsys)
    assert code == 2
    assert len(err.splitlines()) == 1 and err.startswith("error: ")
    assert "seeds" in err and "unique" in err
    assert not (tmp_path / "runs").exists()


def test_llm_backend_without_a_model_fails_before_any_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SURPRISE_LOOP_MODEL", raising=False)
    config = write_config(tmp_path, **{"deliberate: stub": "deliberate: llm"})
    code, err = experiment(config, tmp_path / "runs", capsys)
    assert code == 2
    assert len(err.splitlines()) == 1 and "SURPRISE_LOOP_MODEL" in err
    assert not (tmp_path / "runs").exists()


def test_a_model_that_cannot_load_fails_before_any_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(
        tmp_path, **{"deliberate: stub": "deliberate: llm", "model: null": "model: nosuch:abc"}
    )
    code, err = experiment(config, tmp_path / "runs", capsys)
    assert code == 2
    assert len(err.splitlines()) == 1 and err.startswith("error: ")
    assert "nosuch:abc" in err
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("jobs", ["0", "-1", "x"])
def test_jobs_must_be_positive(jobs: str, tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as info:
        cli.main(["experiment", str(write_config(tmp_path)), "--jobs", jobs])
    assert info.value.code == 2


def test_tables_are_written_atomically(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    written: list[Path] = []
    real = cli.write_atomic

    def recording(path: Path, text: str) -> None:
        written.append(path)
        real(path, text)

    monkeypatch.setattr(cli, "write_atomic", recording)
    runs_dir = tmp_path / "runs"
    config = write_config(tmp_path, **{"ticks: null": "ticks: 3"})
    assert experiment(config, runs_dir, capsys)[0] == 0
    assert sorted(p.name for p in written) == CSV_FILES
    assert sorted(p.name for p in (runs_dir / "exp").iterdir()) == CSV_FILES  # no .tmp left


# --- the pure CSV writers -------------------------------------------------------------------


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory) -> list[RunResult]:
    path = tmp_path_factory.mktemp("cfg") / "exp.yaml"
    path.write_text(CONFIG)
    return [asyncio.run(run(c)) for c in load_config(path).runs()]


def test_format_cell_is_reproducible() -> None:
    assert format_cell(0.1) == "0.1"
    assert format_cell(1 / 3) == repr(1 / 3)
    assert format_cell(2.0) == "2.0"
    assert format_cell(float("nan")) == "nan"
    assert format_cell(3) == "3"
    assert format_cell(True) == "True"
    assert format_cell("a") == "a"
    assert format_cell(None) == ""
    assert format_cell(Channel.HEAT) == "heat"


def test_writers_are_pure_and_deterministic(results: list[RunResult]) -> None:
    for writer in (results_csv, trace_csv, stability_csv):
        assert writer(results) == writer(list(results))
    assert results_csv(results).endswith("\n")
    assert "\r" not in results_csv(results)


def test_experiment_tables_are_the_three_writers(results: list[RunResult]) -> None:
    assert experiment_tables(results) == {
        "results.csv": results_csv(results),
        "trace.csv": trace_csv(results),
        "stability.csv": stability_csv(results),
    }
    one_seed = [r for r in results if r.config.seed == 1]
    assert sorted(experiment_tables(one_seed)) == ["results.csv", "trace.csv"]


def test_writers_follow_the_given_order(results: list[RunResult]) -> None:
    reordered = results[2:] + results[:2]
    rows = list(csv.DictReader(io.StringIO(results_csv(reordered))))
    assert [(r["agent"], r["seed"]) for r in rows][0] == ("tavi", "1")


def test_stability_is_icc1_over_seeds_within_agent(results: list[RunResult]) -> None:
    rows = {r["metric"]: r for r in csv.DictReader(io.StringIO(stability_csv(results)))}
    metrics = [summarize(r) for r in results]
    for name in STABILITY_METRICS:
        groups = [
            [float(getattr(m, name)) for m in metrics[:2]],
            [float(getattr(m, name)) for m in metrics[2:]],
        ]
        assert rows[name]["icc1"] == format_cell(icc1(groups))


def test_stability_of_a_constant_metric_is_nan(results: list[RunResult]) -> None:
    same = [results[0], results[0], results[0], results[0]]
    relabelled = [
        r.model_copy(
            update={
                "config": r.config.model_copy(
                    update={
                        "agent": r.config.agent.model_copy(update={"name": name}),
                        "seed": seed,
                    }
                )
            }
        )
        for r, (name, seed) in zip(same, [("a", 1), ("a", 2), ("b", 1), ("b", 2)], strict=True)
    ]
    rows = list(csv.DictReader(io.StringIO(stability_csv(relabelled))))
    assert all(r["icc1"] == "nan" for r in rows)
    assert all((r["n_agents"], r["n_seeds"]) == ("2", "2") for r in rows)


# --- write errors ---------------------------------------------------------------------------


def test_tables_that_cannot_be_written_fail_with_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    (runs_dir / "exp").write_text("a file where the tables go")
    config = write_config(tmp_path, **{"ticks: null": "ticks: 3"})
    code, err = experiment(config, runs_dir, capsys)
    assert code == 2
    assert "Traceback" not in err
    assert error_lines(err) == [f"error: {runs_dir / 'exp'}: cannot write: File exists"]
    assert err.splitlines()[-1] == error_lines(err)[0]


def test_a_failed_run_with_a_file_where_the_tables_go_reports_the_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    (runs_dir / "exp").write_text("a file where the tables go")

    async def failing(config, run_dir, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cli, "run", failing)
        code, err = experiment(write_config(tmp_path), runs_dir, capsys)
    assert code == 1
    assert "Traceback" not in err
    assert len(error_lines(err)) == 4
    assert (runs_dir / "exp").read_text() == "a file where the tables go"


def test_a_tick_file_that_cannot_be_written_fails_that_run_with_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_dir = tmp_path / "runs"
    blocked = runs_dir / "exp-tavi-s2" / "tick_000002.json"
    real_replace = os.replace

    def replace(src: str | Path, dst: str | Path) -> None:
        if Path(dst) == blocked:
            raise OSError(errno.ENOSPC, "No space left on device")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace)
    config = write_config(tmp_path, **{"ticks: null": "ticks: 3"})
    code, err = experiment(config, runs_dir, capsys, "--jobs", "2")
    assert code == 1
    assert "Traceback" not in err
    assert error_lines(err) == [
        f"error: exp-tavi-s2 failed: RunError: {blocked}: cannot write: No space left on device"
    ]
    assert not (runs_dir / "exp").exists()
