"""`surprise-loop experiment`: concurrent runs, skip and resume, failures, and the four CSVs."""

import asyncio
import csv
import io
import math
import shutil
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest
from pydantic_ai import ModelResponse, ToolCallPart
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel

from surprise_loop import cli
from surprise_loop.agents import deliberate_agent
from surprise_loop.config import load_config
from surprise_loop.loop import RunResult, is_complete, load_result, run
from surprise_loop.metrics import (
    STABILITY_METRICS,
    PEPoint,
    PhaseMetrics,
    RunMetrics,
    format_cell,
    icc1,
    phases_csv,
    results_csv,
    stability_csv,
    summarize,
)
from surprise_loop.state import Traits

CONFIG = """\
name: exp
seeds: [1, 2]
ticks: null
world:
  active_lever: random
  reward_magnitude: 1.0
  schedule:
    - {kind: fixed_ratio, steps: 8, ratio: 1}
    - {kind: progressive_ratio, steps: 14, pr_start: 1, pr_increment: 2}
    - {kind: fixed_ratio, steps: 4, ratio: 1}
    - {kind: extinction, steps: 16}
agents:
  - name: slow
    traits: {reward_sensitivity: 1.0, surprise_learning_rate: 0.2, mood_inertia: 0.5}
  - name: fast
    traits: {reward_sensitivity: 2.0, surprise_learning_rate: 0.8, mood_inertia: 0.1}
backends:
  appraise: stub
  deliberate: stub
  evaluate: stub
model: null
"""
RUN_IDS = ["exp-slow-s1", "exp-slow-s2", "exp-fast-s1", "exp-fast-s2"]
CSV_FILES = ["pe_trace.csv", "phases.csv", "results.csv", "stability.csv"]
TRAIT_COLUMNS = list(Traits.model_fields)


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


def test_writes_one_run_dir_per_agent_and_seed_and_the_four_csvs(reference: Path) -> None:
    assert sorted(p.name for p in reference.iterdir()) == sorted([*RUN_IDS, "exp"])
    assert sorted(p.name for p in (reference / "exp").iterdir()) == CSV_FILES
    for run_id in RUN_IDS:
        assert (reference / run_id / "config.json").is_file()
        assert load_result(reference / run_id).world.phase_index == 4  # schedule ran to its end


def test_results_csv_has_one_row_per_run_in_config_order(reference: Path) -> None:
    path = reference / "exp" / "results.csv"
    assert header(path) == ["agent", *TRAIT_COLUMNS, "seed", *RunMetrics.model_fields]
    rows = read_csv(path)
    assert [(r["agent"], r["seed"]) for r in rows] == [
        ("slow", "1"),
        ("slow", "2"),
        ("fast", "1"),
        ("fast", "2"),
    ]
    assert rows[2]["reward_sensitivity"] == "2.0"
    assert rows[2]["surprise_learning_rate"] == "0.8"
    for row, run_id in zip(rows, RUN_IDS, strict=True):
        metrics = summarize(load_result(reference / run_id))
        for field, value in metrics.model_dump().items():
            assert row[field] == format_cell(value), (run_id, field)
        assert row["truncated"] == "False"
        assert row["steps"] == "42"


def test_phases_csv_has_one_row_per_phase_per_run(reference: Path) -> None:
    path = reference / "exp" / "phases.csv"
    assert header(path) == ["agent", "seed", *PhaseMetrics.model_fields]
    rows = read_csv(path)
    assert len(rows) == 4 * 4
    assert [r["phase_kind"] for r in rows[:4]] == [
        "fixed_ratio",
        "progressive_ratio",
        "fixed_ratio",
        "extinction",
    ]
    assert [(r["agent"], r["seed"], r["phase_index"]) for r in rows[4:6]] == [
        ("slow", "2", "0"),
        ("slow", "2", "1"),
    ]


def test_pe_trace_csv_has_one_row_per_world_step(reference: Path) -> None:
    path = reference / "exp" / "pe_trace.csv"
    assert header(path) == ["agent", "seed", *PEPoint.model_fields]
    rows = read_csv(path)
    assert len(rows) == 4 * 42
    first = [r for r in rows if (r["agent"], r["seed"]) == ("slow", "1")]
    assert [int(r["step"]) for r in first] == list(range(1, 43))


def test_stability_csv_has_one_row_per_behavioural_metric(reference: Path) -> None:
    path = reference / "exp" / "stability.csv"
    assert header(path) == ["metric", "icc1", "n_agents", "n_seeds"]
    rows = read_csv(path)
    assert [r["metric"] for r in rows] == list(STABILITY_METRICS)
    assert all((r["n_agents"], r["n_seeds"]) == ("2", "2") for r in rows)
    for r in rows:
        assert r["icc1"] == "nan" or math.isfinite(float(r["icc1"]))


def test_stability_metrics_are_the_numeric_behavioural_fields() -> None:
    bookkeeping = {"steps", "ticks", "truncated", "breakpoint_censored", "extinction_censored"}
    assert set(STABILITY_METRICS) == set(RunMetrics.model_fields) - bookkeeping
    assert list(STABILITY_METRICS) == [f for f in RunMetrics.model_fields if f in STABILITY_METRICS]


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

    config = write_config(tmp_path)
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
    partial = next(r for r in load_config(config).runs() if r.run_id == "exp-fast-s2")
    asyncio.run(run(partial, runs_dir / partial.run_id, stop_after_tick=11))
    assert len(list((runs_dir / partial.run_id).glob("tick_*.json"))) == 11

    code, err = experiment(config, runs_dir, capsys, "--jobs", "2")

    assert code == 0, err
    assert "exp-fast-s2: resumed at tick 11" in err
    assert csv_bytes(runs_dir) == csv_bytes(reference)
    assert dump_bytes(runs_dir) == dump_bytes(reference)


def test_run_dir_with_a_different_config_fails_before_any_run(
    reference: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    shutil.copytree(reference / "exp-fast-s1", runs_dir / "exp-fast-s1")
    changed = write_config(tmp_path, **{"mood_inertia: 0.1": "mood_inertia: 0.2"})

    code, err = experiment(changed, runs_dir, capsys)

    assert code == 2
    assert error_lines(err) == err.splitlines()
    assert len(err.splitlines()) == 1
    assert "exp-fast-s1" in err and "differs" in err
    assert sorted(p.name for p in runs_dir.iterdir()) == ["exp-fast-s1"]


# --- failures -------------------------------------------------------------------------------


def test_a_failing_run_lets_the_others_finish_exits_1_and_writes_no_csvs(
    reference: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path)
    runs_dir = tmp_path / "runs"
    real_run = cli.run

    async def failing(config, run_dir, **kwargs):  # type: ignore[no-untyped-def]
        if config.run_id == "exp-slow-s2":
            await real_run(config, run_dir, stop_after_tick=7, **kwargs)
            raise RuntimeError("boom\nsecond line")
        return await real_run(config, run_dir, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cli, "run", failing)
        code, err = experiment(config, runs_dir, capsys, "--jobs", "2")

    assert code == 1
    assert "Traceback" not in err
    errors = error_lines(err)
    assert errors == ["error: exp-slow-s2 failed: RuntimeError: boom; second line"]
    assert not (runs_dir / "exp").exists()
    for run_id in set(RUN_IDS) - {"exp-slow-s2"}:
        assert load_result(runs_dir / run_id).world.phase_index == 4
    assert len(load_result(runs_dir / "exp-slow-s2").records) == 7

    # Rerunning resumes the failed run and gives the same files as a run that never failed.
    code, err = experiment(config, runs_dir, capsys)
    assert code == 0, err
    assert "exp-slow-s2: resumed at tick 7" in err
    assert csv_bytes(runs_dir) == csv_bytes(reference)
    assert dump_bytes(runs_dir) == dump_bytes(reference)


def test_llm_failures_are_one_line_each_with_tick_and_pass(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    llm = CONFIG.replace("deliberate: stub", "deliberate: llm").replace(
        "model: null", "model: test"
    )
    config = write_config(tmp_path, llm + "params: {llm: {output_retries: 0}}\n")

    def invalid(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"action": "fly"})])

    with deliberate_agent.override(model=FunctionModel(invalid)):
        code, err = experiment(config, tmp_path / "runs", capsys, "--jobs", "2")

    assert code == 1
    assert "Traceback" not in err
    errors = error_lines(err)
    assert len(errors) == 4
    for line, run_id in zip(sorted(errors), sorted(RUN_IDS), strict=True):
        assert line.startswith(f"error: {run_id} failed: UnexpectedModelBehavior: ")
        assert "tick 1, pass 0" in line
    assert not (tmp_path / "runs" / "exp").exists()


def test_bad_config_fails_with_one_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, err = experiment(tmp_path / "nope.yaml", tmp_path / "runs", capsys)
    assert code == 2
    assert len(err.splitlines()) == 1 and err.startswith("error: ")
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


@pytest.mark.parametrize("jobs", ["0", "-1"])
def test_jobs_must_be_positive(jobs: str, tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as info:
        cli.main(["experiment", str(write_config(tmp_path)), "--jobs", jobs])
    assert info.value.code == 2


# --- the LLM path ---------------------------------------------------------------------------


def test_llm_experiment_on_the_test_model_with_jobs_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    text = CONFIG.replace("model: null", "model: test")
    for phase in ("appraise", "deliberate", "evaluate"):
        text = text.replace(f"{phase}: stub", f"{phase}: llm")
    config = write_config(tmp_path, text)
    runs_dir = tmp_path / "runs"

    code, err = experiment(config, runs_dir, capsys, "--jobs", "2")

    assert code == 0, err
    rows = read_csv(runs_dir / "exp" / "results.csv")
    assert len(rows) == 4
    assert all(r["steps"] == "42" for r in rows)
    for run_id in RUN_IDS:
        result = load_result(runs_dir / run_id)
        assert result.config.model == "test"
        assert all(p.decision.rationale for r in result.records for p in r.passes)


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


def test_writers_are_pure_and_deterministic(results: list[RunResult]) -> None:
    for writer in (results_csv, phases_csv, stability_csv):
        assert writer(results) == writer(list(results))
    assert results_csv(results).endswith("\n")
    assert "\r" not in results_csv(results)


def test_writers_follow_the_given_order(results: list[RunResult]) -> None:
    reordered = results[2:] + results[:2]
    rows = list(csv.DictReader(io.StringIO(results_csv(reordered))))
    assert [(r["agent"], r["seed"]) for r in rows][0] == ("fast", "1")


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
    # every run reaches the schedule's end, so no metric is constant by accident here; force one
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


def test_llm_traits_config_is_the_real_model_experiment() -> None:
    experiment = load_config(Path(__file__).resolve().parent.parent / "experiments/llm_traits.yaml")
    assert len(experiment.agents) == 4 and experiment.seeds == [1, 2]
    assert experiment.model is None and experiment.ticks is None
    assert set(experiment.backends.model_dump().values()) == {"llm"}
    assert sum(phase.steps for phase in experiment.world.schedule) == 150


# --- fix round 1 ----------------------------------------------------------------------------


def test_duplicate_seeds_fail_with_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path, **{"seeds: [1, 2]": "seeds: [1, 1]"})
    code, err = experiment(config, tmp_path / "runs", capsys)
    assert code == 2
    assert len(err.splitlines()) == 1 and err.startswith("error: ")
    assert "seeds" in err and "unique" in err
    assert not (tmp_path / "runs").exists()


def test_run_capped_by_ticks_is_complete_and_not_rerun(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path, **{"ticks: null": "ticks: 5"})
    runs_dir = tmp_path / "runs"
    code, err = experiment(config, runs_dir, capsys)
    assert code == 0, err
    for run_id in RUN_IDS:
        result = load_result(runs_dir / run_id)
        assert len(result.records) == 5 and result.world.phase_index == 0  # world not done
        assert is_complete(result)
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


def test_a_failing_run_removes_tables_left_by_an_earlier_invocation(
    reference: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    shutil.copytree(reference, runs_dir)
    shutil.rmtree(runs_dir / "exp-fast-s1")
    (runs_dir / "exp" / "notes.txt").write_text("not a table")

    async def failing(config, run_dir, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cli, "run", failing)
        code, err = experiment(write_config(tmp_path), runs_dir, capsys)

    assert code == 1
    assert error_lines(err) == ["error: exp-fast-s1 failed: RuntimeError: boom"]
    assert sorted(p.name for p in (runs_dir / "exp").iterdir()) == ["notes.txt"]


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
    assert experiment(write_config(tmp_path), runs_dir, capsys)[0] == 0
    assert sorted(p.name for p in written) == CSV_FILES
    assert sorted(p.name for p in (runs_dir / "exp").iterdir()) == CSV_FILES  # no .tmp left
