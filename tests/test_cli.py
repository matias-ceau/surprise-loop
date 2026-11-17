"""`surprise-loop run`, `metrics` and `replay`, and the one-line errors of the run commands."""

import errno
import io
import json
import os
import shutil
import subprocess
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from surprise_loop.cli import main
from surprise_loop.loop import load_result
from surprise_loop.metrics import summarize
from surprise_loop.state import TickDump

ROOT = Path(__file__).resolve().parent.parent
AFTERNOON = ROOT / "experiments" / "clearing_afternoon.yaml"
RUN_ID = "clearing_afternoon-ilo-s1"
STEPS = 42  # 13:00 to 20:00 in ten-minute world steps; a tick with an override takes two


def assert_one_line_error(capsys: pytest.CaptureFixture[str]) -> str:
    out, err = capsys.readouterr()
    assert out == ""
    lines = err.splitlines()
    assert len(lines) == 1, err
    assert lines[0].startswith("error: ")
    assert "Traceback" not in err
    return lines[0]


def snapshot(directory: Path) -> dict[str, tuple[bytes, int]]:
    return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted(directory.iterdir())}


@pytest.fixture(scope="module")
def afternoon_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    """One whole run of Ilo's afternoon: its run directory and `run`'s stdout."""
    runs_dir = tmp_path_factory.mktemp("runs")
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(["run", str(AFTERNOON), "--runs-dir", str(runs_dir)])
    assert code == 0, err.getvalue()
    return runs_dir / RUN_ID, out.getvalue()


def afternoon_copy(tmp_path: Path, **replacements: str) -> Path:
    text = AFTERNOON.read_text()
    for old, new in replacements.items():
        assert old in text
        text = text.replace(old, new)
    path = tmp_path / "exp.yaml"
    path.write_text(text)
    return path


# --- run -----------------------------------------------------------------------------------


def test_run_writes_a_tick_dump_per_tick_and_the_config(afternoon_run: tuple[Path, str]) -> None:
    run_dir, _ = afternoon_run
    ticks = sorted(p.name for p in run_dir.glob("tick_*.json"))
    result = load_result(run_dir)
    assert sum(len(record.passes) for record in result.records) == STEPS
    assert ticks == [f"tick_{n:06d}.json" for n in range(1, len(result.records) + 1)]
    assert {p.name for p in run_dir.iterdir()} == {*ticks, "config.json"}


def test_run_prints_only_the_metrics_json(afternoon_run: tuple[Path, str]) -> None:
    run_dir, out = afternoon_run
    assert json.loads(out) == json.loads(summarize(load_result(run_dir)).model_dump_json())
    assert json.loads(out)["steps"] == STEPS


def test_run_reports_the_run_directory_on_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = afternoon_copy(tmp_path, **{"ticks: null": "ticks: 3"})
    assert main(["run", str(config), "--runs-dir", str(tmp_path / "runs")]) == 0
    out, err = capsys.readouterr()
    json.loads(out)
    assert str(tmp_path / "runs" / RUN_ID) in err


def test_run_selects_agent_and_seed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = afternoon_copy(tmp_path, **{"ticks: null": "ticks: 3", "seeds: [1]": "seeds: [1, 7]"})
    runs_dir = tmp_path / "runs"

    assert main(["run", str(config), "--runs-dir", str(runs_dir)]) == 0
    assert main(["run", str(config), "--runs-dir", str(runs_dir), "--agent", "tavi"]) == 0
    assert main(["run", str(config), "--runs-dir", str(runs_dir), "--seed", "7"]) == 0
    capsys.readouterr()
    assert sorted(p.name for p in runs_dir.iterdir()) == [
        "clearing_afternoon-ilo-s1",
        "clearing_afternoon-ilo-s7",
        "clearing_afternoon-tavi-s1",
    ]
    stored = load_result(runs_dir / "clearing_afternoon-tavi-s1").config
    assert (stored.agent.name, stored.seed) == ("tavi", 1)
    assert load_result(runs_dir / "clearing_afternoon-ilo-s7").config.seed == 7


def test_runs_dir_defaults_to_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = afternoon_copy(tmp_path, **{"ticks: null": "ticks: 2"})
    monkeypatch.chdir(tmp_path)
    assert main(["run", str(config)]) == 0
    capsys.readouterr()
    assert len(list((tmp_path / "runs" / RUN_ID).glob("tick_*.json"))) == 2


# --- metrics and replay --------------------------------------------------------------------


def test_metrics_prints_the_same_values_as_summarize_and_run(
    afternoon_run: tuple[Path, str], capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir, run_out = afternoon_run
    assert main(["metrics", str(run_dir)]) == 0
    out, err = capsys.readouterr()
    assert err == ""
    printed = json.loads(out)
    assert printed == json.loads(summarize(load_result(run_dir)).model_dump_json())
    assert out == run_out


def test_replay_verifies_every_tick(
    afternoon_run: tuple[Path, str], capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir, _ = afternoon_run
    assert main(["replay", str(run_dir)]) == 0
    out, err = capsys.readouterr()
    ticks = len(load_result(run_dir).records)
    assert out.splitlines() == [f"replay of {run_dir}: {ticks} ticks verified"]
    assert err == ""


def test_replay_mismatch_exits_1(
    afternoon_run: tuple[Path, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "copy"
    shutil.copytree(afternoon_run[0], run_dir)
    path = run_dir / "tick_000020.json"
    dump = TickDump.model_validate_json(path.read_bytes())
    valence = 0.0 if dump.agent.feeling.valence == 0.5 else 0.5  # any valid, different value
    feeling = dump.agent.feeling.model_copy(update={"valence": valence})
    agent = dump.agent.model_copy(update={"feeling": feeling})
    path.write_text(dump.model_copy(update={"agent": agent}).model_dump_json())

    assert main(["replay", str(run_dir)]) == 1
    assert "tick 20" in assert_one_line_error(capsys)


# --- resume --------------------------------------------------------------------------------


def test_resume_of_a_finished_run_succeeds_and_changes_nothing(
    afternoon_run: tuple[Path, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    shutil.copytree(afternoon_run[0], runs_dir / RUN_ID)
    before = snapshot(runs_dir / RUN_ID)

    assert main(["run", str(AFTERNOON), "--runs-dir", str(runs_dir), "--resume"]) == 0
    out, _ = capsys.readouterr()
    assert out == afternoon_run[1]
    assert snapshot(runs_dir / RUN_ID) == before


def test_resume_reruns_the_missing_ticks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = afternoon_copy(tmp_path, **{"ticks: null": "ticks: 5"})
    runs_dir = tmp_path / "runs"
    assert main(["run", str(config), "--runs-dir", str(runs_dir)]) == 0
    first, _ = capsys.readouterr()
    run_dir = runs_dir / RUN_ID
    (run_dir / "tick_000004.json").unlink()
    (run_dir / "tick_000005.json").unlink()

    assert main(["run", str(config), "--runs-dir", str(runs_dir), "--resume"]) == 0
    out, _ = capsys.readouterr()
    assert out == first
    assert len(list(run_dir.glob("tick_*.json"))) == 5


# --- errors --------------------------------------------------------------------------------


def test_non_empty_run_dir_without_resume_fails(
    afternoon_run: tuple[Path, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    shutil.copytree(afternoon_run[0], runs_dir / RUN_ID)
    before = snapshot(runs_dir / RUN_ID)
    assert main(["run", str(AFTERNOON), "--runs-dir", str(runs_dir)]) == 2
    line = assert_one_line_error(capsys)
    assert "not empty" in line and "--resume" in line
    assert snapshot(runs_dir / RUN_ID) == before


def test_resume_with_a_changed_config_fails(
    afternoon_run: tuple[Path, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    shutil.copytree(afternoon_run[0], runs_dir / RUN_ID)
    before = snapshot(runs_dir / RUN_ID)
    changed = afternoon_copy(tmp_path, **{"baseline: -0.1": "baseline: -0.2"})
    assert main(["run", str(changed), "--runs-dir", str(runs_dir), "--resume"]) == 2
    assert "differs" in assert_one_line_error(capsys)
    assert snapshot(runs_dir / RUN_ID) == before


@pytest.mark.parametrize("command", ["run", "experiment"])
def test_runs_dir_that_is_a_file_fails(
    command: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.write_text("not a directory")
    assert main([command, str(AFTERNOON), "--runs-dir", str(runs_dir)]) == 2
    assert "not a directory" in assert_one_line_error(capsys)


def test_run_dir_holding_only_a_config_temp_file_starts_afresh(
    afternoon_run: tuple[Path, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run killed before config.json was renamed into place leaves only config.json.tmp."""
    run_dir = tmp_path / "runs" / RUN_ID
    run_dir.mkdir(parents=True)
    (run_dir / "config.json.tmp").write_text("{")
    assert main(["run", str(AFTERNOON), "--runs-dir", str(tmp_path / "runs")]) == 0
    out, _ = capsys.readouterr()
    assert out == afternoon_run[1]
    assert not (run_dir / "config.json.tmp").exists()


def test_unknown_agent_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", str(AFTERNOON), "--runs-dir", str(tmp_path), "--agent", "nobody"]) == 2
    line = assert_one_line_error(capsys)
    assert "nobody" in line and "ilo, tavi" in line
    assert list(tmp_path.iterdir()) == []


def test_seed_not_in_config_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", str(AFTERNOON), "--runs-dir", str(tmp_path), "--seed", "99"]) == 2
    assert "99" in assert_one_line_error(capsys)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("command", ["run", "experiment"])
def test_config_with_unknown_key_fails(
    command: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = afternoon_copy(tmp_path, **{"    arousal: 0.20\n": "    arousal: 0.20\n    typo: 1\n"})
    assert main([command, str(config), "--runs-dir", str(tmp_path / "runs")]) == 2
    assert "agents.1.typo" in assert_one_line_error(capsys)
    assert not (tmp_path / "runs").exists()


def test_missing_config_file_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", str(tmp_path / "nope.yaml"), "--runs-dir", str(tmp_path)]) == 2
    assert_one_line_error(capsys)


@pytest.mark.parametrize("command", ["metrics", "replay"])
def test_missing_run_dir_fails(
    command: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([command, str(tmp_path / "nope")]) == 2
    assert "no such run directory" in assert_one_line_error(capsys)


@pytest.mark.parametrize("command", ["metrics", "replay"])
def test_empty_run_dir_fails(
    command: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([command, str(tmp_path)]) == 2
    assert "config.json is missing" in assert_one_line_error(capsys)


def test_a_model_backend_not_built_yet_fails_with_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = afternoon_copy(
        tmp_path, **{"deliberate: stub": "deliberate: llm", "model: null": "model: test"}
    )
    assert main(["run", str(config), "--runs-dir", str(tmp_path / "runs")]) == 2
    assert "deliberate" in assert_one_line_error(capsys)


# --- entry point ---------------------------------------------------------------------------


def test_console_script_is_wired() -> None:
    result = subprocess.run(
        ["uv", "run", "surprise-loop", "--help"], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    for command in ("run", "experiment", "metrics", "replay", "lens-page", "ladder-check"):
        assert command in result.stdout


def test_a_tick_file_that_cannot_be_written_fails_with_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    real_replace = os.replace

    def replace(src: str | Path, dst: str | Path) -> None:
        if Path(dst).name == "tick_000002.json":
            raise OSError(errno.ENOSPC, "No space left on device")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace)
    assert main(["run", str(AFTERNOON), "--runs-dir", str(tmp_path)]) == 2
    line = assert_one_line_error(capsys)
    assert line == (
        f"error: {tmp_path / RUN_ID / 'tick_000002.json'}: cannot write: No space left on device"
    )
