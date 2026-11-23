"""`surprise-loop run`, `metrics`, `replay` and `gate2`, and the one-line errors of the commands."""

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest
from creatures import TWO_MAPS

from surprise_loop.cli import main
from surprise_loop.loop import load_result
from surprise_loop.metrics import Gate2, RunMetrics, summarize

RUN_ID = "two_maps-plain-s1"


def assert_one_line_error(capsys: pytest.CaptureFixture[str]) -> str:
    out, err = capsys.readouterr()
    assert out == ""
    lines = err.splitlines()
    assert len(lines) == 1, err
    assert lines[0].startswith("error: ")
    return lines[0]


@pytest.fixture(scope="module")
def day(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    """One whole stand-in day of the plain creature: its run directory and `run`'s stdout."""
    runs_dir = tmp_path_factory.mktemp("runs")
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(["run", str(TWO_MAPS), "--runs-dir", str(runs_dir)])
    assert code == 0, err.getvalue()
    return runs_dir / RUN_ID, out.getvalue()


def short_config(tmp_path: Path, ticks: int = 20) -> Path:
    path = tmp_path / "short.yaml"
    path.write_text(TWO_MAPS.read_text() + f"ticks: {ticks}\n")
    return path


# --- run, metrics, replay -----------------------------------------------------------------------


def test_run_writes_the_run_directory_and_prints_the_metrics(day: tuple[Path, str]) -> None:
    run_dir, out = day
    result = load_result(run_dir)
    assert len(result.records) == result.config.world.steps
    assert (run_dir / "config.json").is_file()
    metrics = RunMetrics.model_validate_json(out)
    assert metrics == summarize(result)


def test_metrics_prints_what_run_printed(
    day: tuple[Path, str], capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir, run_out = day
    assert main(["metrics", str(run_dir)]) == 0
    out, err = capsys.readouterr()
    assert out == run_out
    assert err == ""


def test_replay_verifies_the_stored_day(
    day: tuple[Path, str], capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir, _ = day
    assert main(["replay", str(run_dir)]) == 0
    assert "120 ticks verified" in capsys.readouterr().out


def test_run_picks_the_named_agent_and_seed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = short_config(tmp_path, ticks=3)
    code = main(["run", str(config), "--agent", "wary", "--seed", "2", "--runs-dir", str(tmp_path)])
    assert code == 0, capsys.readouterr().err
    assert len(load_result(tmp_path / "two_maps-wary-s2").records) == 3


def test_run_resumes_a_run_with_resume(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = ["run", str(short_config(tmp_path, ticks=3)), "--runs-dir", str(tmp_path / "runs")]
    assert main(args) == 0
    capsys.readouterr()
    assert main([*args, "--resume"]) == 0
    assert RunMetrics.model_validate_json(capsys.readouterr().out).steps == 3


# --- gate2 --------------------------------------------------------------------------------------


def test_gate2_prints_the_marks_of_a_stand_in_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = tmp_path / "runs"
    assert main(["run", str(short_config(tmp_path)), "--runs-dir", str(runs)]) == 0
    capsys.readouterr()

    code = main(["gate2", str(runs / RUN_ID)])

    out, err = capsys.readouterr()
    marks = Gate2.model_validate(json.loads(out))
    assert marks.completed and marks.passed
    assert code == 0, err


def test_gate2_exits_1_on_a_run_that_stopped_short(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = tmp_path / "runs"
    assert main(["run", str(short_config(tmp_path, ticks=3)), "--runs-dir", str(runs)]) == 0
    capsys.readouterr()
    # the stored config now asks for more ticks than the run holds
    config_file = runs / RUN_ID / "config.json"
    stored = json.loads(config_file.read_text())
    config_file.write_text(json.dumps(stored | {"ticks": 4}))

    assert main(["gate2", str(runs / RUN_ID)]) == 1
    assert not Gate2.model_validate_json(capsys.readouterr().out).completed


# --- errors -------------------------------------------------------------------------------------


def test_a_non_empty_run_directory_without_resume_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / RUN_ID).mkdir()
    (tmp_path / RUN_ID / "stray.txt").write_text("x")
    assert main(["run", str(TWO_MAPS), "--runs-dir", str(tmp_path)]) == 2
    assert "--resume" in assert_one_line_error(capsys)


def test_an_unknown_agent_is_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["run", str(TWO_MAPS), "--agent", "nobody", "--runs-dir", str(tmp_path)])
    assert code == 2
    assert "nobody" in assert_one_line_error(capsys)
    assert list(tmp_path.iterdir()) == []


def test_a_config_with_an_unknown_key_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text(TWO_MAPS.read_text() + "colour: blue\n")
    assert main(["run", str(config), "--runs-dir", str(tmp_path)]) == 2
    assert "colour" in assert_one_line_error(capsys)


@pytest.mark.parametrize("command", ["metrics", "replay", "gate2"])
def test_a_missing_run_directory_is_an_error(
    command: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([command, str(tmp_path / "nowhere")]) == 2
    assert "no such run directory" in assert_one_line_error(capsys)


def test_ctrl_c_during_a_run_is_one_line_and_exit_130(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def interrupted(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("surprise_loop.cli.run", interrupted)
    assert main(["run", str(TWO_MAPS), "--runs-dir", str(tmp_path)]) == 130
    out, err = capsys.readouterr()
    assert out == ""
    assert err == "interrupted; rerun with --resume to continue\n"
    assert "Traceback" not in err
