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
SMOKE = ROOT / "experiments" / "stub_smoke.yaml"
SMOKE_RUN_ID = "stub_smoke-baseline-s1"


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
def smoke_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    """One 100-tick run of the real smoke config: its run directory and `run`'s stdout."""
    runs_dir = tmp_path_factory.mktemp("runs")
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(["run", str(SMOKE), "--runs-dir", str(runs_dir)])
    assert code == 0, err.getvalue()
    return runs_dir / SMOKE_RUN_ID, out.getvalue()


def smoke_copy(tmp_path: Path, **replacements: str) -> Path:
    text = SMOKE.read_text()
    for old, new in replacements.items():
        assert old in text
        text = text.replace(old, new)
    path = tmp_path / "exp.yaml"
    path.write_text(text)
    return path


# --- run -----------------------------------------------------------------------------------


def test_run_writes_100_tick_dumps_and_config(smoke_run: tuple[Path, str]) -> None:
    run_dir, _ = smoke_run
    ticks = sorted(p.name for p in run_dir.glob("tick_*.json"))
    assert len(ticks) == 100
    assert ticks[0] == "tick_000001.json" and ticks[-1] == "tick_000100.json"
    assert (run_dir / "config.json").is_file()
    assert {p.name for p in run_dir.iterdir()} == {*ticks, "config.json"}


def test_run_prints_only_the_metrics_json(smoke_run: tuple[Path, str]) -> None:
    run_dir, out = smoke_run
    assert json.loads(out) == json.loads(summarize(load_result(run_dir)).model_dump_json())
    assert json.loads(out)["ticks"] == 100


def test_run_reports_the_run_directory_on_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = smoke_copy(tmp_path, **{"ticks: 100": "ticks: 3"})
    assert main(["run", str(config), "--runs-dir", str(tmp_path / "runs")]) == 0
    out, err = capsys.readouterr()
    json.loads(out)
    assert str(tmp_path / "runs" / SMOKE_RUN_ID) in err


def test_run_selects_agent_and_seed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    other_agent = (
        "  - name: other\n"
        "    traits: {reward_sensitivity: 2.0, surprise_learning_rate: 0.3, mood_inertia: 0.2}\n"
    )
    replacements = {"ticks: 100": "ticks: 3", "seeds: [1]": "seeds: [1, 7]"}
    config = smoke_copy(tmp_path, **replacements, **{"backends:": other_agent + "backends:"})
    runs_dir = tmp_path / "runs"

    assert main(["run", str(config), "--runs-dir", str(runs_dir)]) == 0
    assert main(["run", str(config), "--runs-dir", str(runs_dir), "--agent", "other"]) == 0
    assert main(["run", str(config), "--runs-dir", str(runs_dir), "--seed", "7"]) == 0
    capsys.readouterr()
    assert sorted(p.name for p in runs_dir.iterdir()) == [
        "stub_smoke-baseline-s1",
        "stub_smoke-baseline-s7",
        "stub_smoke-other-s1",
    ]
    stored = load_result(runs_dir / "stub_smoke-other-s1").config
    assert (stored.agent.name, stored.seed) == ("other", 1)
    assert load_result(runs_dir / "stub_smoke-baseline-s7").config.seed == 7


def test_runs_dir_defaults_to_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = smoke_copy(tmp_path, **{"ticks: 100": "ticks: 2"})
    monkeypatch.chdir(tmp_path)
    assert main(["run", str(config)]) == 0
    capsys.readouterr()
    assert len(list((tmp_path / "runs" / SMOKE_RUN_ID).glob("tick_*.json"))) == 2


# --- metrics and replay --------------------------------------------------------------------


def test_metrics_prints_the_same_values_as_summarize_and_run(
    smoke_run: tuple[Path, str], capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir, run_out = smoke_run
    assert main(["metrics", str(run_dir)]) == 0
    out, err = capsys.readouterr()
    assert err == ""
    printed = json.loads(out)
    assert printed == json.loads(summarize(load_result(run_dir)).model_dump_json())
    assert printed == json.loads(run_out)


def test_replay_verifies_every_tick(
    smoke_run: tuple[Path, str], capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir, _ = smoke_run
    assert main(["replay", str(run_dir)]) == 0
    out, err = capsys.readouterr()
    assert len(out.splitlines()) == 1
    assert "100 ticks verified" in out
    assert err == ""


def test_replay_mismatch_exits_1(
    smoke_run: tuple[Path, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "copy"
    shutil.copytree(smoke_run[0], run_dir)
    path = run_dir / "tick_000050.json"
    dump = TickDump.model_validate_json(path.read_bytes())
    valence = 0.0 if dump.agent.mood.valence == 0.5 else 0.5  # any valid, different value
    mood = dump.agent.mood.model_copy(update={"valence": valence})
    agent = dump.agent.model_copy(update={"mood": mood})
    path.write_text(dump.model_copy(update={"agent": agent}).model_dump_json())

    assert main(["replay", str(run_dir)]) == 1
    assert "tick 50" in assert_one_line_error(capsys)


# --- resume --------------------------------------------------------------------------------


def test_resume_of_a_finished_run_succeeds_and_changes_nothing(
    smoke_run: tuple[Path, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    shutil.copytree(smoke_run[0], runs_dir / SMOKE_RUN_ID)
    before = snapshot(runs_dir / SMOKE_RUN_ID)

    assert main(["run", str(SMOKE), "--runs-dir", str(runs_dir), "--resume"]) == 0
    out, _ = capsys.readouterr()
    assert json.loads(out) == json.loads(smoke_run[1])
    assert snapshot(runs_dir / SMOKE_RUN_ID) == before


def test_resume_reruns_the_missing_ticks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = smoke_copy(tmp_path, **{"ticks: 100": "ticks: 5"})
    runs_dir = tmp_path / "runs"
    assert main(["run", str(config), "--runs-dir", str(runs_dir)]) == 0
    first, _ = capsys.readouterr()
    run_dir = runs_dir / SMOKE_RUN_ID
    (run_dir / "tick_000004.json").unlink()
    (run_dir / "tick_000005.json").unlink()

    assert main(["run", str(config), "--runs-dir", str(runs_dir), "--resume"]) == 0
    out, _ = capsys.readouterr()
    assert json.loads(out) == json.loads(first)
    assert len(list(run_dir.glob("tick_*.json"))) == 5


# --- errors --------------------------------------------------------------------------------


def test_non_empty_run_dir_without_resume_fails(
    smoke_run: tuple[Path, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    shutil.copytree(smoke_run[0], runs_dir / SMOKE_RUN_ID)
    before = snapshot(runs_dir / SMOKE_RUN_ID)
    assert main(["run", str(SMOKE), "--runs-dir", str(runs_dir)]) == 2
    line = assert_one_line_error(capsys)
    assert "not empty" in line and "--resume" in line
    assert snapshot(runs_dir / SMOKE_RUN_ID) == before


def test_resume_with_a_changed_config_fails(
    smoke_run: tuple[Path, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    shutil.copytree(smoke_run[0], runs_dir / SMOKE_RUN_ID)
    before = snapshot(runs_dir / SMOKE_RUN_ID)
    changed = smoke_copy(tmp_path, **{"mood_inertia: 0.5": "mood_inertia: 0.6"})
    assert main(["run", str(changed), "--runs-dir", str(runs_dir), "--resume"]) == 2
    assert "differs" in assert_one_line_error(capsys)
    assert snapshot(runs_dir / SMOKE_RUN_ID) == before


@pytest.mark.parametrize("command", ["run", "experiment"])
def test_runs_dir_that_is_a_file_fails(
    command: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.write_text("not a directory")
    assert main([command, str(SMOKE), "--runs-dir", str(runs_dir)]) == 2
    assert "not a directory" in assert_one_line_error(capsys)


def test_run_dir_holding_only_a_config_temp_file_starts_afresh(
    smoke_run: tuple[Path, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run killed before config.json was renamed into place leaves only config.json.tmp."""
    run_dir = tmp_path / "runs" / SMOKE_RUN_ID
    run_dir.mkdir(parents=True)
    (run_dir / "config.json.tmp").write_text("{")
    assert main(["run", str(SMOKE), "--runs-dir", str(tmp_path / "runs")]) == 0
    out, _ = capsys.readouterr()
    assert out == smoke_run[1]
    assert not (run_dir / "config.json.tmp").exists()


def test_unknown_agent_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", str(SMOKE), "--runs-dir", str(tmp_path), "--agent", "nobody"]) == 2
    assert "nobody" in assert_one_line_error(capsys)
    assert list(tmp_path.iterdir()) == []


def test_seed_not_in_config_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", str(SMOKE), "--runs-dir", str(tmp_path), "--seed", "99"]) == 2
    assert "99" in assert_one_line_error(capsys)
    assert list(tmp_path.iterdir()) == []


def test_config_with_unknown_key_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = smoke_copy(tmp_path, **{"mood_inertia: 0.5": "mood_inertia: 0.5, typo: 1"})
    assert main(["run", str(config), "--runs-dir", str(tmp_path / "runs")]) == 2
    assert "agents.0.traits.typo" in assert_one_line_error(capsys)
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


# --- entry point ---------------------------------------------------------------------------


def test_console_script_is_wired() -> None:
    result = subprocess.run(
        ["uv", "run", "surprise-loop", "--help"], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "run" in result.stdout and "replay" in result.stdout


def test_run_with_an_llm_backend_prints_no_pydantic_ai_banner(tmp_path: Path) -> None:
    """pydantic-ai shows a first-run banner on stderr to a terminal or a coding agent (it stays
    silent under pytest and CI). The CLI owns its output, so it turns the banner off."""
    config = tmp_path / "llm.yaml"
    text = SMOKE.read_text().replace("ticks: 100", "ticks: 3")
    config.write_text(text.replace("deliberate: stub", "deliberate: llm"))
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"PYTEST_VERSION", "CI", "PYDANTIC_AI_NO_BANNER", "SURPRISE_LOOP_MODEL"}
    }
    env["AI_AGENT"] = "test-harness"  # what makes pydantic-ai show the banner on a pipe
    env["SURPRISE_LOOP_MODEL"] = "test"

    result = subprocess.run(
        ["uv", "run", "surprise-loop", "run", str(config), "--runs-dir", str(tmp_path / "runs")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["steps"] == 3
    for text in (result.stdout, result.stderr):
        assert "PYDANTIC_AI_NO_BANNER" not in text
        assert "observability" not in text.lower()
