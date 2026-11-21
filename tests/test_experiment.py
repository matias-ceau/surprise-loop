"""`surprise-loop experiment`: the four tables, determinism, skip and resume."""

import asyncio
import io
import shutil
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest
import yaml
from creatures import TWO_MAPS

from surprise_loop import cli
from surprise_loop.config import load_config
from surprise_loop.loop import RunResult, load_result, run

RUN_IDS = ["exp-plain-s1", "exp-plain-s2", "exp-wary-s1", "exp-wary-s2"]
CSV_FILES = ["distances.csv", "results.csv", "stability.csv", "trace.csv"]
TICKS = 20


def write_config(tmp_path: Path) -> Path:
    """Two agents (plain, wary), two seeds, twenty ticks."""
    data = yaml.safe_load(TWO_MAPS.read_text())
    data |= {"name": "exp", "seeds": [1, 2], "ticks": TICKS}
    data["agents"] = [a for a in data["agents"] if a["name"] in ("plain", "wary")]
    path = tmp_path / "exp.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def experiment(
    config: Path, runs_dir: Path, capsys: pytest.CaptureFixture[str], *extra: str
) -> tuple[int, str]:
    """Run the command; return its exit code and stderr (stdout must stay empty)."""
    code = cli.main(["experiment", str(config), "--runs-dir", str(runs_dir), *extra])
    out, err = capsys.readouterr()
    assert out == ""
    return code, err


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


@pytest.fixture(scope="module")
def reference(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One uninterrupted --jobs 1 experiment; tests compare against its files."""
    tmp = tmp_path_factory.mktemp("reference")
    runs_dir = tmp / "runs"
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli.main(["experiment", str(write_config(tmp)), "--runs-dir", str(runs_dir)])
    assert code == 0, err.getvalue()
    assert out.getvalue() == ""
    return runs_dir


def test_writes_one_run_dir_per_agent_and_seed_and_the_four_tables(reference: Path) -> None:
    assert sorted(p.name for p in reference.iterdir()) == sorted([*RUN_IDS, "exp"])
    assert sorted(p.name for p in (reference / "exp").iterdir()) == CSV_FILES
    for run_id in RUN_IDS:
        assert len(load_result(reference / run_id).records) == TICKS


def test_a_rerun_leaves_the_tables_byte_identical_and_reruns_nothing(
    reference: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs_dir = tmp_path / "runs"
    shutil.copytree(reference, runs_dir)
    before = mtimes(runs_dir)

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


def test_a_partial_run_is_resumed_to_the_uninterrupted_result(
    reference: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path)
    runs_dir = tmp_path / "runs"
    partial = next(r for r in load_config(config).runs() if r.run_id == "exp-wary-s2")
    asyncio.run(run(partial, runs_dir / partial.run_id, stop_after_tick=7))

    code, err = experiment(config, runs_dir, capsys, "--jobs", "2")

    assert code == 0, err
    assert "exp-wary-s2: resumed at tick 7" in err
    assert csv_bytes(runs_dir) == csv_bytes(reference)
    assert dump_bytes(runs_dir) == dump_bytes(reference)


def test_a_run_directory_with_another_config_stops_everything(
    reference: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    shutil.copytree(reference / "exp-wary-s1", runs_dir / "exp-wary-s1")
    config = write_config(tmp_path)
    config.write_text(config.read_text().replace(f"ticks: {TICKS}", "ticks: 21"))

    code, err = experiment(config, runs_dir, capsys)

    assert code == 2
    assert len(err.splitlines()) == 1 and err.startswith("error: ")
    assert "exp-wary-s1" in err and "differs" in err
    assert sorted(p.name for p in runs_dir.iterdir()) == ["exp-wary-s1"]


def test_a_failing_run_lets_the_others_finish_and_removes_old_tables(
    reference: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    (runs_dir / "exp").mkdir(parents=True)
    shutil.copy(reference / "exp" / "results.csv", runs_dir / "exp" / "results.csv")
    real_run = cli.run

    async def failing(config, run_dir, **kwargs):  # type: ignore[no-untyped-def]
        if config.run_id == "exp-plain-s2":
            raise RuntimeError("boom\nsecond line")
        return await real_run(config, run_dir, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cli, "run", failing)
        code, err = experiment(write_config(tmp_path), runs_dir, capsys)

    assert code == 1
    errors = [line for line in err.splitlines() if line.startswith("error: ")]
    assert errors == ["error: exp-plain-s2 failed: RuntimeError: boom; second line"]
    assert list((runs_dir / "exp").iterdir()) == []
    for run_id in ("exp-plain-s1", "exp-wary-s1", "exp-wary-s2"):
        assert len(load_result(runs_dir / run_id).records) == TICKS
