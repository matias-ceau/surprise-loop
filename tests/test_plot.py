"""scripts/plot_results.py: three PNGs from a small stub experiment, one-line errors otherwise."""

import importlib.util
import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("matplotlib")

from surprise_loop import cli  # noqa: E402

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "plot_results.py"
PNGS = ["cumulative_record.png", "metrics_by_agent.png", "pe_trace.png"]

CONFIG = """\
name: tiny
seeds: [1, 2]
ticks: null
world:
  active_lever: random
  reward_magnitude: 1.0
  schedule:
    - {kind: fixed_ratio, steps: 6, ratio: 1}
    - {kind: progressive_ratio, steps: 10, pr_start: 1, pr_increment: 2}
    - {kind: extinction, steps: 10}
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


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("plot_results", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def plot() -> ModuleType:
    return load_script()


@pytest.fixture(scope="module")
def experiment_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("plot")
    config = tmp / "tiny.yaml"
    config.write_text(CONFIG)
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err:
        code = cli.main(["experiment", str(config), "--runs-dir", str(tmp / "runs")])
    assert code == 0, err.getvalue()
    return tmp / "runs" / "tiny"


def run_main(plot: ModuleType, *argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = plot.main(list(argv))
    return code, out.getvalue(), err.getvalue()


def test_writes_three_non_empty_pngs(plot: ModuleType, experiment_dir: Path) -> None:
    code, out, err = run_main(plot, str(experiment_dir))
    assert code == 0, err
    plots = experiment_dir / "plots"
    assert sorted(p.name for p in plots.iterdir()) == PNGS
    for name in PNGS:
        data = (plots / name).read_bytes()
        assert data.startswith(b"\x89PNG") and len(data) > 10_000, name
    assert sorted(Path(line).name for line in out.splitlines()) == PNGS


def test_unsmoothed_run_also_works(plot: ModuleType, experiment_dir: Path) -> None:
    assert run_main(plot, str(experiment_dir), "--smooth", "1")[0] == 0


def test_missing_directory_is_one_line_error(plot: ModuleType, tmp_path: Path) -> None:
    code, out, err = run_main(plot, str(tmp_path / "nope"))
    assert code == 2 and out == ""
    assert len(err.splitlines()) == 1 and err.startswith("error: ")
    assert not (tmp_path / "nope").exists()


def test_missing_csv_is_one_line_error(plot: ModuleType, tmp_path: Path) -> None:
    (tmp_path / "results.csv").write_text("agent,seed\na,1\n")
    code, out, err = run_main(plot, str(tmp_path))
    assert code == 2 and out == ""
    assert len(err.splitlines()) == 1 and "pe_trace.csv" in err
    assert not (tmp_path / "plots").exists()


def test_smoothing_never_crosses_a_phase_boundary(plot: ModuleType) -> None:
    values = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    phases = [0, 0, 0, 1, 1, 1]
    assert plot.smooth_within_phases(list(range(6)), values, phases, 3) == values
    assert plot.smooth_within_phases([0, 1, 2], [0.0, 3.0, 0.0], [0, 0, 0], 3) == [1.5, 1.0, 1.5]
    assert plot.smooth_within_phases([0, 1], [0.0, 3.0], [0, 0], 1) == [0.0, 3.0]
