"""The film: two stored runs side by side, paired by world step, as one self-contained page."""

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import pytest

from surprise_loop.cli import main
from surprise_loop.config import load_config
from surprise_loop.feeling import felt
from surprise_loop.loop import RunError, RunResult, run
from surprise_loop.metrics import flatten
from surprise_loop.phrases import LADDER_WORDS
from surprise_loop.state import Band, ChangeWord, Temperament
from surprise_loop.viewer.lens_page import lens_page_html
from surprise_loop.viewer.scene import expected_felt, scene_html
from surprise_loop.world import Channel, FactKind, Levels

ROOT = Path(__file__).resolve().parent.parent
AFTERNOON = ROOT / "experiments" / "clearing_afternoon.yaml"
DAY = ROOT / "experiments" / "clearing_day.yaml"
FRAGMENT = ROOT / "src" / "surprise_loop" / "viewer" / "clearing.html"


def stub_runs(path: Path, runs_dir: Path | None = None) -> tuple[RunResult, RunResult]:
    configs = load_config(path).runs()
    assert len(configs) == 2
    a, b = (asyncio.run(run(c, None if runs_dir is None else runs_dir / c.run_id)) for c in configs)
    return a, b


def embedded(page: str) -> dict[str, Any]:
    match = re.search(
        r'<script type="application/json" id="scene-data">(.*?)</script>', page, flags=re.S
    )
    assert match is not None, "no embedded scene data"
    return json.loads(match.group(1))


@pytest.fixture(scope="module")
def afternoon() -> tuple[RunResult, RunResult]:
    return stub_runs(AFTERNOON)


@pytest.fixture(scope="module")
def page(afternoon: tuple[RunResult, RunResult]) -> str:
    return scene_html(*afternoon)


@pytest.fixture(scope="module")
def data(page: str) -> dict[str, Any]:
    return embedded(page)


# --- the data -----------------------------------------------------------------------------------


def test_the_data_holds_every_world_step_of_both_runs(
    afternoon: tuple[RunResult, RunResult], data: dict[str, Any]
) -> None:
    a, b = afternoon
    # Ilo's afternoon has ticks of two passes; each pass is a world step of its own
    assert any(len(r.passes) == 2 for r in a.records)
    steps_a, steps_b = flatten(a.records), flatten(b.records)
    assert len(steps_a) == len(steps_b) > len(a.records)
    assert len(data["steps"]) == len(steps_a)
    assert [s["bodies"][0]["action"] for s in data["steps"]] == [
        p.decision.action.value for p in steps_a
    ]
    assert [s["bodies"][1]["action"] for s in data["steps"]] == [
        p.decision.action.value for p in steps_b
    ]
    assert [s["minute_after"] for s in data["steps"]] == [p.outcome.info.minute for p in steps_a]
    assert [s["bodies"][0]["second_pass"] for s in data["steps"]] == [
        p.pass_index == 1 for p in steps_a
    ]
    overridden = [i for i, p in enumerate(steps_a) if p.override is not None]
    assert overridden
    for i in overridden:
        assert data["steps"][i]["bodies"][0]["override"]["channel"] == "heat"
        assert data["steps"][i + 1]["bodies"][0]["second_pass"]


def test_the_minute_and_place_before_a_step_are_where_the_last_one_ended(
    afternoon: tuple[RunResult, RunResult], data: dict[str, Any]
) -> None:
    a, _ = afternoon
    steps = data["steps"]
    assert steps[0]["minute"] == a.records[0].observation.minute
    assert steps[0]["bodies"][0]["place"] == a.records[0].observation.place.value
    for before, after in zip(steps, steps[1:], strict=False):
        assert after["minute"] == before["minute_after"]
        for k in (0, 1):
            assert after["bodies"][k]["place"] == before["bodies"][k]["place_after"]


def test_the_text_under_a_panel_is_the_recorded_text(
    afternoon: tuple[RunResult, RunResult], data: dict[str, Any]
) -> None:
    for k, result in enumerate(afternoon):
        records = flatten(result.records)
        readings = [r.appraisal.reading for r in result.records for _ in r.passes]
        for step, p, reading in zip(data["steps"], records, readings, strict=True):
            body = step["bodies"][k]
            assert body["scene"]["text"] == p.scene.text()
            assert body["scene"]["opening"] == p.scene.opening
            assert body["reading"] == reading
            assert [f["name"] for f in body["scene"]["facts"]] == [f.name for f in p.scene.facts]


def test_noticed_facts_are_exactly_those_with_a_phrase(data: dict[str, Any]) -> None:
    for step in data["steps"]:
        for body in step["bodies"]:
            for fact in body["scene"]["facts"]:
                assert fact["noticed"] == (fact["phrase"] is not None)


def test_the_measured_strip_has_the_worlds_truth_and_the_distances(
    afternoon: tuple[RunResult, RunResult], data: dict[str, Any]
) -> None:
    a, _ = afternoon
    assert data["world"]["paces"] == a.config.world.paces
    for step, p in zip(data["steps"], flatten(a.records), strict=True):
        truth = step["bodies"][0]["truth"]
        info = p.outcome.info
        assert truth["air_c"] == round(info.air_c, 4)
        assert truth["log_shaded"] == info.log_shaded
        assert truth["stream_full"] == info.stream_full
        assert truth["body"]["water"] == round(info.body.water, 4)
    assert [x["name"] for x in data["agents"]] == ["ilo", "tavi"]
    assert data["agents"][0]["setpoint"] == a.config.agent.temperament.setpoint.model_dump()


def test_expected_felt_is_the_felt_strength_at_the_middle_of_each_band() -> None:
    t = Temperament(
        setpoint=Levels(heat=0.0, water=0.8, energy=0.7),
        tightness={Channel.HEAT: 2.0, Channel.WATER: 4.0, Channel.ENERGY: 4.0},
        trust={c: 0.5 for c in Channel},
        habit={k: 0.1 for k in FactKind},
        baseline=0.0,
    )
    bands = {
        Channel.HEAT: Band(lo=0.2, hi=0.6),  # middle 0.4
        Channel.WATER: Band(lo=None, hi=0.5),  # open below: its finite edge
        Channel.ENERGY: Band(lo=0.9, hi=None),  # open above: its finite edge
    }
    got = expected_felt(bands, t)
    want = felt(Levels(heat=0.4, water=0.5, energy=0.9), t)
    assert got == pytest.approx(want)


def test_a_models_words_are_shown_through_the_ladder(
    afternoon: tuple[RunResult, RunResult],
) -> None:
    a, b = afternoon
    words = {Channel.HEAT: ChangeWord.MUCH_LESS, Channel.WATER: ChangeWord.SAME}
    first = a.records[0]
    p = first.passes[0]
    expectation = p.decision.expectation.model_copy(update={"words": words})
    decision = p.decision.model_copy(update={"expectation": expectation})
    passes = [p.model_copy(update={"decision": decision}), *first.passes[1:]]
    records = [first.model_copy(update={"passes": passes}), *a.records[1:]]
    data = embedded(scene_html(a.model_copy(update={"records": records}), b))
    assert data["steps"][0]["bodies"][0]["expected"]["words"] == {
        "heat": LADDER_WORDS[Channel.HEAT][ChangeWord.MUCH_LESS],
        "water": LADDER_WORDS[Channel.WATER][ChangeWord.SAME],
    }


def test_stub_steps_carry_no_words_and_a_number_per_channel(data: dict[str, Any]) -> None:
    for step in data["steps"]:
        for body in step["bodies"]:
            assert body["expected"]["words"] is None
            assert set(body["expected"]["felt"]) == {"heat", "water", "energy"}


# --- refusals -----------------------------------------------------------------------------------


def _with_config(result: RunResult, **update: Any) -> RunResult:
    return result.model_copy(update={"config": result.config.model_copy(update=update)})


def _refusal(a: RunResult, b: RunResult) -> str:
    with pytest.raises(RunError) as caught:
        scene_html(a, b)
    message = str(caught.value)
    assert "\n" not in message
    return message


def test_runs_with_different_seeds_are_refused(afternoon: tuple[RunResult, RunResult]) -> None:
    a, b = afternoon
    assert "seed" in _refusal(a, _with_config(b, seed=b.config.seed + 1))


def test_runs_with_different_worlds_are_refused(afternoon: tuple[RunResult, RunResult]) -> None:
    a, b = afternoon
    world = b.config.world.model_copy(update={"trickle_factor": 0.5})
    assert "world" in _refusal(a, _with_config(b, world=world))


def test_runs_with_different_lengths_are_refused(afternoon: tuple[RunResult, RunResult]) -> None:
    a, b = afternoon
    shorter = b.model_copy(update={"records": b.records[:-1]})
    assert "steps" in _refusal(a, shorter)


# --- the page -----------------------------------------------------------------------------------


def test_page_reaches_for_nothing_outside_itself(page: str) -> None:
    assert "http://" not in page
    assert "https://" not in page
    assert "//" not in page
    assert not re.search(r"<script[^>]*\ssrc", page, flags=re.I)
    assert not re.search(r"<link\b", page, flags=re.I)
    assert "@import" not in page
    assert all(ref.startswith("url(#") for ref in re.findall(r"url\([^)]*", page))


def test_data_is_embedded_once_and_safely(page: str) -> None:
    assert page.count('type="application/json"') == 1
    body = page.split('id="scene-data">', 1)[1].split("</script>", 1)[0]
    assert "</" not in body
    assert "@@" not in page


def test_both_pages_draw_the_clearing_from_the_one_fragment(
    page: str, afternoon: tuple[RunResult, RunResult]
) -> None:
    fragment = FRAGMENT.read_text(encoding="utf-8").strip()
    lens = lens_page_html(load_config(AFTERNOON))
    assert fragment in page
    assert fragment in lens
    assert "function paintScene(" in fragment
    assert "function sceneSvg(" in fragment


def test_a_day_film_embeds_no_episode_list() -> None:
    page = scene_html(*stub_runs(DAY))
    assert "episodes" not in page
    assert "need_before" not in page
    assert len(page.encode("utf-8")) < 3_000_000
    assert len(embedded(page)["steps"]) == (1200 - 360) // 10


# --- the command --------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def run_dirs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    runs_dir = tmp_path_factory.mktemp("runs")
    stub_runs(AFTERNOON, runs_dir)
    return runs_dir / "clearing_afternoon-ilo-s1", runs_dir / "clearing_afternoon-tavi-s1"


def test_cli_writes_the_film(tmp_path: Path, run_dirs: tuple[Path, Path], page: str) -> None:
    out = tmp_path / "film.html"
    assert main(["scene", str(run_dirs[0]), str(run_dirs[1]), "-o", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == page


def test_cli_writes_scene_html_here_by_default(
    tmp_path: Path, run_dirs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["scene", str(run_dirs[0]), str(run_dirs[1])]) == 0
    assert (tmp_path / "scene.html").is_file()


def test_cli_refuses_a_missing_run_in_one_line(
    tmp_path: Path, run_dirs: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "film.html"
    code = main(["scene", str(run_dirs[0]), str(tmp_path / "nope"), "-o", str(out)])
    assert code == 2
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and err.startswith("error: ")
    assert not out.exists()


def test_cli_refuses_runs_that_do_not_belong_together(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    day = load_config(DAY).runs()[0]
    afternoon = load_config(AFTERNOON).runs()[1]
    asyncio.run(run(day, tmp_path / "day"))
    asyncio.run(run(afternoon, tmp_path / "afternoon"))
    out = tmp_path / "film.html"
    code = main(["scene", str(tmp_path / "day"), str(tmp_path / "afternoon"), "-o", str(out)])
    assert code == 2
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and err.startswith("error: ")
    assert not out.exists()
