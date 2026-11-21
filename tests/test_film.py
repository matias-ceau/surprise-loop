"""The film: two stored runs on the same layout, played side by side as three maps."""

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import pytest
from creatures import LAYOUT, run_config

from surprise_loop.cli import main
from surprise_loop.loop import RunError, load_dumps, run
from surprise_loop.phrases import (
    FEEL_WORDS,
    NEAR_WORDS,
    STIR_WORDS,
    SURPRISE_WORDS,
    TONE_WORDS,
    WENT_BETTER,
    WENT_SAME,
    WENT_WORSE,
)
from surprise_loop.reasoner import reasoner_beliefs
from surprise_loop.viewer.film import film_data, film_html
from surprise_loop.world import Belief, Channel, Kind, key, position_of, truth_at

TICKS = 30
AGENTS = ("wary", "hopeful")


def stored_run(runs_dir: Path, agent: str, **changes: Any) -> Path:
    config = run_config(agent, ticks=TICKS, **changes)
    run_dir = runs_dir / config.run_id
    asyncio.run(run(config, run_dir))
    return run_dir


@pytest.fixture(scope="module")
def runs_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("runs")


@pytest.fixture(scope="module")
def runs(runs_dir: Path) -> tuple[Path, Path]:
    a, b = (stored_run(runs_dir, agent) for agent in AGENTS)
    return a, b


@pytest.fixture(scope="module")
def data(runs: tuple[Path, Path]) -> dict[str, Any]:
    return film_data(*runs)


@pytest.fixture(scope="module")
def page(runs: tuple[Path, Path]) -> str:
    return film_html(*runs)


def embedded(page: str) -> Any:
    match = re.search(r'<script type="application/json" id="film-data">(.*?)</script>', page, re.S)
    assert match is not None, "no embedded film data"
    return json.loads(match.group(1))


def frame_keys(data: dict[str, Any]) -> list[str]:
    west, south, east, north = data["world"]["frame"]
    return [key((x, y)) for y in range(north, south - 1, -1) for x in range(west, east + 1)]


# --- the data -----------------------------------------------------------------------------------


def test_two_runs_of_every_tick_over_the_frame_of_the_world(data: dict[str, Any]) -> None:
    assert [r["name"] for r in data["runs"]] == [f"two_maps-{a}-s1" for a in AGENTS]
    assert [r["agent"] for r in data["runs"]] == list(AGENTS)
    assert [len(r["ticks"]) for r in data["runs"]] == [TICKS, TICKS]
    assert [[t["tick"] for t in r["ticks"]] for r in data["runs"]] == [
        list(range(1, TICKS + 1))
    ] * 2

    width, height = len(LAYOUT[0]), len(LAYOUT)
    tiles = data["world"]["tiles"]
    assert sorted(tiles) == sorted(frame_keys(data))
    assert len(tiles) == (width + 2) * (height + 2)
    west, south, east, north = data["world"]["frame"]
    ring = [
        k for k in tiles if position_of(k)[0] in (west, east) or position_of(k)[1] in (south, north)
    ]
    assert len(ring) == 2 * (width + 2) + 2 * height
    for k in ring:
        assert tiles[k]["kind"] == Kind.OBSTACLE.value
        assert tiles[k]["beliefs"] == {b.value: float(b is Belief.BLOCKS) for b in Belief}


def test_the_world_is_the_truth_of_the_layout(data: dict[str, Any]) -> None:
    config = run_config("wary").world
    for k, tile in data["world"]["tiles"].items():
        truth = truth_at(config, position_of(k))
        assert tile["beliefs"] == {b.value: truth[b] for b in Belief}
    assert data["world"]["tiles"]["0,0"]["kind"] == Kind.GROUND.value
    # south-west of the start, row 5 holds toxic food
    assert data["world"]["tiles"]["-1,-1"]["kind"] == Kind.TOXIC.value


def replayed(ticks: list[dict[str, Any]], part: str) -> dict[str, dict[str, float]]:
    tiles: dict[str, dict[str, float]] = {}
    for t in ticks:
        for change in t[part]:
            tiles[key(tuple(change["at"]))] = change["beliefs"]
    return tiles


def test_replaying_the_changes_gives_the_final_maps(
    runs: tuple[Path, Path], data: dict[str, Any]
) -> None:
    for run_dir, shown in zip(runs, data["runs"], strict=True):
        final = load_dumps(run_dir)[-1].agent
        own, reasoner = replayed(shown["ticks"], "changes"), replayed(shown["ticks"], "reasoner")
        assert own and reasoner
        for k in frame_keys(data):
            at = position_of(k)
            want_own = final.own_map.at(at)
            want_reasoner = reasoner_beliefs(final.reasoner, at)
            got_own = own.get(k, shown["untried"]["own"])
            got_reasoner = reasoner.get(k, shown["untried"]["reasoner"])
            assert got_own == pytest.approx({b.value: want_own[b] for b in Belief}, abs=1e-4)
            assert got_reasoner == pytest.approx(
                {b.value: want_reasoner[b] for b in Belief}, abs=1e-4
            )


def test_each_tick_is_its_record(runs: tuple[Path, Path], data: dict[str, Any]) -> None:
    for run_dir, shown in zip(runs, data["runs"], strict=True):
        dumps = load_dumps(run_dir)
        first = dumps[0].record.felt_before
        assert shown["start"]["at"] == list(first.at)
        for dump, t in zip(dumps, shown["ticks"], strict=True):
            r = dump.record
            assert t["at"] == list(r.felt.at)
            assert t["direction"] == r.decision.direction.value
            assert t["target"] == list(r.outcome.info.target)
            assert t["felt"]["moved"] == r.felt.moved
            assert t["expected"]["moved"] == r.decision.expectation.moved
            assert t["misses"]["moved"] == r.comparison.moved_miss
            assert t["misses"]["hunger"] == r.comparison.misses[Channel.HUNGER]
            assert t["reading"] == r.appraisal.reading
            assert t["wrong"] == list(r.measures.wrong)
            assert [c["at"] for c in t["reasoner"]] == [list(at) for at in r.reasoner_changed]


def test_feelings_are_words(data: dict[str, Any]) -> None:
    def check_felt(felt: dict[str, Any]) -> None:
        assert all(felt[c.value] in FEEL_WORDS for c in Channel)
        assert felt["smell"] in NEAR_WORDS and felt["damp"] in NEAR_WORDS

    for shown in data["runs"]:
        start = shown["start"]
        check_felt(start["felt"])
        assert start["tone"] in TONE_WORDS and start["stir"] in STIR_WORDS
        for t in shown["ticks"]:
            check_felt(t["felt"])
            check_felt(t["expected"])
            assert t["tone"] in TONE_WORDS and t["stir"] in STIR_WORDS
            assert t["surprise"] in SURPRISE_WORDS
            assert t["it_went"] in (WENT_WORSE, WENT_BETTER, WENT_SAME)
    ladders = data["ladders"]
    assert ladders["feel"] == list(FEEL_WORDS) and ladders["near"] == list(NEAR_WORDS)


def numbers(value: Any) -> list[float]:
    if isinstance(value, dict):
        return [n for v in value.values() for n in numbers(v)]
    if isinstance(value, list):
        return [n for v in value for n in numbers(v)]
    if isinstance(value, float):
        return [value]
    return []


def test_no_number_has_more_than_four_decimals(data: dict[str, Any]) -> None:
    found = numbers(data)
    assert found
    assert all(round(n, 4) == n for n in found)


def test_the_data_holds_no_path_of_this_machine(runs_dir: Path, page: str) -> None:
    assert str(runs_dir) not in page
    assert str(Path.home()) not in page


# --- the page -----------------------------------------------------------------------------------


def test_the_page_embeds_the_data_once_and_reaches_nothing_outside(
    data: dict[str, Any], page: str
) -> None:
    assert page.count('<script type="application/json"') == 1
    assert embedded(page) == data
    assert "http://" not in page
    assert "https://" not in page
    assert "@@FILM_DATA@@" not in page


# --- runs that do not belong together -----------------------------------------------------------


def test_runs_on_different_layouts_are_refused(runs: tuple[Path, Path], tmp_path: Path) -> None:
    world = run_config("plain").world
    other = world.model_copy(update={"layout": (*LAYOUT[:-1], "....F#..P")})
    with pytest.raises(RunError, match="layout"):
        film_data(runs[0], stored_run(tmp_path, "plain", world=other))


def test_runs_of_different_lengths_are_refused(runs: tuple[Path, Path], tmp_path: Path) -> None:
    config = run_config("plain", ticks=TICKS - 10)
    asyncio.run(run(config, tmp_path / config.run_id))
    with pytest.raises(RunError, match="ticks"):
        film_data(runs[0], tmp_path / config.run_id)


def test_a_run_with_no_tick_is_refused(runs: tuple[Path, Path], tmp_path: Path) -> None:
    config = run_config("plain", ticks=TICKS)
    asyncio.run(run(config, tmp_path / config.run_id, stop_after_tick=0))
    assert (tmp_path / config.run_id / "config.json").is_file()
    with pytest.raises(RunError, match="no tick"):
        film_data(tmp_path / config.run_id, runs[1])


def test_the_same_run_twice_is_refused(runs: tuple[Path, Path]) -> None:
    with pytest.raises(RunError, match="twice"):
        film_data(runs[0], runs[0])
    with pytest.raises(RunError, match="twice"):
        film_data(runs[0], runs[0].parent / "." / runs[0].name)


# --- the command --------------------------------------------------------------------------------


def one_error_line(capsys: pytest.CaptureFixture[str]) -> str:
    out, err = capsys.readouterr()
    assert out == ""
    assert err.count("\n") == 1 and err.startswith("error: "), err
    return err


def test_the_command_writes_the_film(
    runs: tuple[Path, Path], page: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "film.html"
    assert main(["film", str(runs[0]), str(runs[1]), "-o", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == page
    assert capsys.readouterr().out == ""


def test_the_command_refuses_a_missing_run_in_one_line(
    runs: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "film.html"
    assert main(["film", str(runs[0]), str(tmp_path / "nope"), "-o", str(out)]) == 2
    one_error_line(capsys)
    assert not out.exists()


def test_the_command_refuses_an_unwritable_output_in_one_line(
    runs: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "missing" / "film.html"
    assert main(["film", str(runs[0]), str(runs[1]), "-o", str(out)]) == 2
    assert "cannot write" in one_error_line(capsys)
    assert not out.exists()
