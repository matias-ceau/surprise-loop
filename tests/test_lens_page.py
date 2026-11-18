import itertools
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest
from bodies import ILO_TEXT, TAVI_TEXT

from surprise_loop.cli import main
from surprise_loop.config import ConfigError, ExperimentConfig, load_config
from surprise_loop.lens import felt_scene
from surprise_loop.viewer.lens_page import (
    AROUSAL_GRID,
    ENERGY_GRID,
    HEAT_GRID,
    WATER_GRID,
    lens_page_html,
)
from surprise_loop.world import Levels, Place, frozen_noon_facts

ROOT = Path(__file__).resolve().parent.parent
AFTERNOON = ROOT / "experiments" / "clearing_afternoon.yaml"
PRESETS = ["worked example", "ordinary pair", "same body, two temperaments"]
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "wbr"}


@pytest.fixture(scope="module")
def config() -> ExperimentConfig:
    return load_config(AFTERNOON)


@pytest.fixture(scope="module")
def page(config: ExperimentConfig) -> str:
    return lens_page_html(config)


@pytest.fixture(scope="module")
def data(page: str) -> dict[str, Any]:
    match = re.search(
        r'<script type="application/json" id="lens-data">(.*?)</script>', page, flags=re.S
    )
    assert match is not None, "no embedded lens data"
    return json.loads(match.group(1))


def all_entries(data: dict[str, Any]) -> list[dict[str, Any]]:
    entries = [e for agent in data["agents"] for e in agent["entries"]]
    entries += [body["entry"] for preset in data["presets"] for body in preset["bodies"]]
    return entries


class _Structure(HTMLParser):
    """Checks that every opened element is closed in order, outside raw-text elements."""

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.seen: list[str] = []
        self.doctype = False
        self.errors: list[str] = []

    def handle_decl(self, decl: str) -> None:
        self.doctype = self.doctype or decl.lower() == "doctype html"

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.seen.append(tag)
        if tag not in VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.seen.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"</{tag}> closes {self.stack[-1] if self.stack else 'nothing'}")
            return
        self.stack.pop()


def test_page_parses_as_a_document(page: str) -> None:
    parser = _Structure()
    parser.feed(page)
    parser.close()
    assert parser.doctype
    assert parser.errors == []
    assert parser.stack == []
    for tag in ("html", "head", "title", "body", "script"):
        assert tag in parser.seen


def test_page_reaches_for_nothing_outside_itself(page: str) -> None:
    assert "http://" not in page
    assert "https://" not in page
    assert "//" not in page
    assert not re.search(r"<script[^>]*\ssrc", page, flags=re.I)
    assert not re.search(r"<link\b", page, flags=re.I)
    assert "@import" not in page
    assert all(ref.startswith("url(#") for ref in re.findall(r"url\([^)]*", page))


def test_data_is_embedded_once_and_safely(page: str, data: dict[str, Any]) -> None:
    assert page.count('type="application/json"') == 1
    body = page.split('id="lens-data">', 1)[1].split("</script>", 1)[0]
    assert "</" not in body
    assert data["place"]["opening"].startswith("You are standing")


def test_the_grid_is_the_briefs(data: dict[str, Any]) -> None:
    assert len(HEAT_GRID) == 18
    assert HEAT_GRID[0] == -0.5 and HEAT_GRID[-1] == 1.2
    assert all(round(b - a, 6) == 0.1 for a, b in itertools.pairwise(HEAT_GRID))
    assert WATER_GRID == ENERGY_GRID == (0.25, 0.40, 0.55, 0.70, 0.85)
    assert AROUSAL_GRID == (0.2, 0.4, 0.7)
    assert data["grid"] == {
        "heat": list(HEAT_GRID),
        "water": list(WATER_GRID),
        "energy": list(ENERGY_GRID),
        "arousal": list(AROUSAL_GRID),
    }


def test_every_grid_point_of_both_agents_has_its_entry(
    config: ExperimentConfig, data: dict[str, Any]
) -> None:
    facts = frozen_noon_facts()
    points = list(itertools.product(HEAT_GRID, WATER_GRID, ENERGY_GRID, AROUSAL_GRID))
    assert [a["name"] for a in data["agents"]] == ["ilo", "tavi"]
    for agent, spec in zip(data["agents"], config.agents, strict=True):
        assert len(agent["entries"]) == len(points)
        for entry, (heat, water, energy, arousal) in zip(agent["entries"], points, strict=True):
            scene = felt_scene(
                Place.MEADOW,
                facts,
                Levels(heat=heat, water=water, energy=energy),
                spec.temperament,
                arousal,
                None,
                None,
                0,
                config.params.lens,
            )
            assert entry["text"] == scene.text()
            assert [f["name"] for f in entry["facts"]] == [f.name for f in scene.facts]


def test_each_entry_carries_what_the_page_needs_and_no_more(data: dict[str, Any]) -> None:
    for entry in all_entries(data):
        assert set(entry) == {"text", "facts", "body", "felt", "surprise"}
        for fact in entry["facts"]:
            assert set(fact) == {"name", "weight", "value", "sign", "grade", "noticed", "phrase"}
        for line in entry["body"]:
            assert set(line) == {"channel", "grade", "felt", "phrase"}
        assert set(entry["felt"]) == {"heat", "water", "energy"}


def test_the_worked_example_carries_exactly_the_fixed_texts(
    config: ExperimentConfig, data: dict[str, Any]
) -> None:
    worked = next(p for p in data["presets"] if p["name"] == "worked example")
    ilo, tavi = worked["bodies"]
    assert ilo["entry"]["text"] == ILO_TEXT
    assert tavi["entry"]["text"] == TAVI_TEXT
    for body, spec in zip(worked["bodies"], config.agents, strict=True):
        assert body["belief"] == spec.belief.model_dump()
        assert body["arousal"] == spec.arousal


def test_the_three_presets_are_present(data: dict[str, Any]) -> None:
    assert [p["name"] for p in data["presets"]] == PRESETS
    ordinary = data["presets"][1]["bodies"]
    assert ordinary[0]["belief"] == {"heat": 0.40, "water": 0.78, "energy": 0.66}
    assert ordinary[0]["arousal"] == 0.4
    assert ordinary[1]["belief"] == {"heat": 0.25, "water": 0.80, "energy": 0.70}
    assert ordinary[1]["arousal"] == 0.2
    same = data["presets"][2]["bodies"]
    for body in same:
        assert body["belief"] == {"heat": 0.3, "water": 0.70, "energy": 0.55}
        assert body["arousal"] == 0.4
    # the same belief read through two temperaments does not read the same
    assert same[0]["entry"]["text"] != same[1]["entry"]["text"]


def test_noticed_facts_are_exactly_those_with_a_phrase(data: dict[str, Any]) -> None:
    for entry in all_entries(data):
        for fact in entry["facts"]:
            assert fact["noticed"] == (fact["phrase"] is not None)


def test_no_number_in_any_lens_text(data: dict[str, Any]) -> None:
    for entry in all_entries(data):
        assert not re.search(r"\d", entry["text"])


def test_the_page_is_under_six_megabytes(page: str) -> None:
    assert len(page.encode("utf-8")) < 6_000_000


def test_lens_page_needs_two_agents(config: ExperimentConfig) -> None:
    one = config.model_copy(update={"agents": config.agents[:1]})
    with pytest.raises(ConfigError, match="two agents"):
        lens_page_html(one)


def test_cli_writes_the_page(tmp_path: Path, page: str) -> None:
    out = tmp_path / "page.html"
    assert main(["lens-page", str(AFTERNOON), "-o", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == page


def test_cli_writes_lens_page_html_here_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["lens-page", str(AFTERNOON)]) == 0
    assert (tmp_path / "lens_page.html").is_file()


def test_cli_missing_config_is_one_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "page.html"
    code = main(["lens-page", str(tmp_path / "nope.yaml"), "-o", str(out)])
    assert code != 0
    err = capsys.readouterr().err
    assert err.count("\n") == 1
    assert err.startswith("error: ") and "file not found" in err
    assert "Traceback" not in err
    assert not out.exists()


def test_cli_write_error_without_strerror_is_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(self: Path, *args: object, **kwargs: object) -> int:
        raise OSError("the disk went away")  # an OSError with no errno: its strerror is None

    out = tmp_path / "page.html"
    monkeypatch.setattr(Path, "write_text", refuse)
    assert main(["lens-page", str(AFTERNOON), "-o", str(out)]) == 2
    err = capsys.readouterr().err
    assert err == f"error: {out}: cannot write: the disk went away\n"


def _script_function(page: str, name: str) -> str:
    """The source of one top-level function in the page's script, up to its closing brace."""
    match = re.search(rf"^function {name}\(.*?^\}}", page, flags=re.S | re.M)
    assert match is not None, f"no function {name} in the page"
    return match.group(0)


def test_one_bodys_control_leaves_only_that_bodys_preset(page: str) -> None:
    leave = _script_function(page, "leaveExact")
    assert "body.exact = null" in leave
    assert "bodies" not in leave  # no loop over both bodies
    assert "renderAll" not in leave  # only this body is redrawn
    assert "render(body)" in leave
    assert "activePreset = -1" in leave


def test_a_zero_felt_strength_names_no_direction(page: str) -> None:
    why = _script_function(page, "bodyWhy")
    assert "felt >= 0" not in why
    assert "FELT_ZERO" in why
    assert "const FELT_ZERO = 0.005;" in page  # half the last shown decimal of signed(felt, 2)
