"""The ladder check: the swept bodies, the prompts, the pass mark and the resumable sweep.

No test reaches a real model: the sweep runs on a `FunctionModel` that reads the lettered actions
from the prompt, and the command is run only over answers already on disk.
"""

import asyncio
import json
import re
from pathlib import Path

import pytest
from pydantic_ai import ModelResponse, ToolCallPart
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from surprise_loop.agents import load_model, model_settings
from surprise_loop.cli import main
from surprise_loop.config import MODEL_ENV_VAR, LadderParams, LLMParams, Params
from surprise_loop.feeling import felt
from surprise_loop.ladder import (
    ANSWERS_FILE,
    LETTERS,
    SCENES,
    LadderAnswer,
    answer_order,
    ladder_body,
    ladder_prompt,
    read_answers,
    run_ladder,
    summarize,
)
from surprise_loop.phrases import LADDER_ACTIONS, all_phrases
from surprise_loop.world import Channel

LP = LadderParams()
LEVELS = range(len(LP.heat_levels))
SAMPLES = 3  # small, so the sweep runs fast; the real check uses LadderParams().samples


# --- the swept bodies ---------------------------------------------------------------------------


def test_the_heat_strains_of_the_seven_levels() -> None:
    strains = [felt(*ladder_body(level, "A", LP)[:2])[Channel.HEAT] for level in LEVELS]
    assert strains == pytest.approx([-1.2, -0.6, 0.0, 0.6, 1.5, 2.4, 3.3])


def test_scene_a_has_no_thirst_and_no_tiredness() -> None:
    for level in LEVELS:
        strength = felt(*ladder_body(level, "A", LP)[:2])
        assert strength[Channel.WATER] == pytest.approx(0)
        assert strength[Channel.ENERGY] == pytest.approx(0)


def test_scene_b_thirst_and_tiredness() -> None:
    for level in LEVELS:
        strength = felt(*ladder_body(level, "B", LP)[:2])
        assert strength[Channel.WATER] == pytest.approx(1.5)
        assert strength[Channel.ENERGY] == pytest.approx(1.2)


def test_the_swept_body_is_the_protocols() -> None:
    belief, t, arousal = ladder_body(4, "B", LP)
    assert arousal == 0.40
    assert set(t.habit.values()) == {0.3}
    assert t.setpoint.model_dump() == {"heat": 0.0, "water": 0.80, "energy": 0.70}
    assert belief.model_dump() == {"heat": 0.5, "water": 0.50, "energy": 0.46}


# --- the prompts --------------------------------------------------------------------------------


@pytest.mark.parametrize("scene", SCENES)
@pytest.mark.parametrize("level", LEVELS)
def test_a_prompt_has_no_digit(level: int, scene: str) -> None:
    for i in range(LP.samples):
        text = ladder_prompt(level, scene, answer_order(i), Params().lens, LP)
        assert not re.search(r"\d", text)
        assert "_" not in text


def test_a_prompt_is_the_scene_then_the_question_then_four_lettered_actions() -> None:
    order = ["look", "sit", "stream", "shade"]
    lines = ladder_prompt(5, "A", order, Params().lens, LP).split("\n")
    assert lines[0].startswith("You are standing in the middle of the forest clearing.")
    assert lines[1:3] == ["", "What do you do next?"]
    assert lines[3:] == [
        f"{letter}. {LADDER_ACTIONS[a]}" for letter, a in zip(LETTERS, order, strict=True)
    ]


def test_the_ladder_text_is_in_the_phrase_table() -> None:
    phrases = all_phrases()
    assert set(LADDER_ACTIONS) == {"shade", "stream", "sit", "look"}
    for text in LADDER_ACTIONS.values():
        assert text in phrases
    assert LADDER_ACTIONS["shade"] == "Walk to the shade of the oak, a few steps away."
    assert LADDER_ACTIONS["stream"] == "Walk to the stream, a longer walk across the open ground."


def test_the_order_depends_only_on_the_answer_index() -> None:
    orders = [answer_order(i) for i in range(LP.samples)]
    assert orders == [answer_order(i) for i in range(LP.samples)]
    assert all(sorted(o) == sorted(LADDER_ACTIONS) for o in orders)
    assert len({tuple(o) for o in orders}) > 1


# --- the pass mark ------------------------------------------------------------------------------


def answers_with_shares(shares: list[float], n: int = 10, scene: str = "A") -> list[LadderAnswer]:
    """`n` hand-built answers per level, the first `share * n` of them choosing the shade."""
    out = []
    order = ["shade", "stream", "sit", "look"]
    for level, share in enumerate(shares):
        for i in range(n):
            action = "shade" if i < round(share * n) else "stream"
            out.append(
                LadderAnswer(
                    scene=scene,
                    level=level,
                    i=i,
                    order=order,
                    letter=LETTERS[order.index(action)],
                    action=action,
                    why="Because.",
                )
            )
    return out


def test_a_clean_step_at_level_three_passes() -> None:
    curve = summarize(answers_with_shares([0, 0, 0, 1, 1, 1, 1]), LP)["A"]
    assert curve.share == [0, 0, 0, 1, 1, 1, 1]
    assert curve.slope == pytest.approx(6 / 28)
    assert curve.range == 1
    assert curve.worst_drop == 0
    assert curve.passed


def test_a_flat_curve_fails_on_slope() -> None:
    curve = summarize(answers_with_shares([0.5] * 7), LP)["A"]
    assert curve.slope == pytest.approx(0)
    assert not curve.passed


def test_a_curve_with_one_drop_of_three_tenths_fails_on_worst_drop() -> None:
    curve = summarize(answers_with_shares([0, 0.3, 0.6, 0.3, 0.7, 1, 1]), LP)["A"]
    assert curve.slope >= LP.min_slope
    assert curve.range >= LP.min_range
    assert curve.worst_drop == pytest.approx(0.3)
    assert not curve.passed


def test_a_curve_with_range_four_tenths_fails() -> None:
    curve = summarize(answers_with_shares([0, 0.1, 0.2, 0.2, 0.3, 0.4, 0.4]), LP)["A"]
    assert curve.slope >= LP.min_slope
    assert curve.worst_drop < LP.max_drop
    assert curve.range == pytest.approx(0.4)
    assert not curve.passed


def test_summarize_keys_each_scene() -> None:
    answers = answers_with_shares([0] * 7, scene="A") + answers_with_shares([1] * 7, scene="B")
    curves = summarize(answers, LP)
    assert set(curves) == {"A", "B"}
    assert curves["A"].share == [0] * 7
    assert curves["B"].share == [1] * 7


# --- the sweep ----------------------------------------------------------------------------------


def user_prompt(messages: list[ModelMessage]) -> str:
    first = messages[0]
    assert isinstance(first, ModelRequest)
    parts = [p for p in first.parts if isinstance(p, UserPromptPart)]
    assert len(parts) == 1 and isinstance(parts[0].content, str)
    return parts[0].content


def letter_of(prompt: str, action: str) -> str:
    wanted = LADDER_ACTIONS[action]
    for line in prompt.split("\n"):
        letter, _, text = line.partition(". ")
        if text == wanted and letter in LETTERS:
            return letter
    raise AssertionError(f"{action!r} is not offered in: {prompt}")


def always(action: str, calls: list[str] | None = None) -> FunctionModel:
    """A model that always answers the letter of `action`; records the prompts it saw."""

    def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = user_prompt(messages)
        if calls is not None:
            calls.append(prompt)
        args = {"letter": letter_of(prompt, action), "why": "It is the  obvious\nthing."}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])

    return FunctionModel(answer)


def sweep(model: FunctionModel, out: Path, samples: int = SAMPLES, jobs: int = 4) -> list:
    done = read_answers(out / ANSWERS_FILE)
    return asyncio.run(run_ladder(model, samples, jobs, out, Params(), done))


def test_a_sweep_writes_every_answer_and_a_rerun_adds_none(tmp_path: Path) -> None:
    calls: list[str] = []
    new = sweep(always("shade", calls), tmp_path)
    assert len(new) == len(calls) == len(SCENES) * len(LEVELS) * SAMPLES
    stored = read_answers(tmp_path / ANSWERS_FILE)
    assert sorted(stored, key=lambda a: (a.scene, a.level, a.i)) == sorted(
        new, key=lambda a: (a.scene, a.level, a.i)
    )
    assert {(a.scene, a.level, a.i) for a in stored} == {
        (s, level, i) for s in SCENES for level in LEVELS for i in range(SAMPLES)
    }
    assert all(a.action == "shade" for a in stored)
    assert all(a.order[LETTERS.index(a.letter)] == "shade" for a in stored)
    assert all(a.why == "It is the obvious thing." for a in stored)

    again: list[str] = []
    assert sweep(always("shade", again), tmp_path) == []
    assert again == []
    assert len(read_answers(tmp_path / ANSWERS_FILE)) == len(stored)


def test_every_cell_sees_the_same_orders(tmp_path: Path) -> None:
    sweep(always("look"), tmp_path)
    for a in read_answers(tmp_path / ANSWERS_FILE):
        assert a.order == answer_order(a.i)


def test_a_rerun_asks_only_for_what_is_missing(tmp_path: Path) -> None:
    sweep(always("shade"), tmp_path)
    path = tmp_path / ANSWERS_FILE
    lines = path.read_text().splitlines()
    kept = [line for line in lines if json.loads(line)["level"] != 4]
    path.write_text("".join(line + "\n" for line in kept))
    calls: list[str] = []
    new = sweep(always("shade", calls), tmp_path)
    assert {(a.scene, a.level) for a in new} == {(s, 4) for s in SCENES}
    assert len(calls) == len(SCENES) * SAMPLES
    assert len(read_answers(path)) == len(lines)


def test_a_failed_call_is_reported_and_not_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def flaky(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = user_prompt(messages)
        if "Your skin burns" in prompt:
            raise RuntimeError("the provider fell over\nbadly")
        args = {"letter": letter_of(prompt, "shade"), "why": "Shade."}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])

    new = sweep(FunctionModel(flaky), tmp_path, samples=1)
    stored = read_answers(tmp_path / ANSWERS_FILE)
    assert len(new) == len(stored) < len(SCENES) * len(LEVELS)
    err = capsys.readouterr().err.strip().splitlines()
    assert err, "each failed call is reported"
    assert len(err) == len(SCENES) * len(LEVELS) - len(stored)
    assert all("the provider fell over badly" in line for line in err)


def test_at_most_jobs_calls_run_at_once(tmp_path: Path) -> None:
    running = 0
    most = 0

    async def slow(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal running, most
        running += 1
        most = max(most, running)
        await asyncio.sleep(0.001)
        running -= 1
        args = {"letter": "A", "why": "Fine."}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])

    sweep(FunctionModel(slow), tmp_path, samples=2, jobs=3)
    assert most == 3


def test_an_answer_off_the_list_is_retried(tmp_path: Path) -> None:
    calls = 0

    def wrong_then_right(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        letter = "E" if len(messages) == 1 else "B"
        args = {"letter": letter, "why": "Fine."}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])

    new = sweep(FunctionModel(wrong_then_right), tmp_path, samples=1)
    assert len(new) == len(SCENES) * len(LEVELS)
    assert calls == 2 * len(new)
    assert all(a.letter == "B" for a in new)


def test_a_broken_answers_file_names_its_line(tmp_path: Path) -> None:
    path = tmp_path / ANSWERS_FILE
    path.write_text('{"scene": "A"}\n')
    with pytest.raises(ValueError, match=r"answers\.jsonl: line 1"):
        read_answers(path)


def test_no_answers_file_is_no_answers(tmp_path: Path) -> None:
    assert read_answers(tmp_path / ANSWERS_FILE) == []


# --- agents.py ----------------------------------------------------------------------------------


def test_model_settings_carry_the_llm_params() -> None:
    assert model_settings(LLMParams(temperature=0.7, timeout_s=12)) == {
        "timeout": 12,
        "temperature": 0.7,
    }
    assert model_settings(LLMParams(temperature=None)) == {"timeout": 60.0}


def test_load_model() -> None:
    assert load_model(None) is None
    assert load_model("test") is not None


# --- the command --------------------------------------------------------------------------------


def run_command(out: Path, *extra: str) -> int:
    return main(["ladder-check", "--out", str(out), "--samples", str(SAMPLES), *extra])


def test_the_command_fails_on_flat_curves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sweep(always("shade"), tmp_path)
    monkeypatch.setenv(MODEL_ENV_VAR, "test")
    assert run_command(tmp_path) != 0
    out = capsys.readouterr().out
    assert "scene A" in out and "scene B" in out
    assert "FAILED" in out
    assert len(read_answers(tmp_path / ANSWERS_FILE)) == len(SCENES) * len(LEVELS) * SAMPLES
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["model"] == "test"
    assert summary["samples"] == SAMPLES
    assert summary["params"]["ladder"]["samples"] == LP.samples
    assert summary["scenes"]["A"]["share"] == [1.0] * 7
    assert summary["scenes"]["A"]["passed"] is False


def test_the_command_passes_on_rising_curves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / ANSWERS_FILE
    rising = [0, 0, 0, 1, 1, 1, 1]
    answers = answers_with_shares(rising, SAMPLES, "A") + answers_with_shares(rising, SAMPLES, "B")
    path.write_text("".join(a.model_dump_json() + "\n" for a in answers))
    monkeypatch.setenv(MODEL_ENV_VAR, "test")
    assert run_command(tmp_path) == 0
    assert "passed" in capsys.readouterr().out


def test_the_command_names_short_cells_and_gives_no_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # TestModel answers every call, so the cells are short only because the file stops early:
    # run the sweep with fewer samples, then ask the command for more with a model that fails.
    sweep(always("shade"), tmp_path, samples=SAMPLES)
    monkeypatch.setenv(MODEL_ENV_VAR, "test")
    monkeypatch.setattr("surprise_loop.cli.load_model", lambda name: FunctionModel(_fail))
    code = main(["ladder-check", "--out", str(tmp_path), "--samples", str(SAMPLES + 1)])
    assert code != 0
    captured = capsys.readouterr()
    assert "scene A level 0" in captured.err
    assert "FAILED" not in captured.out and "passed" not in captured.out
    assert not (tmp_path / "summary.json").exists()


def _fail(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    raise RuntimeError("no answer")


def test_the_command_needs_a_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
    assert run_command(tmp_path) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and MODEL_ENV_VAR in err
    assert len(err.strip().splitlines()) == 1


def test_the_configs_model_and_params_win(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "ladder.yaml"
    config.write_text(
        "name: x\nseeds: [1]\nmodel: test\n"
        "agents:\n"
        "  - name: a\n"
        "    temperament: {setpoint: {heat: 0, water: 0.8, energy: 0.7},\n"
        "      tightness: {heat: 3, water: 5, energy: 5},\n"
        "      trust: {heat: 0.5, water: 0.5, energy: 0.5},\n"
        "      habit: {burden: 0.3, relief: 0.3, ambient: 0.3}, baseline: 0}\n"
        "    body: {heat: 0, water: 0.8, energy: 0.7}\n"
        "    belief: {heat: 0, water: 0.8, energy: 0.7}\n"
        "params: {ladder: {samples: 1, min_slope: -1, min_range: -1, max_drop: 2}}\n"
    )
    monkeypatch.setenv(MODEL_ENV_VAR, "no-such-provider:nothing")
    out = tmp_path / "out"
    assert main(["ladder-check", "--config", str(config), "--out", str(out)]) == 0
    assert len(read_answers(out / ANSWERS_FILE)) == len(SCENES) * len(LEVELS)
    assert json.loads((out / "summary.json").read_text())["model"] == "test"
    capsys.readouterr()
