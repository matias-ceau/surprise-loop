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

from surprise_loop.agents import deliberate_prompt, load_model, model_settings
from surprise_loop.cli import main
from surprise_loop.config import MODEL_ENV_VAR, LadderParams, LLMParams, Params, Scene
from surprise_loop.feeling import felt
from surprise_loop.ladder import (
    ANSWERS_FILE,
    DELIBERATE_ANSWERS_FILE,
    DELIBERATE_SUMMARY_FILE,
    LADDER_ACTION_OF,
    LETTERS,
    SCENES,
    SUMMARY_FILE,
    LadderAnswer,
    answer_order,
    current_answers,
    deliberate_ladder_prompt,
    ladder_body,
    ladder_prompt,
    read_answers,
    run_ladder,
    stored_prompt,
    summarize,
)
from surprise_loop.lens import felt_scene
from surprise_loop.phrases import (
    DELIBERATE_INSTRUCTION,
    LADDER_ACTIONS,
    LADDER_INSTRUCTION,
    OFFER_PHRASES,
    STUB_READINGS,
    all_phrases,
)
from surprise_loop.state import Offer
from surprise_loop.world import Action, Channel, Place, frozen_noon_facts, offers_at

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


def answers_with_shares(
    shares: list[float], n: int = 10, scene: Scene = "A", model: str = "test"
) -> list[LadderAnswer]:
    """`n` hand-built answers per level, the first `share * n` of them choosing the shade, each
    with the order and prompt the current code builds for its slot."""
    return [
        answer(scene, level, i, "shade" if i < round(share * n) else "stream", model)
        for level, share in enumerate(shares)
        for i in range(n)
    ]


def answer(scene: Scene, level: int, i: int, action: str, model: str = "test") -> LadderAnswer:
    order = answer_order(i)
    return LadderAnswer(
        scene=scene,
        level=level,
        i=i,
        order=order,
        letter=LETTERS[order.index(action)],
        action=action,
        why="Because.",
        model=model,
        prompt=stored_prompt(ladder_prompt(level, scene, order, Params().lens, LP), False),
    )


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


def sweep(
    model: FunctionModel, out: Path, samples: int = SAMPLES, jobs: int = 4, name: str = "test"
) -> list:
    """One run of the sweep over what is stored in `out`, its answers recorded under `name`."""
    done = read_answers(out / ANSWERS_FILE)
    return asyncio.run(run_ladder(model, samples, jobs, out, Params(), done, name))


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
    assert all(a.model == "test" for a in stored)
    assert sorted(a.prompt for a in stored) == sorted(stored_prompt(c, False) for c in calls)

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
    assert run_command(tmp_path) == 1
    out = capsys.readouterr().out
    assert "scene A" in out and "scene B" in out
    assert "FAILED" in out
    assert len(read_answers(tmp_path / ANSWERS_FILE)) == len(SCENES) * len(LEVELS) * SAMPLES
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["model"] == "test"
    assert summary["samples"] == SAMPLES
    assert summary["params"]["ladder"]["samples"] == SAMPLES  # the samples used, not the config's
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
    assert code == 2
    captured = capsys.readouterr()
    last = captured.err.strip().splitlines()[-1]
    assert last.startswith("error: no verdict") and "scene A level 0" in last
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


# --- stale answers ------------------------------------------------------------------------------


def test_answers_from_another_model_are_ignored_and_asked_again(tmp_path: Path) -> None:
    sweep(always("stream"), tmp_path, name="other:model")
    calls: list[str] = []
    new = sweep(always("shade", calls), tmp_path, name="test")
    every = len(SCENES) * len(LEVELS) * SAMPLES
    assert len(new) == len(calls) == every
    stored = read_answers(tmp_path / ANSWERS_FILE)
    assert len(stored) == 2 * every  # stale lines stay in the file
    current, stale = current_answers(stored, "test", SAMPLES, Params())
    assert stale == every
    assert len(current) == every and all(a.action == "shade" for a in current)
    assert summarize(current, LP)["A"].share == [1.0] * 7


def test_answers_to_a_changed_prompt_are_ignored_and_asked_again(tmp_path: Path) -> None:
    sweep(always("shade"), tmp_path)
    path = tmp_path / ANSWERS_FILE
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        if row["level"] == 4:
            row["prompt"] = row["prompt"].replace("It is hot.", "The air is warm.")
            row["action"], row["letter"] = "look", "ABCD"[row["order"].index("look")]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert current_answers(read_answers(path), "test", SAMPLES, Params())[1] == 2 * SAMPLES

    calls: list[str] = []
    new = sweep(always("shade", calls), tmp_path)
    assert {(a.scene, a.level) for a in new} == {(s, 4) for s in SCENES}
    assert len(calls) == len(SCENES) * SAMPLES
    current, stale = current_answers(read_answers(path), "test", SAMPLES, Params())
    assert stale == len(SCENES) * SAMPLES
    assert all(a.action == "shade" for a in current)


def test_the_last_current_answer_for_a_slot_wins() -> None:
    first, last = answer("A", 2, 0, "sit"), answer("A", 2, 0, "look")
    current, stale = current_answers([first, last], "test", SAMPLES, Params())
    assert current == [last]
    assert stale == 0


def test_answers_outside_the_sweep_are_neither_counted_nor_stale() -> None:
    beyond = answer("A", 2, SAMPLES, "sit")
    assert current_answers([beyond], "test", SAMPLES, Params()) == ([], 0)


def test_the_command_reports_stale_answers_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sweep(always("shade"), tmp_path, name="other:model")
    monkeypatch.setenv(MODEL_ENV_VAR, "test")
    assert run_command(tmp_path) == 1  # TestModel always answers A: flat curves
    err = capsys.readouterr().err
    every = len(SCENES) * len(LEVELS) * SAMPLES
    stale_lines = [line for line in err.splitlines() if "stale answers" in line]
    assert len(stale_lines) == 1 and f"ignored {every} stale answers" in stale_lines[0]
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["scenes"]["A"]["share"] != [1.0] * 7  # the other model's answers do not count


# --- answers files written by a killed run ------------------------------------------------------


def test_a_partly_written_last_line_is_dropped(tmp_path: Path) -> None:
    sweep(always("shade"), tmp_path, samples=1)
    path = tmp_path / ANSWERS_FILE
    whole = read_answers(path)
    lines = path.read_text().splitlines()
    path.write_text("".join(line + "\n" for line in lines) + lines[0][: len(lines[0]) // 2])
    assert read_answers(path) == whole


def test_a_rerun_after_a_partly_written_line_appends_whole_lines(tmp_path: Path) -> None:
    sweep(always("shade"), tmp_path, samples=1)
    path = tmp_path / ANSWERS_FILE
    lines = path.read_text().splitlines()
    path.write_text("".join(line + "\n" for line in lines[:-2]) + lines[-2][:15])
    calls: list[str] = []
    new = sweep(always("shade", calls), tmp_path, samples=1)
    assert len(new) == len(calls) == 2
    assert len(read_answers(path)) == len(lines)
    assert path.read_text().endswith("\n")


def test_a_partial_tail_is_cut_without_rewriting_the_lines_before_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sweep(always("shade"), tmp_path, samples=1)
    path = tmp_path / ANSWERS_FILE
    lines = path.read_text().splitlines()
    whole = "".join(line + "\n" for line in lines[:-1])
    path.write_text(whole + lines[-1][:15])

    def no_rewrite(*args: object, **kwargs: object) -> None:
        raise AssertionError("the answers file must not be rewritten")

    monkeypatch.setattr(Path, "write_text", no_rewrite)
    monkeypatch.setattr(Path, "write_bytes", no_rewrite)
    calls: list[str] = []
    new = sweep(always("shade", calls), tmp_path, samples=1)
    assert len(new) == 1
    assert path.read_text().startswith(whole)  # every byte before the cut is untouched


def test_a_whole_last_line_without_its_newline_is_kept(tmp_path: Path) -> None:
    sweep(always("shade"), tmp_path, samples=1)
    path = tmp_path / ANSWERS_FILE
    text = path.read_text()
    path.write_text(text.rstrip("\n"))
    assert len(read_answers(path)) == len(text.splitlines())
    calls: list[str] = []
    assert sweep(always("shade", calls), tmp_path, samples=1) == []
    lines = path.read_text().splitlines()
    path.write_text("".join(line + "\n" for line in lines[:-1]))
    sweep(always("shade"), tmp_path, samples=1)
    assert len(read_answers(path)) == len(lines)


def test_a_bad_last_line_that_ends_in_a_newline_is_an_error(tmp_path: Path) -> None:
    sweep(always("shade"), tmp_path, samples=1)
    path = tmp_path / ANSWERS_FILE
    lines = path.read_text().splitlines()
    path.write_text("".join(line + "\n" for line in lines) + lines[0][:10] + "\n")
    with pytest.raises(ValueError, match=rf"line {len(lines) + 1}"):
        read_answers(path)


def test_a_bad_line_before_the_last_is_still_an_error(tmp_path: Path) -> None:
    sweep(always("shade"), tmp_path, samples=1)
    path = tmp_path / ANSWERS_FILE
    lines = path.read_text().splitlines()
    lines.insert(1, lines[0][:10])
    path.write_text("".join(line + "\n" for line in lines))
    with pytest.raises(ValueError, match=r"answers\.jsonl: line 2"):
        read_answers(path)


def test_a_complete_last_line_that_is_not_an_answer_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / ANSWERS_FILE
    path.write_text('{"scene": "A"}')
    with pytest.raises(ValueError, match=r"line 1"):
        read_answers(path)


# --- the command's exit code and summary --------------------------------------------------------


def test_an_unexpected_error_in_the_command_exits_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def broken(*args: object, **kwargs: object) -> list[LadderAnswer]:
        raise RuntimeError("something\nunexpected")

    monkeypatch.setenv(MODEL_ENV_VAR, "test")
    monkeypatch.setattr("surprise_loop.cli.run_ladder", broken)
    assert run_command(tmp_path) == 2
    err = capsys.readouterr().err.strip().splitlines()
    assert err[-1] == "error: RuntimeError: something; unexpected"


def test_the_summary_records_the_samples_and_jobs_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sweep(always("shade"), tmp_path)
    monkeypatch.setenv(MODEL_ENV_VAR, "test")
    assert run_command(tmp_path, "--jobs", "2") == 1
    summary = json.loads((tmp_path / SUMMARY_FILE).read_text())
    assert summary["samples"] == SAMPLES
    assert summary["params"]["ladder"]["samples"] == SAMPLES
    assert summary["params"]["ladder"]["jobs"] == 2
    capsys.readouterr()


# --- the same sweep through the Deliberate agent ------------------------------------------------

NEUTRAL = STUB_READINGS[(None, True)]


def test_the_deliberate_ladder_actions_are_the_meadows_offers() -> None:
    assert set(LADDER_ACTION_OF) == set(LADDER_ACTIONS)
    assert sorted(LADDER_ACTION_OF.values()) == sorted(offers_at(Place.MEADOW))
    assert LADDER_ACTION_OF["shade"] is Action.GO_OAK


@pytest.mark.parametrize("scene", SCENES)
@pytest.mark.parametrize("level", [0, 3, 6])
def test_the_deliberate_ladder_prompt_is_the_deliberate_prompt(level: int, scene: Scene) -> None:
    order = answer_order(5)
    belief, t, arousal = ladder_body(level, scene, LP)
    text = felt_scene(
        Place.MEADOW, frozen_noon_facts(), belief, t, arousal, None, None, 0, Params().lens
    ).text()
    offers = [
        Offer(action=a, phrase=OFFER_PHRASES[(Place.MEADOW, a)])
        for a in (LADDER_ACTION_OF[k] for k in order)
    ]
    prompt = deliberate_ladder_prompt(level, scene, order, Params().lens, LP)
    assert prompt == deliberate_prompt(text, NEUTRAL, offers, [], "", "")
    assert NEUTRAL in prompt
    assert not re.search(r"[0-9_]", prompt)


def choose(action: Action, calls: list[str] | None = None) -> FunctionModel:
    """A Deliberate model that always answers the letter of `action`; records the prompts."""

    def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.instructions == DELIBERATE_INSTRUCTION
        prompt = user_prompt(messages)
        if calls is not None:
            calls.append(prompt)
        letter = next(
            line[0]
            for line in prompt.split("\n")
            if line[3:] == OFFER_PHRASES[(Place.MEADOW, action)] and line[1:3] == ". "
        )
        args = {
            "letter": letter,
            "heat": "a bit cooler",
            "thirst": "about the same",
            "tiredness": "about the same",
            "expect": "no expectation",
            "sure": "guessing",
            "why": "It is the  obvious\nthing.",
        }
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])

    return FunctionModel(answer)


def deliberate_sweep(model: FunctionModel, out: Path, samples: int = SAMPLES) -> list:
    done = read_answers(out / DELIBERATE_ANSWERS_FILE)
    return asyncio.run(run_ladder(model, samples, 4, out, Params(), done, "test", deliberate=True))


def test_a_deliberate_sweep_writes_its_own_file(tmp_path: Path) -> None:
    calls: list[str] = []
    new = deliberate_sweep(choose(Action.GO_OAK, calls), tmp_path)
    assert len(new) == len(calls) == len(SCENES) * len(LEVELS) * SAMPLES
    assert not (tmp_path / ANSWERS_FILE).exists()
    stored = read_answers(tmp_path / DELIBERATE_ANSWERS_FILE)
    assert len(stored) == len(new)
    assert all(a.action == "shade" for a in stored)
    assert all(a.order == answer_order(a.i) for a in stored)
    assert all(a.why == "It is the obvious thing." for a in stored)
    assert all(
        a.prompt
        == stored_prompt(
            deliberate_ladder_prompt(a.level, a.scene, a.order, Params().lens, LP), True
        )
        for a in stored
    )
    assert summarize(stored, LP)["A"].share == [1.0] * 7
    # a rerun asks nothing; the plain sweep does not count these answers
    assert deliberate_sweep(choose(Action.GO_OAK), tmp_path) == []
    assert current_answers(stored, "test", SAMPLES, Params()) == ([], len(stored))


def test_a_deliberate_answer_counts_as_shade_only_for_the_oak(tmp_path: Path) -> None:
    deliberate_sweep(choose(Action.REST), tmp_path, samples=1)
    stored = read_answers(tmp_path / DELIBERATE_ANSWERS_FILE)
    assert {a.action for a in stored} == {"sit"}
    assert summarize(stored, LP)["B"].share == [0.0] * 7


def test_the_command_with_deliberate_keeps_both_checks_apart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sweep(always("stream"), tmp_path)
    plain = (tmp_path / ANSWERS_FILE).read_text()
    deliberate_sweep(choose(Action.GO_OAK), tmp_path)
    monkeypatch.setenv(MODEL_ENV_VAR, "test")

    assert run_command(tmp_path, "--deliberate") == 1  # every level all shade: a flat curve
    summary = json.loads((tmp_path / DELIBERATE_SUMMARY_FILE).read_text())
    assert summary["scenes"]["A"]["share"] == [1.0] * 7
    assert not (tmp_path / SUMMARY_FILE).exists()
    assert (tmp_path / ANSWERS_FILE).read_text() == plain

    assert run_command(tmp_path) == 1
    assert json.loads((tmp_path / SUMMARY_FILE).read_text())["scenes"]["A"]["share"] == [0.0] * 7
    assert json.loads((tmp_path / DELIBERATE_SUMMARY_FILE).read_text()) == summary
    capsys.readouterr()


# --- an answer is tied to the instructions too -------------------------------------------------


@pytest.mark.parametrize("deliberate", [False, True], ids=["plain", "deliberate"])
def test_a_changed_instruction_makes_the_stored_answers_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deliberate: bool
) -> None:
    if deliberate:
        deliberate_sweep(choose(Action.GO_OAK), tmp_path)
    else:
        sweep(always("shade"), tmp_path)
    stored = read_answers(tmp_path / (DELIBERATE_ANSWERS_FILE if deliberate else ANSWERS_FILE))
    every = len(SCENES) * len(LEVELS) * SAMPLES
    assert current_answers(stored, "test", SAMPLES, Params(), deliberate) == (stored, 0)
    assert all(
        a.prompt.startswith(DELIBERATE_INSTRUCTION if deliberate else LADDER_INSTRUCTION)
        for a in stored
    )

    name = "DELIBERATE_INSTRUCTION" if deliberate else "LADDER_INSTRUCTION"
    monkeypatch.setattr(f"surprise_loop.ladder.{name}", "Choose what you do next.")
    assert current_answers(stored, "test", SAMPLES, Params(), deliberate) == ([], every)
