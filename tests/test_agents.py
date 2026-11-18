"""Appraise and Deliberate on a model: prompts, output mapping, validation and retry, whole runs.

Only TestModel and FunctionModel are used; conftest.py forbids requests to any real model.
"Prompt" here is what a model reads: the agent's instructions and the user prompt, not the
output tool's schema.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from llm_helpers import (
    AFTERNOON_LLM,
    APPRAISAL,
    DECISION,
    FORBIDDEN,
    MACHINE_LIKE,
    Recorder,
    afternoon,
    failing_model,
    offered_letters,
    prefer,
    retry_feedback,
    runtime,
)
from pydantic_ai import ModelResponse, ToolCallPart, UnexpectedModelBehavior, models
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from surprise_loop.agents import (
    LETTERS,
    appraise_agent,
    appraise_prompt,
    deliberate_agent,
)
from surprise_loop.config import (
    MODEL_ENV_VAR,
    Backends,
    ConfigError,
    LLMParams,
    initial_agent_state,
    load_config,
)
from surprise_loop.loop import load_dumps, replay, run
from surprise_loop.phases import appraise, deliberate
from surprise_loop.phases.appraise import appraise_llm
from surprise_loop.phases.deliberate import deliberate_llm, offer_order
from surprise_loop.phases.heuristics import bands_from_words
from surprise_loop.phases.perceive import build_percept
from surprise_loop.phrases import (
    ACTION_QUESTION,
    APPRAISE_INSTRUCTION,
    APPRAISE_QUESTION,
    COLLAPSED_LINE,
    DELIBERATE_INSTRUCTION,
    DID_LINES,
    EXPECTED_LINE,
    FEEL_QUESTION,
    OFFER_PHRASES,
    OFFERS_HEADING,
    OPEN_QUESTION_LEAD,
    OVERRIDE_LINES,
    PULL_PHRASES,
    QUESTION_PHRASES,
    READING_HEADING,
    RECALLED_HEADING,
    REGULATION_LINES,
    SURE_QUESTION,
    SURPRISE_PHRASES,
    WHY_QUESTION,
    WORLD_SURPRISE_PHRASES,
)
from surprise_loop.state import (
    Appraisal,
    Band,
    ChangeWord,
    Comparison,
    Decision,
    Episode,
    Expectation,
    Memory,
    Override,
    Sureness,
    TickDump,
    Workspace,
)
from surprise_loop.world import (
    QUESTION_OF,
    Action,
    Channel,
    Event,
    OpenQuestion,
    Place,
    offers_at,
)

HEAT, WATER, ENERGY = Channel.HEAT, Channel.WATER, Channel.ENERGY
READING = "Everything in me wants the shade."


def test_the_suite_forbids_real_model_requests() -> None:
    assert models.ALLOW_MODEL_REQUESTS is False


# --- hand-built workspaces ----------------------------------------------------------------------


def comparison(heat: float = 0.0, world: bool | None = None, size: float = 0.0) -> Comparison:
    return Comparison(
        body=dict.fromkeys(Channel, 0.0),
        body_felt={HEAT: heat, WATER: 0.0, ENERGY: 0.0},
        world=world,
        better=0.0,
        size=size,
    )


def expectation(words: dict[Channel, ChangeWord] | None, world: bool | None = None) -> Expectation:
    return Expectation(
        words=words,
        bands=dict.fromkeys(Channel, Band(lo=None, hi=None)),
        world=world,
        sureness=Sureness.GUESSING,
    )


WORDS = {HEAT: ChangeWord.BIT_MORE, WATER: ChangeWord.SAME, ENERGY: ChangeWord.BIT_LESS}


def episode(
    tick: int,
    place: Place,
    action: Action,
    event: Event,
    exp: Expectation,
    comp: Comparison,
    regulation: str = "",
) -> Episode:
    return Episode(
        tick=tick,
        step=tick,
        place=place,
        action=action,
        event=event,
        expectation=exp,
        comparison=comp,
        need_before=dict.fromkeys(Channel, 0.0),
        salience=0.5,
        scene="an old scene",
        reading="an old reading",
        regulation=regulation,
        score=None if not regulation else 0.5,
    )


# a body surprise with words; a stub's episode with a regulation line; a world surprise
LOOKED = episode(
    1, Place.MEADOW, Action.LOOK, Event.LOOKED, expectation(WORDS), comparison(heat=0.9, size=0.9)
)
RESTED = episode(
    2,
    Place.MEADOW,
    Action.REST,
    Event.RESTED,
    expectation(None),
    comparison(),
    REGULATION_LINES[(Action.REST, 1)],
)
TO_OAK = episode(
    3,
    Place.MEADOW,
    Action.GO_OAK,
    Event.ARRIVED,
    expectation(WORDS, world=True),
    comparison(heat=0.1, world=True, size=1.1),
)
RECALLED = [LOOKED, RESTED, TO_OAK]


def workspace(
    config_agent: int = 0, *, recalled: list[Episode] | None = None, pull: float = 0.6
) -> tuple[Workspace, Any]:
    """Ilo (or tavi) at the afternoon's first tick, with `recalled` in memory and a pull toward
    the oak; perceived and appraised, ready for Deliberate."""
    config = afternoon(config_agent)
    rt = runtime(config, TestModel())
    agent = initial_agent_state(config)
    values = {p: dict(v) for p, v in agent.pulls.values.items()}
    values[Place.MEADOW][Action.GO_OAK] = pull
    agent = agent.model_copy(
        update={
            "memory": Memory(episodes=list(RECALLED if recalled is None else recalled)),
            "pulls": agent.pulls.model_copy(update={"values": values}),
        }
    )
    observation = rt.world.observe()
    ws = Workspace(
        agent=agent,
        observation=observation,
        percept=build_percept(observation, agent, config.params),
        appraisal=Appraisal(reading=READING, about=HEAT),
    )
    return ws, rt


def pass_one(ws: Workspace, actions: list[Action]) -> Workspace:
    """`ws` as the second pass of its tick, overruled by ilo's heat and offered `actions`."""
    override = Override(channel=HEAT, actions=actions, detail="heat need 2.10 is at least 1.50")
    percept = build_percept(ws.observation, ws.agent, afternoon().params, offers=actions)
    return ws.model_copy(update={"pass_index": 1, "override": override, "percept": percept})


def answered(rec: Recorder, ws: Workspace, rt: Any) -> Workspace:
    with deliberate_agent.override(model=rec.model):
        return asyncio.run(deliberate_llm(ws, rt))


# --- Appraise -----------------------------------------------------------------------------------


def test_the_appraise_prompt_is_the_scene_then_one_question() -> None:
    ws, rt = workspace()
    assert ws.percept is not None
    text = ws.percept.scene.text()
    assert appraise_prompt(text) == f"{text}\n\n{APPRAISE_QUESTION}"


@pytest.mark.parametrize(
    ("about", "channel"),
    [("heat", HEAT), ("thirst", WATER), ("tiredness", ENERGY), ("nothing", None)],
)
def test_an_appraise_answer_maps_to_the_appraisal(about: str, channel: Channel | None) -> None:
    ws, rt = workspace()
    rec = Recorder(appraisal={"reading": "  Too   hot,\nfar too hot. ", "about": about})
    with appraise_agent.override(model=rec.model):
        out = asyncio.run(appraise_llm(ws.model_copy(update={"appraisal": None}), rt))
    assert out.appraisal == Appraisal(reading="Too hot, far too hot.", about=channel)
    [(name, instructions, prompt)] = rec.calls
    assert name == "appraise" and instructions == APPRAISE_INSTRUCTION
    assert ws.percept is not None
    assert prompt == appraise_prompt(ws.percept.scene.text())


def test_the_appraise_dispatcher_selects_the_model() -> None:
    ws, rt = workspace()
    rec = Recorder()
    with appraise_agent.override(model=rec.model):
        out = asyncio.run(appraise(ws, rt))
    assert out.appraisal is not None and out.appraisal.reading == APPRAISAL["reading"]


@pytest.mark.parametrize(
    "bad",
    [{"reading": "Hot.", "about": "the sun"}, {"reading": "   ", "about": "heat"}],
    ids=["about-off-the-list", "empty-reading"],
)
def test_a_bad_appraisal_is_retried(bad: dict[str, Any]) -> None:
    ws, rt = workspace()
    answers = [bad, APPRAISAL]
    calls: list[list[ModelMessage]] = []

    def function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(messages)
        args = answers[min(len(calls), len(answers)) - 1]
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])

    with appraise_agent.override(model=FunctionModel(function)):
        out = asyncio.run(appraise_llm(ws, rt))
    assert len(calls) == 2
    assert out.appraisal is not None and out.appraisal.about is HEAT


# --- the Deliberate prompt ----------------------------------------------------------------------


def test_the_deliberate_prompt_shows_every_element_in_order() -> None:
    ws, rt = workspace()
    rec = Recorder()
    answered(rec, ws, rt)
    [(name, instructions, prompt)] = rec.calls
    assert name == "deliberate" and instructions == DELIBERATE_INSTRUCTION
    assert ws.percept is not None

    order = offer_order(rt.config.seed, 1, 0, offers_at(Place.MEADOW))
    offer_lines = [
        f"{letter}. {OFFER_PHRASES[(Place.MEADOW, a)]}"
        for letter, a in zip(LETTERS, order, strict=False)
    ]
    looked = " ".join(
        [
            DID_LINES[(Place.MEADOW, Action.LOOK)],
            EXPECTED_LINE.format(
                heat="a bit warmer", thirst="about the same", tiredness="a bit less tired"
            ),
            SURPRISE_PHRASES[(HEAT, True)],
        ]
    )
    rested = " ".join([DID_LINES[(Place.MEADOW, Action.REST)], REGULATION_LINES[(Action.REST, 1)]])
    to_oak = " ".join(
        [
            DID_LINES[(Place.MEADOW, Action.GO_OAK)],
            EXPECTED_LINE.format(
                heat="a bit warmer", thirst="about the same", tiredness="a bit less tired"
            ),
            WORLD_SURPRISE_PHRASES[(OpenQuestion.LOG_SHADED, False)],
        ]
    )
    questions = [
        f"{OPEN_QUESTION_LEAD.format(letter=letter)} {QUESTION_PHRASES[q]}"
        for letter, a in zip(LETTERS, order, strict=False)
        if (q := QUESTION_OF[a]) is not None
    ]
    expected = [
        ws.percept.scene.text(),
        f"{READING_HEADING} {READING}",
        OFFERS_HEADING,
        *offer_lines,
        RECALLED_HEADING,
        f"- {looked}",
        f"- {rested}",
        f"- {to_oak}",
        PULL_PHRASES[(Action.GO_OAK, True)],
        ACTION_QUESTION,
        FEEL_QUESTION,
        *questions,
        SURE_QUESTION,
        WHY_QUESTION,
    ]
    positions = [prompt.index(part) for part in expected]
    assert positions == sorted(positions)
    assert len(questions) == 2  # the oak's log and the stream
    assert all(line in prompt.split("\n") for line in offer_lines)
    assert not any(line in prompt for line in OVERRIDE_LINES.values())


def test_an_episode_without_words_is_told_without_the_expectation() -> None:
    ws, rt = workspace(recalled=[RESTED])
    rec = Recorder()
    answered(rec, ws, rt)
    prompt = rec.prompts("deliberate")[0]
    assert "You expected" not in prompt
    assert DID_LINES[(Place.MEADOW, Action.REST)] in prompt


def test_a_collapse_is_told_as_a_collapse() -> None:
    collapsed = RESTED.model_copy(update={"action": Action.GO_OAK, "event": Event.COLLAPSED})
    ws, rt = workspace(recalled=[collapsed])
    rec = Recorder()
    answered(rec, ws, rt)
    prompt = rec.prompts("deliberate")[0]
    assert COLLAPSED_LINE in prompt
    assert DID_LINES[(Place.MEADOW, Action.GO_OAK)] not in prompt


def test_no_memory_and_no_pull_leave_their_parts_out() -> None:
    ws, rt = workspace(recalled=[], pull=0.0)
    rec = Recorder()
    answered(rec, ws, rt)
    prompt = rec.prompts("deliberate")[0]
    assert RECALLED_HEADING not in prompt
    assert not any(line in prompt for line in PULL_PHRASES.values())


def test_pass_one_shows_the_override_line_and_only_the_serving_offers() -> None:
    ws, rt = workspace()
    rec = Recorder()
    out = answered(rec, pass_one(ws, [Action.GO_OAK]), rt)
    prompt = rec.prompts("deliberate")[0]
    assert OVERRIDE_LINES[(HEAT, True)] in prompt  # ilo's heat need is one of too much
    assert "heat need" not in prompt  # the override's detail is for the logs
    assert prompt.index(OVERRIDE_LINES[(HEAT, True)]) < prompt.index(ACTION_QUESTION)
    assert offered_letters(prompt) == {"A": Action.GO_OAK}
    assert out.decision is not None and out.decision.action is Action.GO_OAK


# --- mapping ------------------------------------------------------------------------------------


def letter_for(order: list[Action], action: Action) -> str:
    return LETTERS[order.index(action)]


@pytest.mark.parametrize(
    ("action", "expect", "world"),
    [
        (Action.GO_OAK, "yes", True),
        (Action.GO_OAK, "no", False),
        (Action.GO_OAK, "no expectation", None),
        (Action.GO_STREAM, "no", False),
        (Action.REST, "yes", None),  # sitting down settles no open question
        (Action.LOOK, "no", None),
    ],
)
def test_a_valid_answer_maps_to_the_decision(
    action: Action, expect: str, world: bool | None
) -> None:
    ws, rt = workspace()
    order = offer_order(rt.config.seed, 1, 0, offers_at(Place.MEADOW))
    answer = {**DECISION, "letter": letter_for(order, action), "expect": expect}
    out = answered(Recorder(lambda prompt: answer), ws, rt)

    words = {HEAT: ChangeWord.MUCH_LESS, WATER: ChangeWord.SAME, ENERGY: ChangeWord.BIT_MORE}
    p = rt.config.params.expectation
    t = ws.agent.temperament
    assert out.decision == Decision(
        action=action,
        expectation=Expectation(
            words=words,
            bands=bands_from_words(words, ws.agent.belief, t, p),
            world=world,
            sureness=Sureness.FAIRLY_SURE,
        ),
        why=DECISION["why"],
    )
    assert out == ws.model_copy(update={"decision": out.decision})


@pytest.mark.parametrize(
    ("sure", "sureness"),
    [
        ("guessing", Sureness.GUESSING),
        ("fairly sure", Sureness.FAIRLY_SURE),
        ("sure", Sureness.SURE),
    ],
)
def test_sureness_maps(sure: str, sureness: Sureness) -> None:
    ws, rt = workspace()
    out = answered(Recorder(lambda prompt: {**DECISION, "sure": sure}), ws, rt)
    assert out.decision is not None and out.decision.expectation.sureness is sureness


def test_the_request_carries_the_llm_params() -> None:
    ws, rt = workspace()
    config = rt.config.model_copy(
        update={
            "params": rt.config.params.model_copy(
                update={"llm": LLMParams(temperature=0.4, timeout_s=12.5)}
            )
        }
    )
    seen: list[AgentInfo] = []

    def function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, DECISION)])

    asyncio.run(deliberate_llm(ws, runtime(config, FunctionModel(function))))
    [info] = seen
    assert info.model_settings is not None
    assert info.model_settings.get("temperature") == 0.4
    assert info.model_settings.get("timeout") == 12.5


def test_the_deliberate_dispatcher_selects_the_model() -> None:
    ws, rt = workspace()
    rec = Recorder()
    with deliberate_agent.override(model=rec.model):
        out = asyncio.run(deliberate(ws, rt))
    assert len(rec.calls) == 1
    assert out.decision is not None and out.decision.why == DECISION["why"]


@pytest.mark.parametrize("phase", [appraise_llm, deliberate_llm], ids=["appraise", "deliberate"])
def test_a_model_phase_without_a_resolved_model_raises_config_error(phase: Any) -> None:
    ws, rt = workspace()
    with pytest.raises(ConfigError, match="no resolved model"):
        asyncio.run(phase(ws, runtime(rt.config, None)))


# --- validation and retry -----------------------------------------------------------------------

INVALID = [
    pytest.param({**DECISION, "letter": "E"}, "E", id="letter-not-on-offer"),
    pytest.param({**DECISION, "letter": "F"}, "letter", id="letter-off-the-alphabet"),
    pytest.param({**DECISION, "heat": "boiling"}, "heat", id="heat-off-the-ladder"),
    pytest.param({**DECISION, "thirst": "a bit cooler"}, "thirst", id="another-channels-phrase"),
    pytest.param({**DECISION, "tiredness": "tired"}, "tiredness", id="tiredness-off-the-ladder"),
    pytest.param({**DECISION, "expect": "maybe"}, "expect", id="expect-off-the-list"),
    pytest.param({**DECISION, "sure": "certain"}, "sure", id="sure-off-the-list"),
    pytest.param({**DECISION, "why": " "}, "why", id="empty-why"),
    pytest.param({**DECISION, "mood": "fine"}, "mood", id="extra-field"),
]


def scripted(*answers: dict[str, Any]) -> tuple[FunctionModel, list[list[ModelMessage]]]:
    calls: list[list[ModelMessage]] = []

    def function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(list(messages))
        args = answers[min(len(calls), len(answers)) - 1]
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])

    return FunctionModel(function), calls


@pytest.mark.parametrize(("bad", "field"), INVALID)
def test_an_invalid_answer_is_retried_then_accepted(bad: dict[str, Any], field: str) -> None:
    ws, rt = workspace()  # four offers: A to D
    model, calls = scripted(bad, DECISION)
    with deliberate_agent.override(model=model):
        out = asyncio.run(deliberate_llm(ws, rt))
    assert len(calls) == 2
    assert field in retry_feedback(calls[1])
    assert out.decision is not None and out.decision.why == DECISION["why"]


def test_a_letter_beyond_a_single_offer_is_retried() -> None:
    ws, rt = workspace()
    model, calls = scripted({**DECISION, "letter": "B"}, DECISION)
    with deliberate_agent.override(model=model):
        out = asyncio.run(deliberate_llm(pass_one(ws, [Action.GO_OAK]), rt))
    assert len(calls) == 2
    assert out.decision is not None and out.decision.action is Action.GO_OAK


@pytest.mark.parametrize("retries", [0, 1, 3])
@pytest.mark.parametrize(("bad", "field"), INVALID[:3])
def test_a_persistently_invalid_answer_raises_after_the_retry_budget(
    bad: dict[str, Any], field: str, retries: int
) -> None:
    ws, rt = workspace()
    config = rt.config.model_copy(
        update={
            "params": rt.config.params.model_copy(update={"llm": LLMParams(output_retries=retries)})
        }
    )
    model, calls = scripted(bad)
    with pytest.raises(UnexpectedModelBehavior, match="output retries") as info:
        asyncio.run(deliberate_llm(ws, runtime(config, model)))
    assert len(calls) == retries + 1
    assert any("deliberate failed at tick 1, pass 0" in note for note in info.value.__notes__)


def test_an_invalid_answer_stops_the_run_without_a_stub_fallback(tmp_path: Path) -> None:
    good = Recorder(prefer(Action.LOOK))
    count = 0

    def decide(prompt: str) -> dict[str, Any]:
        nonlocal count
        count += 1
        return good.decide(prompt) if count <= 3 else {**DECISION, "heat": "scorching"}

    rec = Recorder(decide)
    config = afternoon(1, llm=LLMParams(output_retries=1))  # tavi: no override, one pass a tick
    run_dir = tmp_path / "run"
    with (
        appraise_agent.override(model=rec.model),
        deliberate_agent.override(model=rec.model),
        pytest.raises(UnexpectedModelBehavior) as info,
    ):
        asyncio.run(run(config, run_dir))
    assert any("tick 4, pass 0" in note for note in info.value.__notes__)
    # the ticks before the failure are on disk, the failing one is not: a resume starts there
    assert [d.tick for d in load_dumps(run_dir)] == [1, 2, 3]


# --- whole runs ---------------------------------------------------------------------------------


def test_the_model_experiment_is_the_afternoon_on_a_model() -> None:
    llm = load_config(AFTERNOON_LLM)
    stub = load_config(AFTERNOON_LLM.with_name("clearing_afternoon.yaml"))
    assert llm.backends == Backends(appraise="llm", deliberate="llm")
    assert llm.model is None
    assert llm.model_copy(update={"name": stub.name, "backends": stub.backends}) == stub


def collected_run(decide: Any, ticks: int = 40, agent: int = 0) -> tuple[Recorder, list[Any]]:
    rec = Recorder(decide)
    with appraise_agent.override(model=rec.model), deliberate_agent.override(model=rec.model):
        result = asyncio.run(run(afternoon(agent, ticks=ticks)))
    return rec, result.records


def test_no_prompt_of_a_forty_tick_run_holds_a_digit_or_an_identifier() -> None:
    # ilo sits down in the hot meadow and expects to cool: the heat overrules her on pass 1
    rec, records = collected_run(prefer(Action.REST))
    assert len(records) == 40
    overrides = [p.override for r in records for p in r.passes if p.override is not None]
    assert overrides, "the run should hold at least one override"
    deliberate_prompts = rec.prompts("deliberate")
    assert len(deliberate_prompts) == sum(len(r.passes) for r in records)
    pass_one = [p for p in deliberate_prompts if any(line in p for line in OVERRIDE_LINES.values())]
    assert len(pass_one) == len(overrides)
    assert any(RECALLED_HEADING in p for p in deliberate_prompts)

    for _, instructions, prompt in rec.calls:
        for text in (instructions, prompt):
            assert not MACHINE_LIKE.search(text), text
            assert not FORBIDDEN.search(text), text
    for override in overrides:
        assert all(override.detail not in p for p in deliberate_prompts)


def test_the_offer_order_differs_between_ticks_and_repeats_on_a_second_run() -> None:
    rec, records = collected_run(prefer(Action.LOOK), ticks=12, agent=1)
    orders = [tuple(offered_letters(p).values()) for p in rec.prompts("deliberate")]
    assert all(set(o) == set(offers_at(Place.MEADOW)) for o in orders)  # tavi stays in the meadow
    assert len(set(orders)) > 1
    for tick, order in enumerate(orders, start=1):
        assert list(order) == offer_order(1, tick, 0, offers_at(Place.MEADOW))
    again, _ = collected_run(prefer(Action.LOOK), ticks=12, agent=1)
    assert again.calls == rec.calls


def test_offer_order_is_a_seeded_shuffle() -> None:
    offers = offers_at(Place.MEADOW)
    assert sorted(offer_order(3, 5, 1, offers)) == sorted(offers)
    assert offer_order(3, 5, 1, offers) == offer_order(3, 5, 1, list(offers))
    assert len({tuple(offer_order(3, t, p, offers)) for t in range(1, 9) for p in (0, 1)}) > 1


def test_a_twenty_tick_run_on_the_test_model_writes_valid_dumps_and_replays(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    result = asyncio.run(run(afternoon(0, model="test", ticks=20), run_dir))
    assert len(result.records) == 20
    dumps = load_dumps(run_dir)
    assert [d.tick for d in dumps] == list(range(1, 21))
    for path in sorted(run_dir.glob("tick_*.json")):
        dump = TickDump.model_validate_json(path.read_bytes())
        assert dump.record.appraisal.reading  # from the model, not a stub line
        for record in dump.record.passes:
            assert record.decision.expectation.words is not None  # from the model
    assert json.loads((run_dir / "config.json").read_text())["model"] == "test"

    # replay uses the recorded outputs: a model that fails on any call proves none is made
    with (
        appraise_agent.override(model=failing_model()),
        deliberate_agent.override(model=failing_model()),
    ):
        assert asyncio.run(replay(run_dir)) == 20


def test_a_model_run_with_no_model_fails_before_tick_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
    rec = Recorder()
    run_dir = tmp_path / "run"
    with (
        appraise_agent.override(model=rec.model),
        deliberate_agent.override(model=rec.model),
        pytest.raises(ConfigError, match=MODEL_ENV_VAR),
    ):
        asyncio.run(run(afternoon(model=None), run_dir))
    assert rec.calls == []
    assert not run_dir.exists()


def test_the_model_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MODEL_ENV_VAR, "test")
    result = asyncio.run(run(afternoon(model=None, ticks=2)))
    assert result.config.model == "test"
    assert len(result.records) == 2


def test_letters_cover_every_offer_a_place_can_make() -> None:
    assert len(LETTERS) >= max(len(offers_at(p)) for p in Place)
