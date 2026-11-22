"""Deliberate and Appraise on a model: the view, the prompts, the outputs, retries, and whole runs
on a FunctionModel. No real model is called."""

import asyncio
import dataclasses
import json
import re
from pathlib import Path
from typing import Any

import pytest
from creatures import TWO_MAPS, load_two_maps, run_config
from llm_helpers import (
    APPRAISAL,
    BOTH_LLM,
    DECISION,
    FORBIDDEN,
    Recorder,
    beliefs,
    failing_model,
    llm_config,
    retry_feedback,
    runtime,
    scripted,
)
from pydantic import ValidationError
from pydantic_ai import UnexpectedModelBehavior

from surprise_loop import phrases
from surprise_loop.agents import (
    APPRAISE_INSTRUCTIONS,
    DELIBERATE_INSTRUCTIONS,
    AppraiseOutput,
    ChangeOutput,
    DeliberateOutput,
    appraise_agent,
    appraise_prompt,
    creature_view,
    deliberate_agent,
    deliberate_prompt,
    model_settings,
    to_appraisal,
    to_decision,
)
from surprise_loop.cli import main
from surprise_loop.config import (
    MODEL_ENV_VAR,
    AppraiseParams,
    Backends,
    ConfigError,
    LLMParams,
    MemoryParams,
    Params,
    RunConfig,
    Runtime,
    TokenCount,
    initial_agent_state,
    load_config,
)
from surprise_loop.loop import load_dumps, replay, run
from surprise_loop.phases import act, appraise, deliberate, evaluate, monitor, perceive
from surprise_loop.state import (
    AgentState,
    Appraisal,
    Change,
    Comparison,
    Decision,
    Expectation,
    FeltRecord,
    Sureness,
    TickDump,
    Workspace,
)
from surprise_loop.world import Belief, Channel, Direction, GridWorld, Kind, moved_to

EXPERIMENTS = TWO_MAPS.parent
YOU_FEEL = ["moved", *Channel, "smell", "damp", "tone", "stir"]
EXPECTED = ["moved", *Channel, "smell", "damp", "sure"]
STEP_KEYS = ["went", "toward", "expected", "felt", "surprise", "it_went", "reading"]


def ran(ticks: int, **changes: Any) -> AgentState:
    """The plain creature after `ticks` ticks on the stand-ins."""
    return asyncio.run(run(run_config(ticks=ticks, **changes))).agent


def strings(value: Any) -> list[str]:
    """Every key and string value inside a parsed JSON value."""
    if isinstance(value, dict):
        return [*value, *(s for v in value.values() for s in strings(v))]
    if isinstance(value, list):
        return [s for v in value for s in strings(v)]
    return [value] if isinstance(value, str) else []


def initial() -> AgentState:
    config = run_config()
    return initial_agent_state(config, GridWorld(config.world, config.seed).observe())


def until_appraise(config: RunConfig, model: object = None) -> tuple[Workspace, Runtime]:
    """A workspace of tick 1 through Evaluate, its Deliberate on the rule, and the runtime of
    `config` on the same world."""
    rule = config.model_copy(update={"backends": Backends(appraise=config.backends.appraise)})
    rt = runtime(rule, model)
    ws = Workspace(agent=initial_agent_state(rule, rt.world.observe()))
    for phase in (perceive, deliberate, act, monitor, evaluate):
        ws = asyncio.run(phase(ws, rt))
    return ws, dataclasses.replace(rt, config=config)


# --- the view ------------------------------------------------------------------------------------


def test_the_view_has_exactly_the_keys_of_the_contract_and_feelings_are_words() -> None:
    agent = ran(10)
    view = creature_view(agent, agent.memory.episodes, Params())

    assert list(view) == ["you_feel", "you_are_at", "your_map", "your_last_steps"]
    assert list(view["you_feel"]) == YOU_FEEL
    assert view["you_feel"]["moved"] is agent.felt.moved
    for c in Channel:
        assert view["you_feel"][c] == phrases.FEEL_WORDS[agent.felt.levels[c]]
    assert view["you_feel"]["smell"] == phrases.NEAR_WORDS[agent.felt.smell]
    assert view["you_feel"]["damp"] == phrases.NEAR_WORDS[agent.felt.damp]
    assert view["you_feel"]["tone"] == phrases.TONE_WORDS[agent.felt.tone]
    assert view["you_feel"]["stir"] == phrases.STIR_WORDS[agent.felt.stir]
    assert view["you_are_at"] == list(agent.felt.at)
    assert list(view["your_map"]) == ["untried", "spots"]
    assert list(view["your_map"]["untried"]) == list(Belief)
    for step in view["your_last_steps"]:
        assert list(step) == STEP_KEYS
        assert list(step["expected"]) == EXPECTED
        assert list(step["felt"]) == YOU_FEEL
        assert step["surprise"] in phrases.SURPRISE_WORDS
        assert step["it_went"] in (phrases.WENT_WORSE, phrases.WENT_BETTER, phrases.WENT_SAME)
    json.dumps(view)  # plain JSON


def test_spots_hold_tried_and_changed_tiles_north_first_then_east_with_rounded_beliefs() -> None:
    agent = initial()
    changed = agent.own_map.changed(
        [
            Change(at=(-1, 0), beliefs=beliefs(0.123456, 0.0, 0.5, 0.0, 0.987654)),
            Change(at=(2, 3), beliefs=beliefs(1.0, 0.0, 0.0, 0.0, 0.0)),
        ]
    )
    agent = agent.model_copy(
        update={"own_map": changed, "tries": {"1,0": 2, "0,1": 1, "5,5": 0, "-1,0": 1}}
    )

    spots = creature_view(agent, [], Params())["your_map"]["spots"]

    # the starting tile has an entry from the start: it is known to be walkable
    assert [s["at"] for s in spots] == [[2, 3], [0, 1], [-1, 0], [0, 0], [1, 0]]
    assert [s["tries"] for s in spots] == [0, 1, 1, 0, 2]
    assert spots[3]["blocks"] == 0.0
    assert spots[2] == {
        "at": [-1, 0],
        "tries": 1,
        "blocks": 0.12,
        "hurts": 0.0,
        "feeds": 0.5,
        "quenches": 0.0,
        "pleases": 0.99,
    }
    untried = agent.own_map.untried
    # a tried spot with no entry of its own holds what every untried spot is believed to be
    assert spots[1] == {"at": [0, 1], "tries": 1, **{b.value: round(untried[b], 2) for b in Belief}}
    view = creature_view(
        agent, [], Params(prompt=Params().prompt.model_copy(update={"decimals": 1}))
    )
    assert view["your_map"]["spots"][2]["blocks"] == 0.1
    assert view["your_map"]["untried"] == {b.value: round(untried[b], 1) for b in Belief}


def test_last_steps_are_the_most_recent_k_with_words() -> None:
    agent = ran(10)
    episodes = agent.memory.episodes
    assert len(episodes) == 6

    for k in (0, 3, 6, 9):
        p = Params(memory=MemoryParams(recent_k=k))
        steps = creature_view(agent, episodes, p)["your_last_steps"]
        assert len(steps) == min(k, len(episodes))
        assert [s["went"] for s in steps] == [
            e.direction.value for e in episodes[len(episodes) - len(steps) :]
        ]

    last = episodes[-1]
    step = creature_view(agent, episodes, Params())["your_last_steps"][-1]
    assert step["expected"] == {
        "moved": last.expectation.moved,
        **{c.value: phrases.FEEL_WORDS[last.expectation.levels[c]] for c in Channel},
        "smell": phrases.NEAR_WORDS[last.expectation.smell],
        "damp": phrases.NEAR_WORDS[last.expectation.damp],
        "sure": last.expectation.sure.value,
    }
    assert step["felt"]["moved"] is last.felt.moved
    assert step["felt"]["hunger"] == phrases.FEEL_WORDS[last.felt.levels[Channel.HUNGER]]
    assert step["reading"] == last.reading


@pytest.mark.parametrize(
    ("surprise", "worse", "word", "went"),
    [
        (0.0, 0.0, "none", phrases.WENT_SAME),
        (0.049, 0.099, "none", phrases.WENT_SAME),
        (0.05, 0.1, "small", phrases.WENT_WORSE),
        (0.39, -0.1, "small", phrases.WENT_BETTER),
        (0.4, -0.099, "large", phrases.WENT_SAME),
        (1.0, 1.5, "large", phrases.WENT_WORSE),
    ],
)
def test_surprise_and_how_it_went_are_words_by_the_edges(
    surprise: float, worse: float, word: str, went: str
) -> None:
    agent = ran(2)
    last = agent.memory.episodes[-1]
    comparison = last.comparison.model_copy(update={"surprise": surprise, "worse": worse})
    episode = last.model_copy(update={"comparison": comparison})

    step = creature_view(agent, [episode], Params())["your_last_steps"][0]

    assert (step["surprise"], step["it_went"]) == (word, went)


# --- what never reaches a model ------------------------------------------------------------------


def distinctive() -> RunConfig:
    """A run whose temperament and body carry numbers that cannot arise by chance in a prompt."""
    base = llm_config(ticks=12)
    temperament = base.agent.temperament.model_copy(
        update={"gain": dict.fromkeys(Channel, 1.37), "noise": dict.fromkeys(Channel, 0.0246)}
    )
    world = base.world.model_copy(
        update={"start_body": base.world.start_body | {Channel.HUNGER: 0.4321}}
    )
    agent = base.agent.model_copy(update={"temperament": temperament})
    return base.model_copy(update={"agent": agent, "world": world})


DISTINCT = ("1.37", "137", "0246", "4321", "1234")


def assert_clean(text: str) -> None:
    assert not FORBIDDEN.search(text), text
    for digits in DISTINCT:
        assert digits not in text, (digits, text)


def test_no_text_a_model_reads_holds_a_kind_a_temperament_or_body_number_or_a_framing_word() -> (
    None
):
    rec = Recorder()
    with deliberate_agent.override(model=rec.model), appraise_agent.override(model=rec.model):
        result = asyncio.run(run(distinctive()))
    assert len(result.records) == 12
    assert len(rec.prompts("deliberate")) == len(rec.prompts("appraise")) == 12

    kinds = {k.value for k in Kind}
    for _, instructions, prompt in rec.calls:
        assert_clean(instructions)
        assert_clean(prompt)
        assert not kinds & set(strings(json.loads(prompt)))
    for tool in rec.tools:
        assert_clean(json.dumps([tool.name, tool.description, tool.parameters_json_schema]))


def test_no_fixed_text_names_a_game_a_simulation_or_an_experiment() -> None:
    public = [v for name, v in vars(phrases).items() if not name.startswith("_")]
    texts = [v for v in public if isinstance(v, str)]
    texts += [s for v in public if isinstance(v, tuple) for s in v]
    texts += [
        phrases.TOO_MANY_CHANGES.format(max_changes=12),
        phrases.TOO_FAR.format(at=[40, 0], reach=12),
    ]
    texts += [DELIBERATE_INSTRUCTIONS, APPRAISE_INSTRUCTIONS]
    for text in texts:
        assert not FORBIDDEN.search(text), text
    assert f"{phrases.CREATURE_TEXT}\n\n{phrases.DELIBERATE_TEXT}" == DELIBERATE_INSTRUCTIONS
    assert f"{phrases.CREATURE_TEXT}\n\n{phrases.APPRAISE_TEXT}" == APPRAISE_INSTRUCTIONS


def test_nothing_of_the_reasoner_reaches_a_prompt() -> None:
    agent = ran(6)
    shares = {k: 0.1234 if k is Kind.FOOD else 0.0 for k in Kind} | {Kind.GROUND: 1 - 0.1234}
    reasoner = agent.reasoner.model_copy(
        update={"presumes": shares, "tiles": {"1,0": shares, "0,1": shares}}
    )
    agent = agent.model_copy(update={"reasoner": reasoner})
    view = creature_view(agent, agent.memory.episodes, Params())
    last = agent.memory.episodes[-1]
    decision = Decision(direction=last.direction, expectation=last.expectation, why="")

    for prompt in (
        deliberate_prompt(view),
        appraise_prompt(view, decision, last.felt, last.comparison, Params()),
    ):
        assert "1234" not in prompt
        assert "8766" not in prompt


# --- the prompts ---------------------------------------------------------------------------------


def test_the_deliberate_prompt_is_the_view_as_json() -> None:
    agent = ran(4)
    view = creature_view(agent, agent.memory.episodes, Params())
    prompt = deliberate_prompt(view)
    assert prompt == json.dumps(view, ensure_ascii=False)
    assert json.loads(prompt) == view


def test_the_appraise_prompt_describes_the_step_and_the_body_after_it() -> None:
    agent = ran(4)
    view = creature_view(agent, agent.memory.episodes, Params())
    expectation = Expectation(
        moved=True,
        levels=dict.fromkeys(Channel, 0) | {Channel.HUNGER: 2},
        smell=1,
        damp=0,
        sure=Sureness.FAIRLY_SURE,
    )
    decision = Decision(direction=Direction.NORTH, expectation=expectation, why="")
    at = (agent.felt.at[0], agent.felt.at[1] + 1)
    felt_after = FeltRecord(
        moved=True,
        levels=dict.fromkeys(Channel, 0) | {Channel.THIRST: 3},
        smell=2,
        damp=1,
        tone=1,
        stir=2,
        at=at,
    )
    comparison = Comparison(
        moved_miss=False,
        misses=dict.fromkeys(Channel, 0),
        smell_miss=1,
        damp_miss=1,
        surprise=0.5,
        worse=0.25,
    )

    shown = json.loads(appraise_prompt(view, decision, felt_after, comparison, Params()))

    assert list(shown) == [
        "you_feel",
        "you_are_at",
        "your_map",
        "your_last_steps",
        "you_went",
        "you_stepped_toward",
        "you_felt_before",
        "you_expected",
        "what_changed",
        "surprise",
        "it_went",
    ]
    felt_words = {
        "moved": True,
        **{c.value: "none" for c in Channel},
        "thirst": "strong",
        "smell": "strong",
        "damp": "faint",
        "tone": "bad",
        "stir": "shaken",
    }
    assert shown["you_feel"] == felt_words
    assert shown["you_felt_before"] == view["you_feel"]
    assert "you_felt" not in shown
    assert shown["you_are_at"] == list(at) != view["you_are_at"]
    assert shown["your_map"] == view["your_map"]
    assert shown["your_last_steps"] == view["your_last_steps"]
    assert shown["you_went"] == "north"
    assert shown["you_expected"] == {
        "moved": True,
        **{c.value: "none" for c in Channel},
        "hunger": "clear",
        "smell": "faint",
        "damp": "none",
        "sure": "fairly sure",
    }
    assert (shown["surprise"], shown["it_went"]) == ("large", phrases.WENT_WORSE)


def felt(hunger: int = 0, damp: int = 0, tone: int = 2, stir: int = 0) -> FeltRecord:
    return FeltRecord(
        moved=True,
        levels=dict.fromkeys(Channel, 0) | {Channel.HUNGER: hunger},
        smell=0,
        damp=damp,
        tone=tone,
        stir=stir,
        at=(0, 0),
    )


def shown_after(before: FeltRecord, after: FeltRecord) -> dict[str, Any]:
    """The Appraise prompt, parsed, for a step from `before` to `after`."""
    agent = initial().model_copy(update={"felt": before})
    last = ran(2).memory.episodes[-1]
    decision = Decision(direction=Direction.EAST, expectation=last.expectation, why="")
    view = creature_view(agent, [], Params())
    return json.loads(appraise_prompt(view, decision, after, last.comparison, Params()))


def test_the_appraise_prompt_shows_the_record_before_the_step_in_words() -> None:
    before = felt(hunger=2, damp=0)
    shown = shown_after(before, felt(hunger=0, damp=1))
    assert shown["you_felt_before"] == {
        "moved": True,
        **{c.value: "none" for c in Channel},
        "hunger": "clear",
        "smell": "none",
        "damp": "none",
        "tone": "even",
        "stir": "calm",
    }
    assert (
        shown["you_felt_before"]
        == creature_view(initial().model_copy(update={"felt": before}), [], Params())["you_feel"]
    )
    assert "you_felt" not in shown


@pytest.mark.parametrize(
    ("before", "after", "changed"),
    [
        (
            felt(hunger=2, damp=0),
            felt(hunger=0, damp=1),
            ["hunger: clear to none", "damp: none to faint"],
        ),
        (felt(hunger=1, damp=2), felt(hunger=1, damp=2), []),
        (felt(tone=2, stir=0), felt(tone=0, stir=2), []),
    ],
    ids=["hunger-and-damp", "same-words", "tone-and-stir-only"],
)
def test_what_changed_lists_each_feeling_smell_or_damp_whose_word_changed(
    before: FeltRecord, after: FeltRecord, changed: list[str]
) -> None:
    assert shown_after(before, after)["what_changed"] == changed


def step_east_from_1_3(moved: bool) -> dict[str, Any]:
    """The Appraise prompt, parsed, for a step east from [1, 3] that moved or was blocked."""
    agent = initial()
    agent = agent.model_copy(update={"felt": agent.felt.model_copy(update={"at": (1, 3)})})
    after = agent.felt.model_copy(update={"moved": moved, "at": (2, 3) if moved else (1, 3)})
    last = ran(2).memory.episodes[-1]
    decision = Decision(direction=Direction.EAST, expectation=last.expectation, why="")
    view = creature_view(agent, [], Params())
    return json.loads(appraise_prompt(view, decision, after, last.comparison, Params()))


def test_a_blocked_step_was_toward_the_next_spot_and_left_the_creature_where_it_was() -> None:
    shown = step_east_from_1_3(moved=False)
    assert shown["you_stepped_toward"] == [2, 3]
    assert shown["you_are_at"] == [1, 3]


def test_a_step_that_moved_was_toward_the_spot_it_arrived_on() -> None:
    shown = step_east_from_1_3(moved=True)
    assert shown["you_stepped_toward"] == shown["you_are_at"] == [2, 3]


@pytest.mark.parametrize(
    ("moved", "at"), [(True, (2, 3)), (False, (1, 3))], ids=["moved", "blocked"]
)
def test_a_last_step_says_which_spot_it_was_toward(moved: bool, at: tuple[int, int]) -> None:
    agent = ran(2)
    last = agent.memory.episodes[-1]
    felt_after = last.felt.model_copy(update={"moved": moved, "at": at})
    episode = last.model_copy(update={"direction": Direction.EAST, "felt": felt_after})

    step = creature_view(agent, [episode], Params())["your_last_steps"][0]

    assert step["went"] == "east"
    assert step["toward"] == [2, 3]
    assert list(step)[:2] == ["went", "toward"]


def test_the_appraise_text_says_what_a_change_in_the_body_is_about() -> None:
    assert phrases.APPRAISE_TEXT == (
        "You have just taken a step. You are shown what you felt before it, what you expected, "
        "what you feel now, and what changed. What changes in your body right after a step is "
        "about the spot you stepped toward: the spot you arrived on if you moved, or the thing "
        "that stopped you if you did not. Sickness is different: it comes from something you ate "
        "or drank a few steps ago. Say in one short line what you make of it. Then list the "
        "spots whose beliefs you now change, each with all five beliefs. Change only what this "
        "step, or something you now understand about an earlier step, gives you reason to "
        "change. List nothing if nothing changed."
    )


def test_the_appraise_phase_shows_the_map_before_this_tick_and_the_place_after_the_step() -> None:
    config = llm_config(backends=Backends(deliberate="rule", appraise="llm"))
    rec = Recorder()
    ws, rt = until_appraise(config, rec.model)
    out = asyncio.run(appraise(ws, rt))

    [prompt] = rec.prompts("appraise")
    shown = json.loads(prompt)
    assert ws.felt_after is not None and ws.decision is not None
    assert shown["you_are_at"] == list(ws.felt_after.at)
    assert shown["you_went"] == ws.decision.direction.value
    assert shown["you_stepped_toward"] == list(moved_to(ws.agent.felt.at, ws.decision.direction))
    before = creature_view(ws.agent, [], config.params)
    assert shown["your_map"] == before["your_map"]
    assert shown["you_felt_before"] == before["you_feel"]
    assert out.appraisal == Appraisal(reading=APPRAISAL["reading"], changes=())


# --- the outputs ---------------------------------------------------------------------------------


def test_a_deliberate_output_maps_to_the_decision() -> None:
    output = DeliberateOutput.model_validate(
        {
            **DECISION,
            "why": "  The east\n looks   open. ",
            "direction": "west",
            "moved": False,
            "hunger": "overwhelming",
            "ache": "slight",
            "pleasure": "strong",
            "smell": "strong",
            "sure": "sure",
        }
    )

    assert to_decision(output) == Decision(
        direction=Direction.WEST,
        expectation=Expectation(
            moved=False,
            levels={
                Channel.HUNGER: 4,
                Channel.THIRST: 1,
                Channel.ACHE: 1,
                Channel.SICKNESS: 0,
                Channel.UNEASE: 0,
                Channel.PLEASURE: 3,
            },
            smell=2,
            damp=1,
            sure=Sureness.SURE,
        ),
        why="The east looks open.",
    )


def test_an_appraise_output_maps_to_the_appraisal() -> None:
    output = AppraiseOutput.model_validate(
        {
            "reading": "That\nwas food.  ",
            "changes": [
                {"at": [1, 0], **beliefs(1.0, 0.0, 0.9, 0.0, 0.0)},
                {"at": [-2, 3], **beliefs(0.0, 0.5, 0.0, 0.0, 0.25)},
            ],
        }
    )

    assert to_appraisal(output) == Appraisal(
        reading="That was food.",
        changes=(
            Change(at=(1, 0), beliefs=beliefs(1.0, 0.0, 0.9, 0.0, 0.0)),
            Change(at=(-2, 3), beliefs=beliefs(0.0, 0.5, 0.0, 0.0, 0.25)),
        ),
    )


def test_a_belief_beyond_one_is_rejected_by_validation() -> None:
    with pytest.raises(ValidationError):
        ChangeOutput.model_validate({"at": [1, 0], **beliefs(1.2, 0.0, 0.0, 0.0, 0.0)})
    with pytest.raises(ValidationError):
        ChangeOutput.model_validate({"at": [1, 0], **beliefs(0.0, -0.1, 0.0, 0.0, 0.0)})


@pytest.mark.parametrize("at", [[1], [1, 0, 2], []], ids=["one", "three", "none"])
def test_a_spot_of_other_than_two_numbers_is_rejected_by_validation(at: list[int]) -> None:
    with pytest.raises(ValidationError):
        ChangeOutput.model_validate({"at": at, **beliefs(0.0, 0.0, 0.0, 0.0, 0.0)})


def test_the_schema_of_a_spot_is_a_plain_array_of_two_integers() -> None:
    schema = AppraiseOutput.model_json_schema()
    assert "prefixItems" not in json.dumps(schema)
    at = schema["$defs"]["ChangeOutput"]["properties"]["at"]
    assert (at["type"], at["items"], at["minItems"], at["maxItems"]) == (
        "array",
        {"type": "integer"},
        2,
        2,
    )


def test_the_request_carries_the_llm_params() -> None:
    settings = model_settings(LLMParams(temperature=0.4, timeout_s=12.5))
    assert settings == {"timeout": 12.5, "temperature": 0.4}
    assert model_settings(LLMParams(temperature=None)) == {"timeout": 60.0}


SETTINGS = {"max_tokens": 8000, "openrouter_reasoning": {"effort": "low"}}


def test_the_config_settings_are_merged_into_the_request_as_given() -> None:
    settings = model_settings(LLMParams(temperature=0.4, timeout_s=12.5, settings=SETTINGS))
    assert settings == {"timeout": 12.5, "temperature": 0.4, **SETTINGS}


def test_nested_settings_survive_a_json_round_trip_of_the_run_config() -> None:
    config = with_llm(llm_config(), settings=SETTINGS)
    again = RunConfig.model_validate_json(config.model_dump_json())
    assert again == config
    assert again.params.llm.settings == SETTINGS


# --- retries -------------------------------------------------------------------------------------


def deliberate_ws(config: RunConfig, model: object) -> tuple[Workspace, Any]:
    rt = runtime(config, model)
    ws = Workspace(agent=initial_agent_state(config, rt.world.observe()))
    return asyncio.run(perceive(ws, rt)), rt


def with_llm(config: RunConfig, **llm: Any) -> RunConfig:
    params = config.params.model_copy(update={"llm": LLMParams(**llm)})
    return config.model_copy(update={"params": params})


@pytest.mark.parametrize(
    ("bad", "field"),
    [
        pytest.param({**DECISION, "direction": "up"}, "direction", id="direction"),
        pytest.param({**DECISION, "hunger": "famished"}, "hunger", id="feeling"),
        pytest.param({**DECISION, "damp": "soaked"}, "damp", id="damp"),
        pytest.param({**DECISION, "sure": "certain"}, "sure", id="sure"),
    ],
)
def test_an_unknown_word_is_retried_then_accepted(bad: dict[str, Any], field: str) -> None:
    model, calls = scripted(bad, DECISION)
    ws, rt = deliberate_ws(llm_config(), model)

    out = asyncio.run(deliberate(ws, rt))

    assert len(calls) == 2
    assert field in retry_feedback(calls[1])
    assert out.decision == to_decision(DeliberateOutput.model_validate(DECISION))


@pytest.mark.parametrize("retries", [0, 2])
@pytest.mark.parametrize(
    "bad",
    [{**DECISION, "direction": "up"}, {**DECISION, "hunger": "famished"}],
    ids=["direction", "feeling"],
)
def test_a_persistently_unknown_word_stops_with_an_error_naming_the_tick(
    bad: dict[str, Any], retries: int
) -> None:
    model, calls = scripted(bad)
    ws, rt = deliberate_ws(with_llm(llm_config(), output_retries=retries), model)
    ws = ws.model_copy(update={"agent": ws.agent.model_copy(update={"tick": 6})})

    with pytest.raises(UnexpectedModelBehavior) as info:
        asyncio.run(deliberate(ws, rt))

    assert len(calls) == retries + 1
    assert any("deliberate failed at tick 7" in note for note in info.value.__notes__)


def change(at: list[int], *values: float) -> dict[str, Any]:
    return {"at": at, **beliefs(*values)}


def appraise_config(max_changes: int = 2, **llm: Any) -> RunConfig:
    config = with_llm(llm_config(backends=Backends(deliberate="rule", appraise="llm")), **llm)
    params = config.params.model_copy(update={"appraise": AppraiseParams(max_changes=max_changes)})
    return config.model_copy(update={"params": params})


GOOD_CHANGE = change([1, 0], 1.0, 0.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("bad", "said"),
    [
        pytest.param([GOOD_CHANGE] * 3, "at most 2", id="too-many"),
        pytest.param([change([40, 0], 0.0, 0.0, 0.0, 0.0, 0.0)], "[40, 0]", id="too-far-east"),
        pytest.param([change([0, -13], 0.0, 0.0, 0.0, 0.0, 0.0)], "[0, -13]", id="too-far-south"),
        pytest.param([change([1, 0], 1.2, 0.0, 0.0, 0.0, 0.0)], "blocks", id="belief-above-one"),
    ],
)
def test_a_bad_list_of_changes_is_retried_then_accepted(bad: list[Any], said: str) -> None:
    model, calls = scripted(
        {"reading": "", "changes": bad}, {"reading": "", "changes": [GOOD_CHANGE]}
    )
    ws, rt = until_appraise(appraise_config(), model)

    out = asyncio.run(appraise(ws, rt))

    assert len(calls) == 2
    assert said in retry_feedback(calls[1])
    assert out.appraisal is not None
    assert out.appraisal.changes == (Change(at=(1, 0), beliefs=beliefs(1.0, 0.0, 0.0, 0.0, 0.0)),)


def test_twelve_changes_at_the_edge_of_reach_are_accepted() -> None:
    edge = [change([12, -12], 0.0, 0.0, 0.0, 0.0, 0.0)] * 2
    model, calls = scripted({"reading": "", "changes": edge})
    ws, rt = until_appraise(appraise_config(), model)
    out = asyncio.run(appraise(ws, rt))
    assert len(calls) == 1
    assert out.appraisal is not None and len(out.appraisal.changes) == 2


@pytest.mark.parametrize(
    "bad",
    [[GOOD_CHANGE] * 3, [change([40, 0], 0.0, 0.0, 0.0, 0.0, 0.0)]],
    ids=["too-many", "too-far"],
)
def test_a_persistently_bad_list_of_changes_stops_with_an_error_naming_the_tick(
    bad: list[Any],
) -> None:
    model, calls = scripted({"reading": "", "changes": bad})
    ws, rt = until_appraise(appraise_config(output_retries=1), model)

    with pytest.raises(UnexpectedModelBehavior) as info:
        asyncio.run(appraise(ws, rt))

    assert len(calls) == 2
    assert any("appraise failed at tick 1" in note for note in info.value.__notes__)


def test_of_two_changes_to_one_spot_in_one_answer_the_later_is_applied() -> None:
    first, later = change([0, 1], 1.0, 0.0, 0.0, 0.0, 0.0), change([0, 1], 0.0, 0.0, 0.0, 0.0, 0.75)
    rec = Recorder(appraise=lambda prompt: {"reading": "", "changes": [first, later]})
    with deliberate_agent.override(model=rec.model), appraise_agent.override(model=rec.model):
        result = asyncio.run(run(llm_config(ticks=1)))

    assert result.agent.own_map.tiles["0,1"] == beliefs(0.0, 0.0, 0.0, 0.0, 0.75)


# --- the two free lines --------------------------------------------------------------------------


def short_lines(config: RunConfig, line_chars: int) -> RunConfig:
    prompt = config.params.prompt.model_copy(update={"line_chars": line_chars})
    return config.model_copy(update={"params": config.params.model_copy(update={"prompt": prompt})})


def free_line(field: str, text: str, retries: int = 3) -> tuple[Workspace, Runtime, Any, Any]:
    """The phase that asks for `field`, its workspace and runtime with lines of at most 20
    characters, and the answer carrying `text` there."""
    if field == "why":
        config = short_lines(with_llm(llm_config(), output_retries=retries), 20)
        ws, rt = deliberate_ws(config, None)
        return ws, rt, deliberate, {**DECISION, "why": text}
    ws, rt = until_appraise(short_lines(appraise_config(output_retries=retries), 20))
    return ws, rt, appraise, {"reading": text, "changes": []}


LONG = "a line of more than twenty characters"


@pytest.mark.parametrize("field", ["why", "reading"])
def test_a_long_free_line_is_retried_then_accepted(field: str) -> None:
    ws, rt, phase, long = free_line(field, LONG)
    _, _, _, short = free_line(field, "short enough")
    model, calls = scripted(long, short)

    out = asyncio.run(phase(ws, dataclasses.replace(rt, model=model)))

    assert len(calls) == 2
    assert phrases.LINE_TOO_LONG.format(line_chars=20) in retry_feedback(calls[1])
    said = out.decision.why if field == "why" else out.appraisal.reading
    assert said == "short enough"


@pytest.mark.parametrize("field", ["why", "reading"])
def test_a_persistently_long_free_line_stops_with_an_error_naming_the_tick(field: str) -> None:
    ws, rt, phase, long = free_line(field, LONG, retries=1)
    model, calls = scripted(long)

    with pytest.raises(UnexpectedModelBehavior) as info:
        asyncio.run(phase(ws, dataclasses.replace(rt, model=model)))

    assert len(calls) == 2
    agent = "deliberate" if field == "why" else "appraise"
    assert any(f"{agent} failed at tick 1" in note for note in info.value.__notes__)


@pytest.mark.parametrize("field", ["why", "reading"])
def test_a_free_line_of_exactly_line_chars_passes_after_whitespace_is_collapsed(
    field: str,
) -> None:
    exact = "ten chars. ten chars"
    ws, rt, phase, answer = free_line(field, "  ten chars.\n\t  ten chars  ")
    model, calls = scripted(answer)

    out = asyncio.run(phase(ws, dataclasses.replace(rt, model=model)))

    assert len(calls) == 1
    said = out.decision.why if field == "why" else out.appraisal.reading
    assert said == exact and len(said) == 20


def test_a_model_phase_without_a_resolved_model_raises_config_error() -> None:
    ws, rt = deliberate_ws(llm_config(), None)
    with pytest.raises(ConfigError, match="no resolved model"):
        asyncio.run(deliberate(ws, rt))


# --- whole runs ----------------------------------------------------------------------------------


def test_a_twenty_tick_model_run_writes_valid_tick_files_and_replays_without_a_call(
    tmp_path: Path,
) -> None:
    rec = Recorder()
    run_dir = tmp_path / "run"
    with deliberate_agent.override(model=rec.model), appraise_agent.override(model=rec.model):
        result = asyncio.run(run(llm_config(ticks=20), run_dir))
    assert len(result.records) == 20
    assert len(rec.calls) == 40

    assert [d.tick for d in load_dumps(run_dir)] == list(range(1, 21))
    for path in sorted(run_dir.glob("tick_*.json")):
        dump = TickDump.model_validate_json(path.read_bytes())
        assert dump.record.decision.why == DECISION["why"]
        assert dump.record.appraisal.reading == APPRAISAL["reading"]
    assert json.loads((run_dir / "config.json").read_text())["model"] == "test"

    with (
        deliberate_agent.override(model=failing_model()),
        appraise_agent.override(model=failing_model()),
    ):
        assert asyncio.run(replay(run_dir)) == 20


def test_a_run_stopped_by_a_model_error_resumes_from_the_next_tick(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # the pilot, cut to twelve ticks, on the test model (JSON is YAML)
    experiment = load_config(EXPERIMENTS / "pilot.yaml").model_copy(
        update={"ticks": 12, "model": "test"}
    )
    path = tmp_path / "pilot.yaml"
    path.write_text(json.dumps(experiment.model_dump(mode="json")))
    args = ["run", str(path), "--runs-dir", str(tmp_path / "runs")]
    run_dir = tmp_path / "runs" / experiment.runs()[0].run_id

    broken = Recorder(fail_on=8)
    with deliberate_agent.override(model=broken.model), appraise_agent.override(model=broken.model):
        assert main(args) == 1
    assert "rerun with --resume" in capsys.readouterr().err
    assert [d.tick for d in load_dumps(run_dir)] == list(range(1, 8))

    working = Recorder()
    with (
        deliberate_agent.override(model=working.model),
        appraise_agent.override(model=working.model),
    ):
        assert main([*args, "--resume"]) == 0
    assert [d.tick for d in load_dumps(run_dir)] == list(range(1, 13))
    assert len(working.prompts("deliberate")) == 5
    first = json.loads(working.prompts("deliberate")[0])
    stored = load_dumps(run_dir)[6].agent
    assert first == creature_view(stored, stored.memory.episodes, experiment.params)


def test_a_model_error_in_a_run_names_the_tick_and_keeps_the_ticks_before(tmp_path: Path) -> None:
    rec = Recorder(fail_on=8)
    run_dir = tmp_path / "run"
    with (
        deliberate_agent.override(model=rec.model),
        appraise_agent.override(model=rec.model),
        pytest.raises(RuntimeError, match="the model is gone") as info,
    ):
        asyncio.run(run(llm_config(ticks=20), run_dir))
    assert any("deliberate failed at tick 8" in note for note in info.value.__notes__)
    assert [d.tick for d in load_dumps(run_dir)] == list(range(1, 8))


def test_a_model_run_counts_its_requests_and_tokens() -> None:
    rec, usage = Recorder(), TokenCount()
    with deliberate_agent.override(model=rec.model), appraise_agent.override(model=rec.model):
        result = asyncio.run(run(llm_config(ticks=5), usage=usage))

    assert len(result.records) == 5
    assert usage.requests >= 10
    assert usage.input_tokens > 0 and usage.output_tokens > 0


USAGE_LINE = re.compile(r"^\S+: \d+ model requests, \d+ input tokens, \d+ output tokens$", re.M)


def cli(args: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    rec = Recorder()
    with deliberate_agent.override(model=rec.model), appraise_agent.override(model=rec.model):
        code = main(args)
    out, err = capsys.readouterr()
    return code, out, err


def short_experiment(tmp_path: Path, name: str, **changes: Any) -> Path:
    """The two-maps experiment cut to two creatures, one seed and four ticks, with `changes`."""
    base = load_two_maps()
    experiment = base.model_copy(
        update={"name": name, "seeds": [1], "ticks": 4, "agents": base.agents[:2], **changes}
    )
    path = tmp_path / f"{name}.yaml"
    path.write_text(json.dumps(experiment.model_dump(mode="json")))
    return path


def test_the_run_command_prints_the_tokens_of_a_model_run_on_stderr_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    llm = short_experiment(tmp_path, "llm", backends=BOTH_LLM, model="test")
    stand_in = short_experiment(tmp_path, "rule")
    runs = ["--runs-dir", str(tmp_path / "runs")]

    code, out, err = cli(["run", str(llm), *runs], capsys)
    assert code == 0
    [line] = USAGE_LINE.findall(err)
    assert line.startswith("llm-plain-s1: ") and not line.startswith("llm-plain-s1: 0 ")
    assert not USAGE_LINE.search(out)
    json.loads(out)  # stdout is the metrics alone

    code, out, err = cli(["run", str(stand_in), *runs], capsys)
    assert code == 0
    assert not USAGE_LINE.search(err) and not USAGE_LINE.search(out)


def test_the_experiment_command_prints_the_tokens_of_each_model_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    llm = short_experiment(tmp_path, "llm", backends=BOTH_LLM, model="test")
    stand_in = short_experiment(tmp_path, "rule")
    runs = ["--runs-dir", str(tmp_path / "runs")]

    code, out, err = cli(["experiment", str(llm), *runs], capsys)
    assert code == 0 and out == ""
    assert [line.split(":")[0] for line in USAGE_LINE.findall(err)] == [
        "llm-plain-s1",
        "llm-wary-s1",
    ]

    code, out, err = cli(["experiment", str(stand_in), *runs], capsys)
    assert code == 0
    assert not USAGE_LINE.search(err)


def test_a_failed_model_run_prints_its_tokens_before_the_error_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    llm = short_experiment(tmp_path, "llm", backends=BOTH_LLM, model="test", ticks=12)
    runs = ["--runs-dir", str(tmp_path / "runs")]
    rec = Recorder(fail_on=8)
    with deliberate_agent.override(model=rec.model), appraise_agent.override(model=rec.model):
        code = main(["run", str(llm), *runs])
    out, err = capsys.readouterr()
    assert code == 1 and out == ""
    lines = err.splitlines()
    assert USAGE_LINE.match(lines[0]) and not lines[0].startswith("llm-plain-s1: 0 ")
    assert lines[1].startswith("error: llm-plain-s1 failed")
    assert len(lines) == 2


def test_a_failed_model_run_of_an_experiment_prints_its_tokens(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    llm = short_experiment(tmp_path, "llm", backends=BOTH_LLM, model="test", ticks=12)
    runs = ["--runs-dir", str(tmp_path / "runs")]
    rec = Recorder(fail_on=8)
    with deliberate_agent.override(model=rec.model), appraise_agent.override(model=rec.model):
        code = main(["experiment", str(llm), *runs])
    err = capsys.readouterr().err
    assert code == 1
    usage = USAGE_LINE.findall(err)
    assert usage and all(not line.split(": ")[1].startswith("0 ") for line in usage)
    assert err.index(usage[-1]) < err.index("error: ")


def test_a_complete_run_not_rerun_prints_no_token_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    llm = short_experiment(tmp_path, "llm", backends=BOTH_LLM, model="test")
    runs = ["--runs-dir", str(tmp_path / "runs")]
    code, _, err = cli(["experiment", str(llm), *runs], capsys)
    assert code == 0 and len(USAGE_LINE.findall(err)) == 2

    code, _, err = cli(["experiment", str(llm), *runs], capsys)
    assert code == 0
    assert "complete, not rerun" in err
    assert not USAGE_LINE.search(err)


def test_resuming_a_complete_model_run_needs_no_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    llm = short_experiment(tmp_path, "llm", backends=BOTH_LLM)
    args = ["run", str(llm), "--runs-dir", str(tmp_path / "runs")]
    monkeypatch.setenv(MODEL_ENV_VAR, "test")
    code, first, _ = cli(args, capsys)
    assert code == 0

    monkeypatch.delenv(MODEL_ENV_VAR)
    rec = Recorder()
    with deliberate_agent.override(model=rec.model), appraise_agent.override(model=rec.model):
        code = main([*args, "--resume"])
    out, err = capsys.readouterr()
    assert code == 0, err
    assert out == first
    assert rec.calls == []
    assert not USAGE_LINE.search(err)


def test_an_llm_backend_with_no_model_fails_before_tick_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
    rec = Recorder()
    run_dir = tmp_path / "run"
    with (
        deliberate_agent.override(model=rec.model),
        appraise_agent.override(model=rec.model),
        pytest.raises(ConfigError, match=MODEL_ENV_VAR),
    ):
        asyncio.run(run(llm_config(model=None), run_dir))
    assert rec.calls == []
    assert not run_dir.exists()


@pytest.mark.parametrize(
    ("backends", "asked"),
    [
        (Backends(deliberate="llm", appraise="reasoner"), "deliberate"),
        (Backends(deliberate="rule", appraise="llm"), "appraise"),
    ],
    ids=["model-deliberate", "model-appraise"],
)
def test_mixed_backends_run(backends: Backends, asked: str) -> None:
    rec = Recorder()
    with deliberate_agent.override(model=rec.model), appraise_agent.override(model=rec.model):
        result = asyncio.run(run(llm_config(backends=backends, ticks=8)))

    assert len(result.records) == 8
    assert {name for name, _, _ in rec.calls} == {asked}
    assert len(rec.calls) == 8
    for r in result.records:
        assert (r.decision.why == DECISION["why"]) is (asked == "deliberate")
        assert (r.appraisal.reading == APPRAISAL["reading"]) is (asked == "appraise")


def test_the_whole_run_on_the_test_model_works(tmp_path: Path) -> None:
    result = asyncio.run(run(llm_config(ticks=5), tmp_path / "run"))
    assert len(result.records) == 5


# --- the experiment files ------------------------------------------------------------------------


PILOT_LLM = LLMParams(temperature=0.5, timeout_s=120, settings=SETTINGS)


def test_the_pilot_is_the_plain_creature_on_a_model() -> None:
    pilot, base = load_config(EXPERIMENTS / "pilot.yaml"), load_two_maps()
    assert pilot.seeds == [1]
    assert pilot.ticks == 60
    assert pilot.backends == BOTH_LLM
    assert pilot.model is None
    assert pilot.world == base.world
    assert pilot.agents == [a for a in base.agents if a.name == "plain"]
    assert pilot.params == base.params.model_copy(update={"llm": PILOT_LLM})


def test_two_maps_on_a_model_is_three_creatures_on_one_seed() -> None:
    llm, base = load_config(EXPERIMENTS / "two_maps_llm.yaml"), load_two_maps()
    assert llm.seeds == [1]
    assert llm.ticks is None
    assert llm.params == base.params.model_copy(update={"llm": PILOT_LLM})
    assert llm.backends == BOTH_LLM
    assert llm.model is None
    assert llm.world == base.world
    named = {a.name: a for a in base.agents}
    assert llm.agents == [named["plain"], named["wary"], named["hopeful"]]
    assert re.fullmatch(r"[a-z_]+", llm.name)
