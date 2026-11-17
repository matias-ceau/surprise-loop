import math
from typing import Any

import pytest
from bodies import HOTTER, ILO, ILO_AFTERNOON
from pydantic import BaseModel, ValidationError

from surprise_loop import config, state, world
from surprise_loop.state import (
    AgentState,
    Appraisal,
    Band,
    BodyLine,
    Comparison,
    Decision,
    Episode,
    Expectation,
    Feedback,
    Feeling,
    FeltFact,
    FeltScene,
    Memory,
    Offer,
    Override,
    PassRecord,
    Percept,
    Pulls,
    Sureness,
    SurpriseNote,
    TickDump,
    TickRecord,
    Workspace,
)
from surprise_loop.world import (
    Action,
    BodyState,
    Channel,
    Event,
    FactName,
    Levels,
    Observation,
    OpenQuestion,
    Outcome,
    Place,
    StepInfo,
    WorldState,
    frozen_noon_facts,
)

NAN, INF = math.nan, math.inf


def signal() -> Levels:
    return Levels(heat=0.75, water=0.6, energy=0.25)


def observation() -> Observation:
    return Observation(
        step=3,
        minute=810,
        place=Place.MEADOW,
        last_action=Action.LOOK,
        event=Event.LOOKED,
        facts=frozen_noon_facts(),
        signal=signal(),
        offers=[Action.GO_OAK, Action.GO_STREAM, Action.REST, Action.LOOK],
    )


def body() -> BodyState:
    return BodyState(heat=0.99, water=0.47, energy=0.22, gut=0.1)


def outcome() -> Outcome:
    info = StepInfo(
        step=3,
        minute=810,
        place=Place.MEADOW,
        air_c=30.0,
        sun=1.0,
        log_shaded=False,
        stream_full=True,
        body=body(),
        stumbled=False,
        collapsed=False,
    )
    return Outcome(observation=observation(), info=info)


def expectation() -> Expectation:
    return Expectation(
        words={Channel.HEAT: "bit_less", Channel.WATER: "same", Channel.ENERGY: "same"},
        bands={
            Channel.HEAT: Band(lo=0.4, hi=0.6),
            Channel.WATER: Band(lo=None, hi=0.6),
            Channel.ENERGY: Band(lo=0.2, hi=None),
        },
        world=True,
        sureness=Sureness.FAIRLY_SURE,
    )


def comparison() -> Comparison:
    return Comparison(
        body={Channel.HEAT: 0.15, Channel.WATER: 0.0, Channel.ENERGY: 0.0},
        body_felt={Channel.HEAT: 0.45, Channel.WATER: 0.0, Channel.ENERGY: 0.0},
        world=False,
        better=-0.45,
        size=0.45,
    )


def feeling() -> Feeling:
    return Feeling(discomfort=6.5, valence=-0.6, arousal=0.7, mood=-0.3, phasic=0.0)


def scene() -> FeltScene:
    return FeltScene(
        place=Place.MEADOW,
        opening="You are standing in the middle of the forest clearing.",
        facts=[
            FeltFact(
                name=FactName.SUN,
                about=None,
                weight=0.4,
                value=-2.5,
                sign="-",
                grade=3,
                noticed=True,
                phrase="The sun beats down on your head.",
            ),
            FeltFact(
                name=FactName.BIRDSONG,
                about=None,
                weight=0.1,
                value=0.0,
                sign="0",
                grade=0,
                noticed=False,
                phrase=None,
            ),
        ],
        body=[BodyLine(channel=Channel.HEAT, grade=3, felt=2.5, phrase="Your head is thick.")],
        surprise=None,
        felt={Channel.HEAT: 2.5, Channel.WATER: 1.3, Channel.ENERGY: 2.7},
    )


def episode(tick: int = 0) -> Episode:
    return Episode(
        tick=tick,
        step=tick + 1,
        place=Place.MEADOW,
        action=Action.GO_OAK,
        event=Event.ARRIVED,
        expectation=expectation(),
        comparison=comparison(),
        need_before={Channel.HEAT: 2.5, Channel.WATER: 1.3, Channel.ENERGY: 2.7},
        salience=0.35,
        scene="You are standing in the middle of the forest clearing.",
        reading="It is too hot here.",
        regulation="That helped.",
        score=0.4,
    )


def pulls(value: float = 0.0) -> Pulls:
    return Pulls(values={p: {a: value for a in Action} for p in Place})


def agent_state() -> AgentState:
    return AgentState(
        temperament=ILO,
        belief=ILO_AFTERNOON,
        feeling=feeling(),
        memory=Memory(episodes=[episode(0), episode(1)]),
        pulls=pulls(0.1),
        attention=Channel.HEAT,
        voiced=HOTTER,
        tick=2,
    )


def decision() -> Decision:
    return Decision(action=Action.GO_OAK, expectation=expectation(), why="shade")


def feedback() -> Feedback:
    return Feedback(judged_step=1, score=0.5, line="That helped.")


def override() -> Override:
    return Override(channel=Channel.HEAT, actions=[Action.GO_OAK], detail="too hot")


def pass_record(pass_index: int = 0) -> PassRecord:
    return PassRecord(
        tick=2,
        pass_index=pass_index,
        scene=scene(),
        offers=[Action.GO_OAK, Action.REST],
        decision=decision(),
        outcome=outcome(),
        comparison=comparison(),
        feedback=feedback(),
        belief_after=ILO_AFTERNOON,
        feeling_after=feeling(),
        override=override() if pass_index == 0 else None,
    )


def tick_record() -> TickRecord:
    return TickRecord(
        tick=2,
        observation=observation(),
        appraisal=Appraisal(reading="It is too hot here.", about=Channel.HEAT),
        passes=[pass_record(0), pass_record(1)],
    )


def world_state() -> WorldState:
    return WorldState(
        seed=7,
        step=3,
        minute=810,
        place=Place.MEADOW,
        body=body(),
        water_history=[0.48, 0.47],
        stream_full=True,
        forced_rest_left=0,
        last_observation=observation(),
    )


def tick_dump() -> TickDump:
    return TickDump(
        run_id="clearing-ilo-s7",
        tick=2,
        agent=agent_state(),
        world=world_state(),
        record=tick_record(),
    )


def percept() -> Percept:
    return Percept(
        scene=scene(),
        offers=[Offer(action=Action.GO_OAK, phrase="The oak, a few steps off.")],
        recalled=[episode()],
        pull_line="",
    )


# --- bounds ----------------------------------------------------------------------------------

# (model, factory of a valid instance, field, out-of-range values); NaN and ±inf are added below
BOUNDED: list[tuple[type[BaseModel], Any, str, list[Any]]] = [
    (Feedback, feedback, "score", [-1.01, 1.01]),
    (Feedback, feedback, "judged_step", [-1]),
    (Episode, episode, "tick", [-1]),
    (Episode, episode, "step", [-1]),
    (Episode, episode, "salience", [-0.01, 1.01]),
    (Episode, episode, "score", [-1.01, 1.01]),
    (AgentState, agent_state, "tick", [-1]),
    (PassRecord, pass_record, "tick", [-1]),
    (PassRecord, pass_record, "pass_index", [-1, 2]),
    (TickRecord, tick_record, "tick", [-1]),
    (TickDump, tick_dump, "tick", [-1]),
]

CASES = [
    pytest.param(model, make, field, bad, id=f"{model.__name__}.{field}={bad}")
    for model, make, field, bads in BOUNDED
    for bad in [*bads, NAN, INF, -INF]
]


@pytest.mark.parametrize(("model", "make", "field", "bad"), CASES)
def test_bounded_fields_reject_out_of_range_nan_and_inf(
    model: type[BaseModel], make: Any, field: str, bad: Any
) -> None:
    data = make().model_dump()
    model.model_validate(data)  # the baseline is valid
    data[field] = bad
    with pytest.raises(ValidationError):
        model.model_validate(data)


@pytest.mark.parametrize("bad", [NAN, INF, -INF])
def test_episode_needs_reject_nan_and_inf(bad: float) -> None:
    data = episode().model_dump()
    data["need_before"][Channel.WATER] = bad
    with pytest.raises(ValidationError):
        Episode.model_validate(data)


@pytest.mark.parametrize("bad", [NAN, INF, -INF])
def test_pulls_reject_nan_and_inf(bad: float) -> None:
    data = pulls().model_dump()
    data["values"][Place.OAK][Action.REST] = bad
    with pytest.raises(ValidationError):
        Pulls.model_validate(data)


@pytest.mark.parametrize(
    "make",
    [
        episode,
        agent_state,
        pass_record,
        tick_record,
        tick_dump,
        percept,
        decision,
        feedback,
        override,
        pulls,
        lambda: Appraisal(reading="calm", about=None),
        lambda: Offer(action=Action.REST, phrase="Sit down here."),
        lambda: Memory(episodes=[]),
        lambda: Workspace(agent=agent_state(), observation=observation()),
    ],
)
def test_unknown_fields_are_rejected(make: Any) -> None:
    instance = make()
    data = instance.model_dump()
    data["surprise"] = 1
    with pytest.raises(ValidationError, match="surprise"):
        type(instance).model_validate(data)


def test_episode_needs_cover_every_channel() -> None:
    data = episode().model_dump()
    del data["need_before"][Channel.ENERGY]
    with pytest.raises(ValidationError, match="energy"):
        Episode.model_validate(data)


def test_pulls_cover_every_place_and_every_action() -> None:
    data = pulls().model_dump()
    del data["values"][Place.STREAM]
    with pytest.raises(ValidationError, match="stream"):
        Pulls.model_validate(data)
    data = pulls().model_dump()
    del data["values"][Place.OAK][Action.DRINK]
    with pytest.raises(ValidationError, match="drink"):
        Pulls.model_validate(data)


def test_override_needs_at_least_one_action() -> None:
    with pytest.raises(ValidationError):
        Override(channel=Channel.HEAT, actions=[], detail="")


def test_tick_record_has_one_or_two_passes() -> None:
    data = tick_record().model_dump()
    TickRecord.model_validate({**data, "passes": data["passes"][:1]})
    for passes in ([], data["passes"] + data["passes"][:1]):
        with pytest.raises(ValidationError):
            TickRecord.model_validate({**data, "passes": passes})


def test_text_fields_default_to_empty() -> None:
    assert Decision(action=Action.REST, expectation=expectation()).why == ""
    assert Feedback(judged_step=None, score=None).line == ""
    plain = episode().model_dump()
    del plain["regulation"], plain["score"]
    loaded = Episode.model_validate(plain)
    assert loaded.regulation == ""
    assert loaded.score is None


# --- the surprise note -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "fields",
    [
        {"channel": Channel.WATER, "worse": False},
        {"question": OpenQuestion.STREAM_FULL, "actual": False},
    ],
    ids=["body", "world"],
)
def test_surprise_note_holds_one_pair(fields: dict[str, Any]) -> None:
    SurpriseNote(**fields)


@pytest.mark.parametrize(
    "fields",
    [
        {"channel": Channel.HEAT, "worse": True, "question": OpenQuestion.LOG_SHADED},
        {"channel": Channel.HEAT, "worse": True, "actual": True},
        {
            "channel": Channel.HEAT,
            "worse": True,
            "question": OpenQuestion.LOG_SHADED,
            "actual": True,
        },
        {"channel": Channel.HEAT},
        {"worse": True},
        {"question": OpenQuestion.LOG_SHADED},
        {"actual": False},
        {},
    ],
    ids=["body+question", "body+actual", "both pairs", "channel only", "worse only",
         "question only", "actual only", "empty"],
)  # fmt: skip
def test_surprise_note_rejects_mixed_half_or_empty_pairs(fields: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="surprise"):
        SurpriseNote(**fields)


# --- round trips -----------------------------------------------------------------------------


def test_agent_state_survives_json_round_trip() -> None:
    s = agent_state()
    assert AgentState.model_validate_json(s.model_dump_json()) == s


def test_tick_dump_survives_json_round_trip() -> None:
    dump = tick_dump()
    text = dump.model_dump_json()
    loaded = TickDump.model_validate_json(text)
    assert loaded == dump
    assert loaded.model_dump_json() == text


def test_tick_dump_with_unset_optional_fields_reserializes_identically() -> None:
    record = pass_record(0).model_copy(
        update={"override": None, "feedback": Feedback(judged_step=None, score=None)}
    )
    agent = agent_state().model_copy(update={"attention": None, "voiced": None})
    dump = tick_dump().model_copy(
        update={"agent": agent, "record": tick_record().model_copy(update={"passes": [record]})}
    )
    text = dump.model_dump_json()
    loaded = TickDump.model_validate_json(text)
    assert loaded == dump
    assert loaded.model_dump_json() == text


# --- memory (as v0) --------------------------------------------------------------------------


def test_memory_last_k_returns_the_last_k_in_order() -> None:
    episodes = [episode(t) for t in range(5)]
    memory = Memory(episodes=episodes)
    assert memory.last_k(2) == episodes[3:]
    assert memory.last_k(5) == episodes
    assert memory.last_k(0) == []


def test_memory_last_k_returns_all_when_k_exceeds_length() -> None:
    episodes = [episode(t) for t in range(3)]
    assert Memory(episodes=episodes).last_k(10) == episodes


def test_memory_last_k_of_empty_memory_is_empty() -> None:
    assert Memory(episodes=[]).last_k(3) == []


# ticks 0..7; ticks 1 and 3 tie on the highest salience
SALIENCES = [0.5, 0.9, 0.2, 0.9, 0.1, 0.3, 0.7, 0.4]


def salient_memory(saliences: list[float] = SALIENCES) -> Memory:
    return Memory(
        episodes=[episode(t).model_copy(update={"salience": s}) for t, s in enumerate(saliences)]
    )


@pytest.mark.parametrize(
    ("k", "ticks"),
    [
        # k // 2 = 0 recent: the single most salient, the tie going to the more recent tick 3
        (1, [3]),
        # tick 7 recent; the most salient before it is tick 3 (tie with tick 1)
        (2, [3, 7]),
        # tick 7 recent; both tied ticks fill the two salient slots
        (3, [1, 3, 7]),
        # ticks 6, 7 recent; ticks 3, 1 most salient before them
        (4, [1, 3, 6, 7]),
        # ticks 6, 7 recent; ticks 3, 1, 0 most salient before them
        (5, [0, 1, 3, 6, 7]),
        # ticks 5, 6, 7 recent (tick 6, salience 0.7, is not counted twice); then 3, 1, 0
        (6, [0, 1, 3, 5, 6, 7]),
        # ticks 5, 6, 7 recent; every earlier tick but the least salient (tick 4)
        (7, [0, 1, 2, 3, 5, 6, 7]),
    ],
)
def test_memory_salient_k_selects_recent_half_plus_most_salient(k: int, ticks: list[int]) -> None:
    recalled = salient_memory().salient_k(k)
    assert [e.tick for e in recalled] == ticks
    assert len(recalled) == k


def test_memory_salient_k_ties_go_to_the_more_recent() -> None:
    memory = salient_memory([0.5] * 8)
    for k in range(9):
        assert memory.salient_k(k) == memory.last_k(k)


@pytest.mark.parametrize("n", [0, 1, 2, 5, 8])
def test_memory_salient_k_equals_last_k_with_at_most_k_episodes(n: int) -> None:
    memory = salient_memory(SALIENCES[:n])
    for k in range(n, n + 3):
        assert memory.salient_k(k) == memory.last_k(k) == memory.episodes


def test_memory_salient_k_zero_and_negative_match_last_k() -> None:
    memory = salient_memory()
    assert memory.salient_k(0) == memory.last_k(0) == []
    with pytest.raises(ValueError, match="k must be >= 0"):
        memory.salient_k(-1)
    with pytest.raises(ValueError, match="k must be >= 0"):
        memory.last_k(-1)


# --- frozen ----------------------------------------------------------------------------------


def test_workspace_is_frozen_and_has_documented_defaults() -> None:
    ws = Workspace(agent=agent_state(), observation=observation())
    assert ws.pass_index == 0
    assert ws.passes == ()
    for name in (
        "percept",
        "appraisal",
        "override",
        "decision",
        "outcome",
        "comparison",
        "feedback",
    ):
        assert getattr(ws, name) is None
    with pytest.raises(ValidationError):
        ws.pass_index = 1  # type: ignore[misc]
    with pytest.raises(ValidationError):
        Workspace(agent=agent_state(), observation=observation(), pass_index=2)


def test_workspace_copies_carry_passes_as_a_tuple() -> None:
    ws = Workspace(agent=agent_state(), observation=observation())
    later = ws.model_copy(update={"percept": percept(), "passes": (pass_record(),)})
    assert ws.percept is None and ws.passes == ()
    assert later.passes == (pass_record(),)


def test_state_models_are_frozen() -> None:
    s = agent_state()
    with pytest.raises(ValidationError):
        s.tick = 3  # type: ignore[misc]
    with pytest.raises(ValidationError):
        s.feeling.mood = 0.5  # type: ignore[misc]


def test_every_model_is_frozen_and_forbids_unknown_fields() -> None:
    """Every pydantic model of the state, world and config modules."""
    modules = [state, world, config]
    found = [
        obj
        for module in modules
        for obj in vars(module).values()
        if isinstance(obj, type)
        and issubclass(obj, BaseModel)
        and obj.__module__ == module.__name__
    ]
    assert len(found) > 50
    for model in found:
        assert model.model_config.get("extra") == "forbid", model.__name__
        assert model.model_config.get("frozen") is True, model.__name__
        assert model.model_config.get("allow_inf_nan") is False, model.__name__
