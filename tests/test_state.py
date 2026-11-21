"""The state models: bounds, required keys, the creature's map, and JSON round trips."""

import math
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from surprise_loop.state import (
    AgentState,
    Appraisal,
    Change,
    Comparison,
    CreatureMap,
    Decision,
    Episode,
    Expectation,
    FeltRecord,
    Measures,
    Memory,
    Percept,
    ReasonerState,
    RuleChoice,
    Sureness,
    Temperament,
    TickDump,
    TickRecord,
    Touch,
    Workspace,
)
from surprise_loop.world import (
    Belief,
    Channel,
    Direction,
    Kind,
    Outcome,
    Signal,
    StepInfo,
    WorldState,
)

PLAIN = {
    Kind.GROUND: 0.60,
    Kind.PLEASANT: 0.06,
    Kind.UNPLEASANT: 0.05,
    Kind.OBSTACLE: 0.12,
    Kind.FOOD: 0.06,
    Kind.TOXIC: 0.03,
    Kind.WATER: 0.06,
    Kind.FOUL: 0.02,
}
UNTRIED = {
    Belief.BLOCKS: 0.29,
    Belief.HURTS: 0.1,
    Belief.FEEDS: 0.09,
    Belief.QUENCHES: 0.08,
    Belief.PLEASES: 0.06,
}
FOOD = {
    Belief.BLOCKS: 1.0,
    Belief.HURTS: 0.0,
    Belief.FEEDS: 0.9,
    Belief.QUENCHES: 0.0,
    Belief.PLEASES: 0.0,
}
BAD = [-0.1, 1.1, math.nan, math.inf, -math.inf]


def levels(**steps: int) -> dict[Channel, int]:
    return {channel: steps.get(channel.value, 0) for channel in Channel}


def felt(**changes: Any) -> FeltRecord:
    data: dict[str, Any] = {
        "moved": True,
        "levels": levels(hunger=1, thirst=2),
        "smell": 1,
        "damp": 0,
        "tone": 2,
        "stir": 1,
        "at": (-1, 2),
    }
    return FeltRecord.model_validate(data | changes)


def expectation() -> Expectation:
    return Expectation(
        moved=False, levels=levels(hunger=1), smell=0, damp=2, sure=Sureness.FAIRLY_SURE
    )


def comparison() -> Comparison:
    return Comparison(
        moved_miss=True,
        misses={channel: -1 if channel is Channel.HUNGER else 0 for channel in Channel},
        smell_miss=1,
        damp_miss=-2,
        surprise=0.375,
        worse=-0.25,
    )


def episode() -> Episode:
    return Episode(
        tick=3,
        direction=Direction.WEST,
        expectation=expectation(),
        felt=felt(),
        comparison=comparison(),
        reading="the bush to the west fed me",
        salience=0.6,
    )


def agent_state() -> AgentState:
    return AgentState(
        temperament=Temperament(presumes=PLAIN, blur=0.25),
        felt=felt(),
        stir=0.3,
        own_map=CreatureMap(untried=UNTRIED, tiles={"-2,2": FOOD, "0,1": UNTRIED}),
        reasoner=ReasonerState(
            presumes=PLAIN,
            tiles={"-2,2": {kind: 1.0 if kind is Kind.FOOD else 0.0 for kind in Kind}},
            touched=(Touch(step=3, at=(-2, 2)),),
            onsets=(2,),
        ),
        tries={"-2,2": 2, "0,1": 1},
        memory=Memory(episodes=[episode()]),
        tick=4,
    )


def tick_record() -> TickRecord:
    signal = Signal(
        moved=False,
        body=dict.fromkeys(Channel, 0.1) | {Channel.HUNGER: 0.35},
        smell=1,
        damp=0,
        position=(-1, 2),
    )
    return TickRecord(
        tick=4,
        felt_before=felt(at=(-1, 2)),
        rule_best=(Direction.NORTH, Direction.WEST),
        decision=Decision(direction=Direction.WEST, expectation=expectation(), why="hungry"),
        outcome=Outcome(
            signal=signal,
            info=StepInfo(step=4, target=(-2, 2), target_kind=Kind.FOOD, sickness_arrived=False),
        ),
        felt=felt(moved=False),
        comparison=comparison(),
        reasoner_changed=((-2, 2), (-1, 3)),
        appraisal=Appraisal(reading="it fed me", changes=(Change(at=(-2, 2), beliefs=FOOD),)),
        measures=Measures(
            tiles_tried=2,
            start_error=0.2,
            own_error=0.1,
            reasoner_error=0.05,
            gap=0.04,
            own_error_all=0.3,
            reasoner_error_all=0.25,
            gap_all=0.1,
            false_harm=0,
            false_safe=1,
            reasoner_false_harm=0,
            reasoner_false_safe=0,
            wrong=("-2,2",),
        ),
    )


# --- bounds and unknown fields ---------------------------------------------------------------


@pytest.mark.parametrize("value", BAD)
@pytest.mark.parametrize(
    ("model", "field", "valid"),
    [
        (Comparison, "surprise", comparison()),
        (Temperament, "blur", Temperament(presumes=PLAIN)),
        (AgentState, "stir", agent_state()),
        (Episode, "salience", episode()),
    ],
)
def test_unit_fields_reject_out_of_range_nan_and_inf(
    model: type[BaseModel], field: str, valid: BaseModel, value: float
) -> None:
    data = valid.model_dump() | {field: value}
    with pytest.raises(ValidationError, match=field):
        model.model_validate(data)


@pytest.mark.parametrize("value", [-1, 5])
def test_five_word_steps_are_bounded(value: int) -> None:
    with pytest.raises(ValidationError):
        felt(tone=value)
    with pytest.raises(ValidationError):
        felt(levels=levels(ache=value))


@pytest.mark.parametrize("value", [-1, 3])
def test_three_word_steps_are_bounded(value: int) -> None:
    for field in ("smell", "damp", "stir"):
        with pytest.raises(ValidationError, match=field):
            felt(**{field: value})


@pytest.mark.parametrize("value", BAD)
def test_beliefs_in_a_map_are_bounded(value: float) -> None:
    with pytest.raises(ValidationError):
        CreatureMap(untried=UNTRIED | {Belief.HURTS: value}, tiles={})
    with pytest.raises(ValidationError):
        CreatureMap(untried=UNTRIED, tiles={"0,1": FOOD | {Belief.FEEDS: value}})
    with pytest.raises(ValidationError):
        Change(at=(0, 1), beliefs=FOOD | {Belief.FEEDS: value})


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_plain_floats_reject_nan_and_inf(value: float) -> None:
    with pytest.raises(ValidationError, match="worse"):
        Comparison.model_validate(comparison().model_dump() | {"worse": value})
    with pytest.raises(ValidationError, match="gap"):
        Measures.model_validate(tick_record().measures.model_dump() | {"gap": value})


@pytest.mark.parametrize("value", [-0.01, math.nan, math.inf])
def test_gain_and_noise_reject_negative_nan_and_inf(value: float) -> None:
    with pytest.raises(ValidationError):
        Temperament(presumes=PLAIN, gain=dict.fromkeys(Channel, 1.0) | {Channel.ACHE: value})
    with pytest.raises(ValidationError):
        Temperament(presumes=PLAIN, noise=dict.fromkeys(Channel, 0.02) | {Channel.ACHE: value})


def test_counts_reject_negative_values() -> None:
    with pytest.raises(ValidationError):
        Touch(step=-1, at=(0, 0))
    with pytest.raises(ValidationError):
        AgentState.model_validate(agent_state().model_dump() | {"tries": {"0,1": -1}})
    with pytest.raises(ValidationError):
        AgentState.model_validate(agent_state().model_dump() | {"tick": -1})


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="extra"):
        FeltRecord.model_validate(felt().model_dump() | {"heat": 1})
    with pytest.raises(ValidationError, match="extra"):
        AgentState.model_validate(agent_state().model_dump() | {"mood": 0.5})


def test_a_missing_key_of_an_enum_keyed_dict_is_rejected() -> None:
    partial = {channel: 0 for channel in Channel if channel is not Channel.PLEASURE}
    with pytest.raises(ValidationError, match="missing keys: pleasure"):
        felt(levels=partial)
    with pytest.raises(ValidationError, match="missing keys: pleases"):
        CreatureMap(untried={b: 0.0 for b in Belief if b is not Belief.PLEASES}, tiles={})


def test_workspace_is_frozen() -> None:
    ws = Workspace(agent=agent_state())
    with pytest.raises(ValidationError, match="frozen"):
        ws.stir = 0.5  # type: ignore[misc]
    assert ws.percept is None and ws.record is None


def test_sureness_words() -> None:
    assert [s.value for s in Sureness] == ["guessing", "fairly sure", "sure"]


# --- temperament -----------------------------------------------------------------------------


def test_a_missing_kind_in_presumes_is_rejected() -> None:
    shares = {kind: share for kind, share in PLAIN.items() if kind is not Kind.FOUL}
    with pytest.raises(ValidationError, match="missing keys: foul"):
        Temperament(presumes=shares | {Kind.GROUND: 0.62})


def test_presumes_must_sum_to_one() -> None:
    with pytest.raises(ValidationError, match="sum to 1"):
        Temperament(presumes=PLAIN | {Kind.GROUND: 0.50})


def test_gain_and_noise_default_for_every_channel() -> None:
    t = Temperament(presumes=PLAIN)
    assert t.gain == dict.fromkeys(Channel, 1.0)
    assert t.noise == dict.fromkeys(Channel, 0.02)
    assert t.blur == 0.0


# --- the creature's map ----------------------------------------------------------------------


def test_map_at_gives_the_entry_or_untried() -> None:
    m = CreatureMap(untried=UNTRIED, tiles={"-2,2": FOOD})
    assert m.at((-2, 2)) == FOOD
    assert m.at((3, 3)) == UNTRIED


def test_changed_applies_changes_in_order_and_leaves_the_original() -> None:
    m = CreatureMap(untried=UNTRIED, tiles={"-2,2": FOOD})
    first = UNTRIED | {Belief.HURTS: 0.9}
    later = UNTRIED | {Belief.PLEASES: 0.7}
    new = m.changed([Change(at=(1, 0), beliefs=first), Change(at=(1, 0), beliefs=later)])
    assert new.at((1, 0)) == later
    assert new.at((-2, 2)) == FOOD
    assert m.tiles == {"-2,2": FOOD}
    assert m.at((1, 0)) == UNTRIED


# --- rule choice -----------------------------------------------------------------------------


def test_rule_choice_best_is_nonempty_ordered_and_holds_the_choice() -> None:
    RuleChoice(best=(Direction.NORTH, Direction.WEST), choice=Direction.WEST)
    with pytest.raises(ValidationError, match="at least one"):
        RuleChoice(best=(), choice=Direction.NORTH)
    with pytest.raises(ValidationError, match="Direction order"):
        RuleChoice(best=(Direction.WEST, Direction.NORTH), choice=Direction.WEST)
    with pytest.raises(ValidationError, match="Direction order"):
        RuleChoice(best=(Direction.EAST, Direction.EAST), choice=Direction.EAST)
    with pytest.raises(ValidationError, match="one of best"):
        RuleChoice(best=(Direction.EAST,), choice=Direction.SOUTH)


# --- JSON round trips ------------------------------------------------------------------------


def test_agent_state_survives_a_json_round_trip() -> None:
    state = agent_state()
    back = AgentState.model_validate_json(state.model_dump_json())
    assert back == state
    assert back.felt.at == (-1, 2)
    assert back.reasoner.touched[0].at == (-2, 2)


def test_tick_dump_survives_a_json_round_trip() -> None:
    dump = TickDump(
        run_id="two_maps-plain-s1",
        tick=4,
        agent=agent_state(),
        world=WorldState(
            seed=1,
            step=4,
            position=(-1, 2),
            body=dict.fromkeys(Channel, 0.1),
            pending_sickness=(6,),
        ),
        record=tick_record(),
    )
    back = TickDump.model_validate_json(dump.model_dump_json())
    assert back == dump
    assert back.record.reasoner_changed == ((-2, 2), (-1, 3))
    assert back.world.position == (-1, 2)


def test_workspace_with_every_slot_survives_a_json_round_trip() -> None:
    record = tick_record()
    ws = Workspace(
        agent=agent_state(),
        percept=Percept(
            recent=(episode(),), rule=RuleChoice(best=(Direction.WEST,), choice=Direction.WEST)
        ),
        decision=record.decision,
        outcome=record.outcome,
        felt_after=record.felt,
        stir=0.4,
        comparison=record.comparison,
        reasoner=agent_state().reasoner,
        reasoner_changed=record.reasoner_changed,
        appraisal=record.appraisal,
        record=record,
    )
    assert Workspace.model_validate_json(ws.model_dump_json()) == ws
