"""Reconcile: belief, feeling, attention, the voiced surprise, memory, pulls, the override and the
pass record, on hand-built workspaces."""

import pytest
from bodies import ILO
from test_phases import PARAMS, agent_with, base_ws, episode, make_rt, remembering, run

from surprise_loop.config import Runtime
from surprise_loop.feeling import feel, needs
from surprise_loop.phases import evaluate, reconcile
from surprise_loop.phases.heuristics import compare, updated_belief
from surprise_loop.phases.perceive import build_percept
from surprise_loop.state import (
    AgentState,
    Appraisal,
    Band,
    Decision,
    Episode,
    Expectation,
    Feedback,
    Override,
    PassRecord,
    Pulls,
    Sureness,
    SurpriseNote,
    Workspace,
)
from surprise_loop.world import Action, BodyState, Channel, Event, Levels, OpenQuestion, Place

HEAT, WATER, ENERGY = Channel.HEAT, Channel.WATER, Channel.ENERGY
OPEN = Band(lo=None, hi=None)
OPEN_BANDS = dict.fromkeys(Channel, OPEN)
HOT_READING = Appraisal(reading="I am too hot; I need to cool down.", about=HEAT)
NO_FEEDBACK = Feedback(judged_step=None, score=None, line="")


def ready(
    rt: Runtime,
    action: Action,
    bands: dict[Channel, Band] = OPEN_BANDS,
    *,
    agent: AgentState | None = None,
    world: bool | None = None,
    feedback: Feedback = NO_FEEDBACK,
    appraisal: Appraisal = HOT_READING,
    pass_index: int = 0,
) -> Workspace:
    """A workspace filled up to Reconcile: `action` decided with `bands`, the world stepped."""
    ws = base_ws(rt, agent).model_copy(update={"pass_index": pass_index})
    expectation = Expectation(words=None, bands=bands, world=world, sureness=Sureness.FAIRLY_SURE)
    decision = Decision(action=action, expectation=expectation)
    outcome = rt.world.step(action)
    return ws.model_copy(
        update={
            "percept": build_percept(ws.observation, ws.agent, PARAMS),
            "appraisal": appraisal,
            "decision": decision,
            "outcome": outcome,
            "comparison": compare(decision, outcome.observation, ILO, PARAMS.monitor),
            "feedback": feedback,
        }
    )


def heat_band(lo: float, hi: float) -> dict[Channel, Band]:
    return OPEN_BANDS | {HEAT: Band(lo=lo, hi=hi)}


# --- belief, feeling, attention -----------------------------------------------------------------


def test_belief_feeling_and_attention() -> None:
    rt = make_rt()
    bands = heat_band(0.4, 0.5) | {WATER: Band(lo=0.5, hi=0.55)}
    ws = ready(rt, Action.REST, bands)
    out = run(reconcile, ws, rt)

    signal = ws.outcome.observation.signal
    belief = updated_belief(signal, bands, ILO, PARAMS.belief)
    assert out.agent.belief == belief
    assert out.agent.feeling == feel(belief, ILO, ws.agent.feeling, ws.comparison, PARAMS.feeling)
    assert out.agent.attention is HEAT
    assert out.agent.tick == ws.agent.tick


def test_attention_follows_the_reading_even_when_it_is_about_nothing() -> None:
    rt = make_rt()
    reading = Appraisal(reading="Nothing presses on me just now.", about=None)
    out = run(reconcile, ready(rt, Action.REST, appraisal=reading,
                               agent=agent_with(rt, attention=WATER)), rt)  # fmt: skip
    assert out.agent.attention is None


# --- the voiced surprise ------------------------------------------------------------------------


@pytest.mark.parametrize(("bands", "worse"), [(heat_band(-1.0, -0.9), True),
                                              (heat_band(2.0, 2.1), False)])  # fmt: skip
def test_a_loud_body_error_is_voiced(bands: dict[Channel, Band], worse: bool) -> None:
    rt = make_rt()
    ws = ready(rt, Action.REST, bands)
    assert abs(ws.comparison.body_felt[HEAT]) >= PARAMS.reconcile.voice_threshold
    out = run(reconcile, ws, rt)
    assert out.agent.voiced == SurpriseNote(channel=HEAT, worse=worse)


def test_the_largest_body_error_is_voiced() -> None:
    rt = make_rt()
    ws = ready(rt, Action.REST, heat_band(-1.0, -0.9) | {WATER: Band(lo=0.95, hi=1.0)})
    felt = ws.comparison.body_felt
    loudest = max(Channel, key=lambda c: abs(felt[c]))
    out = run(reconcile, ws, rt)
    assert out.agent.voiced == SurpriseNote(channel=loudest, worse=felt[loudest] > 0)


def test_a_world_mismatch_is_voiced_when_the_body_is_quiet() -> None:
    rt = make_rt()  # 13:00: the log under the oak is in the sun
    ws = ready(rt, Action.GO_OAK, world=True)
    assert ws.comparison.world is True
    out = run(reconcile, ws, rt)
    assert out.agent.voiced == SurpriseNote(question=OpenQuestion.LOG_SHADED, actual=False)


def test_a_loud_body_error_wins_over_a_world_mismatch() -> None:
    rt = make_rt()
    ws = ready(rt, Action.GO_OAK, heat_band(-1.0, -0.9), world=True)
    assert run(reconcile, ws, rt).agent.voiced == SurpriseNote(channel=HEAT, worse=True)


def test_no_surprise_is_none() -> None:
    rt = make_rt()
    ws = ready(rt, Action.GO_OAK, world=False)
    assert ws.comparison.world is False
    assert run(reconcile, ws, rt).agent.voiced is None


def test_a_quiet_body_error_is_not_voiced() -> None:
    rt = make_rt()
    probe = ready(make_rt(), Action.REST).outcome.observation.signal
    # a heat band ending just below the signal: an error under voice_threshold
    small = PARAMS.reconcile.voice_threshold / ILO.tightness[HEAT] / 2
    ws = ready(rt, Action.REST, heat_band(probe.heat - 1, probe.heat - small))
    assert 0 < ws.comparison.body_felt[HEAT] < PARAMS.reconcile.voice_threshold
    assert run(reconcile, ws, rt).agent.voiced is None


# --- memory -------------------------------------------------------------------------------------


def test_one_episode_per_pass() -> None:
    rt = make_rt()
    earlier = [episode(s) for s in range(1, 3)]
    agent = remembering(rt, earlier)
    ws = ready(rt, Action.REST, heat_band(0.4, 0.5), agent=agent)
    out = run(reconcile, ws, rt)

    assert out.agent.memory.episodes[:2] == earlier
    stored = out.agent.memory.episodes[2]
    assert stored.salience == pytest.approx(min(1, 0.2 + 0.3 * ws.comparison.size))
    assert stored == Episode(
        tick=agent.tick + 1,
        step=ws.outcome.observation.step,
        place=Place.MEADOW,
        action=Action.REST,
        event=ws.outcome.observation.event,
        expectation=ws.decision.expectation,
        comparison=ws.comparison,
        need_before=needs(agent.belief, ILO),
        salience=stored.salience,
        scene=ws.percept.scene.text(),
        reading=HOT_READING.reading,
    )


def test_the_episode_place_is_where_the_decision_was_made() -> None:
    rt = make_rt()
    out = run(reconcile, ready(rt, Action.GO_OAK), rt)
    assert out.outcome.observation.place is Place.OAK
    assert out.agent.memory.episodes[-1].place is Place.MEADOW


def test_salience_is_capped_at_one() -> None:
    rt = make_rt()
    ws = ready(rt, Action.REST, heat_band(-1.0, -0.9))
    assert 0.2 + 0.3 * ws.comparison.size > 1
    assert run(reconcile, ws, rt).agent.memory.episodes[-1].salience == 1


# --- the judged episode and its pull ------------------------------------------------------------


def pulls_with(place: Place, action: Action, v: float) -> Pulls:
    table = {p: dict.fromkeys(Action, 0.0) for p in Place}
    table[place][action] = v
    return Pulls(values=table)


def test_the_judged_episode_gets_its_score_and_line_and_its_pull_moves() -> None:
    rt = make_rt()
    eps = [episode(s) for s in range(1, 6)]
    eps[1] = episode(2, Action.GO_OAK, Place.STREAM)  # 5 episodes, window 4: index 1
    agent = remembering(rt, eps).model_copy(
        update={"pulls": pulls_with(Place.STREAM, Action.GO_OAK, 0.2)}
    )
    feedback = Feedback(judged_step=2, score=0.6, line="After going to the oak, it helped.")
    out = run(reconcile, ready(rt, Action.REST, agent=agent, feedback=feedback), rt)

    stored = out.agent.memory.episodes
    assert stored[1] == eps[1].model_copy(update={"score": 0.6, "regulation": feedback.line})
    assert stored[:1] + stored[2:5] == eps[:1] + eps[2:]
    moved = out.agent.pulls.values[Place.STREAM][Action.GO_OAK]
    assert moved == pytest.approx(0.2 + 0.5 * (0.6 - 0.2))
    assert out.agent.pulls == pulls_with(Place.STREAM, Action.GO_OAK, moved)


def test_without_a_judgement_no_pull_moves_and_no_episode_changes() -> None:
    rt = make_rt()
    eps = [episode(s) for s in range(1, 6)]
    agent = remembering(rt, eps).model_copy(
        update={"pulls": pulls_with(Place.MEADOW, Action.REST, 0.3)}
    )
    out = run(reconcile, ready(rt, Action.REST, agent=agent), rt)
    assert out.agent.memory.episodes[:5] == eps
    assert out.agent.pulls == agent.pulls


@pytest.mark.parametrize("event", [Event.STUMBLED, Event.COLLAPSED])
def test_a_stumbled_or_collapsed_episode_and_every_pull_stay_as_they_were(event: Event) -> None:
    rt = make_rt()
    eps = [episode(s) for s in range(1, 6)]
    eps[1] = episode(2, Action.GO_OAK, need_before={HEAT: 2.5, WATER: 0, ENERGY: 0})
    eps[1] = eps[1].model_copy(update={"event": event})  # 5 episodes, window 4: index 1
    agent = remembering(rt, eps).model_copy(
        update={"pulls": pulls_with(Place.MEADOW, Action.GO_OAK, 0.2)}
    )
    ws = ready(rt, Action.REST, agent=agent)
    ws = run(evaluate, ws.model_copy(update={"feedback": None}), rt)
    assert ws.feedback == NO_FEEDBACK
    out = run(reconcile, ws, rt)
    assert out.agent.memory.episodes[:5] == eps
    assert out.agent.memory.episodes[1].score is None
    assert out.agent.pulls == agent.pulls


def test_feedback_for_a_step_that_is_not_window_back_is_refused() -> None:
    rt = make_rt()
    agent = remembering(rt, [episode(s) for s in range(1, 6)])
    feedback = Feedback(judged_step=4, score=0.6, line="x")
    with pytest.raises(ValueError, match="judged"):
        run(reconcile, ready(rt, Action.REST, agent=agent, feedback=feedback), rt)


# --- the override -------------------------------------------------------------------------------


def urgent_heat(rt: Runtime, action: Action = Action.REST, **kw: object) -> Workspace:
    """Ilo, hot in the meadow, rests; heat came out much hotter than the band said."""
    ws = ready(rt, action, heat_band(0.4, 0.5), **kw)
    belief = updated_belief(ws.outcome.observation.signal, heat_band(0.4, 0.5), ILO, PARAMS.belief)
    assert needs(belief, ILO)[HEAT] >= PARAMS.override.urgent_felt
    assert ws.comparison.body_felt[HEAT] >= PARAMS.override.worse_margin
    return ws


def test_override_fires_when_an_urgent_need_came_out_worse_and_was_not_served() -> None:
    rt = make_rt()
    out = run(reconcile, urgent_heat(rt), rt)
    assert out.override is not None
    assert out.override.channel is HEAT
    assert out.override.actions == [Action.GO_OAK]
    assert out.override.detail
    assert out.passes[-1].override == out.override


def test_override_actions_are_the_serving_offers_in_offer_order() -> None:
    # thirsty in the meadow, resting; water came out much lower than the band said
    body = BodyState(heat=0.99, water=0.3, energy=0.22)
    rt = make_rt(belief=Levels(heat=0.7, water=0.2, energy=0.7), body=body)
    bands = OPEN_BANDS | {WATER: Band(lo=0.5, hi=0.6)}
    ws = ready(rt, Action.REST, bands)
    belief = updated_belief(ws.outcome.observation.signal, bands, ILO, PARAMS.belief)
    assert needs(belief, ILO)[WATER] >= PARAMS.override.urgent_felt
    assert ws.comparison.body_felt[WATER] >= PARAMS.override.worse_margin
    out = run(reconcile, ws, rt)
    assert out.override == Override(
        channel=WATER, actions=[Action.GO_STREAM], detail=out.override.detail
    )


def test_no_override_on_pass_one_and_the_pass_zero_override_is_kept() -> None:
    rt = make_rt()
    kept = Override(channel=WATER, actions=[Action.GO_STREAM], detail="from pass zero")
    ws = urgent_heat(rt, pass_index=1).model_copy(update={"override": kept})
    out = run(reconcile, ws, rt)
    assert out.override == kept
    assert out.passes[-1].override is None


def test_no_override_when_disabled() -> None:
    params = PARAMS.model_copy(
        update={"override": PARAMS.override.model_copy(update={"enabled": False})}
    )
    rt = make_rt(params)
    assert run(reconcile, urgent_heat(rt), rt).override is None


def test_no_override_when_the_need_is_not_urgent() -> None:
    params = PARAMS.model_copy(
        update={"override": PARAMS.override.model_copy(update={"urgent_felt": 10.0})}
    )
    rt = make_rt(params)
    assert run(reconcile, ready(rt, Action.REST, heat_band(0.4, 0.5)), rt).override is None


def test_no_override_when_the_need_came_out_as_expected() -> None:
    rt = make_rt()
    assert run(reconcile, ready(rt, Action.REST), rt).override is None


def test_no_override_when_the_action_served_the_channel() -> None:
    rt = make_rt()
    ws = ready(rt, Action.GO_OAK, heat_band(0.0, 0.1))
    assert ws.comparison.body_felt[HEAT] >= PARAMS.override.worse_margin
    assert run(reconcile, ws, rt).override is None


def test_no_override_and_no_crash_when_nothing_offered_serves_the_need() -> None:
    rt = make_rt()
    ws = urgent_heat(rt)
    obs = ws.outcome.observation.model_copy(update={"offers": [Action.REST, Action.LOOK]})
    ws = ws.model_copy(update={"outcome": ws.outcome.model_copy(update={"observation": obs})})
    out = run(reconcile, ws, rt)
    assert out.override is None
    assert out.passes[-1].override is None


# --- the pass record ----------------------------------------------------------------------------


def test_the_pass_record_holds_what_deliberate_saw_and_the_state_after() -> None:
    rt = make_rt()
    other = make_rt()
    earlier = run(reconcile, ready(other, Action.LOOK), other).passes[0]
    ws = urgent_heat(rt).model_copy(update={"passes": (earlier,)})
    out = run(reconcile, ws, rt)

    assert out.passes[0] == earlier
    assert out.passes[1] == PassRecord(
        tick=ws.agent.tick + 1,
        pass_index=0,
        scene=ws.percept.scene,
        offers=[o.action for o in ws.percept.offers],
        decision=ws.decision,
        outcome=ws.outcome,
        comparison=ws.comparison,
        feedback=ws.feedback,
        belief_after=out.agent.belief,
        feeling_after=out.agent.feeling,
        override=out.override,
    )
