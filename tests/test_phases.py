"""Perceive, Appraise, Deliberate, Act, Monitor and Evaluate on stubs, on hand-built workspaces."""

import asyncio
from collections.abc import Awaitable, Callable

import pytest
from bodies import ILO, ILO_AFTERNOON

from surprise_loop.config import (
    AgentSpec,
    Backends,
    BeliefParams,
    ConfigError,
    DeliberateParams,
    EvaluateParams,
    ExpectationParams,
    MemoryParams,
    MonitorParams,
    OverrideParams,
    Params,
    PullParams,
    ReconcileParams,
    RunConfig,
    Runtime,
    StubModelParams,
    initial_agent_state,
)
from surprise_loop.feeling import needs
from surprise_loop.lens import felt_scene
from surprise_loop.phases import act, appraise, deliberate, evaluate, monitor, perceive, reconcile
from surprise_loop.phases.deliberate import STUB_WHY
from surprise_loop.phases.heuristics import (
    bands_from_forecast,
    clamp,
    compare,
    forecast,
    heat_pull,
)
from surprise_loop.phases.perceive import build_percept
from surprise_loop.phrases import OFFER_PHRASES, PULL_PHRASES, REGULATION_LINES, STUB_READINGS
from surprise_loop.state import (
    AgentState,
    Band,
    Comparison,
    Episode,
    Expectation,
    Feedback,
    Memory,
    Offer,
    Pulls,
    Sureness,
    Workspace,
)
from surprise_loop.world import (
    GO_TARGET,
    Action,
    BodyState,
    Channel,
    ClearingWorld,
    Event,
    Levels,
    Place,
    WorldConfig,
    facts_at,
)

Phase = Callable[[Workspace, Runtime], Awaitable[Workspace]]
HEAT, WATER, ENERGY = Channel.HEAT, Channel.WATER, Channel.ENERGY
PHASES: tuple[Phase, ...] = (perceive, appraise, deliberate, act, monitor, evaluate, reconcile)

WORLD = WorldConfig(start_minute=780)
MODEL = StubModelParams(
    heat_rate=0.25,
    heat_per_bearing=0.5,
    water_drain=0.004,
    sweat_water=0.01,
    drink_water=0.08,
    walk_energy=0.03,
    rest_energy=0.02,
)
# pinned here so no assertion depends on the provisional defaults in config.py
PARAMS = Params(
    expectation=ExpectationParams(
        bit=0.25, much=1.0, stub_tolerance={HEAT: 0.05, WATER: 0.03, ENERGY: 0.03}
    ),
    monitor=MonitorParams(world_weight=1.0),
    belief=BeliefParams(expectation_weight={HEAT: 0.5, WATER: 0.5, ENERGY: 0.5}),
    memory=MemoryParams(retrieval_k=4, base_salience=0.2, salience_gain=0.3, retrieval="last_k"),
    evaluate=EvaluateParams(window=4, score_scale=1.0, helped=0.25),
    pulls=PullParams(rate=0.5, speak=0.2),
    override=OverrideParams(enabled=True, urgent_felt=2.0, worse_margin=0.5),
    deliberate=DeliberateParams(stub="reactive", act_threshold=0.5, horizon=6, model=MODEL),
    reconcile=ReconcileParams(voice_threshold=0.5),
)
ILO_BODY = BodyState(heat=0.99, water=0.47, energy=0.22)


def with_stub(stub: str) -> Params:
    return PARAMS.model_copy(
        update={"deliberate": PARAMS.deliberate.model_copy(update={"stub": stub})}
    )


def make_rt(
    params: Params = PARAMS,
    *,
    seed: int = 7,
    place: Place = Place.MEADOW,
    body: BodyState = ILO_BODY,
    belief: Levels = ILO_AFTERNOON,
    backends: Backends | None = None,
) -> Runtime:
    config = RunConfig(
        run_id="test",
        seed=seed,
        ticks=None,
        world=WORLD,
        start_place=place,
        agent=AgentSpec(name="ilo", temperament=ILO, body=body, belief=belief, arousal=0.7),
        params=params,
        backends=backends or Backends(),
        model=None,
    )
    return Runtime(config=config, world=ClearingWorld(WORLD, seed, body, place))


def run(phase: Phase, ws: Workspace, rt: Runtime) -> Workspace:
    return asyncio.run(phase(ws, rt))


def agent_with(rt: Runtime, **update: object) -> AgentState:
    return initial_agent_state(rt.config).model_copy(update=update)


def base_ws(rt: Runtime, agent: AgentState | None = None) -> Workspace:
    return Workspace(agent=agent or agent_with(rt), observation=rt.world.observe())


def through(ws: Workspace, rt: Runtime, n: int) -> Workspace:
    """`ws` after the first `n` phases."""
    for phase in PHASES[:n]:
        ws = run(phase, ws, rt)
    return ws


BAND = Band(lo=None, hi=None)
ANY = Expectation(words=None, bands=dict.fromkeys(Channel, BAND), world=None,
                  sureness=Sureness.GUESSING)  # fmt: skip
NOTHING = Comparison(
    body=dict.fromkeys(Channel, 0.0),
    body_felt=dict.fromkeys(Channel, 0.0),
    world=None,
    better=0.0,
    size=0.0,
)


def episode(
    step: int,
    action: Action = Action.REST,
    place: Place = Place.MEADOW,
    need_before: dict[Channel, float] | None = None,
    score: float | None = None,
) -> Episode:
    return Episode(
        tick=step,
        step=step,
        place=place,
        action=action,
        event=Event.RESTED,
        expectation=ANY,
        comparison=NOTHING,
        need_before=need_before or dict.fromkeys(Channel, 0.0),
        salience=0.2,
        scene="Here.",
        reading="Nothing presses on me just now.",
        score=score,
    )


def remembering(rt: Runtime, episodes: list[Episode]) -> AgentState:
    return agent_with(rt, memory=Memory(episodes=episodes), tick=len(episodes))


# --- purity -------------------------------------------------------------------------------------


@pytest.mark.parametrize("position", range(len(PHASES)), ids=[p.__name__ for p in PHASES])
def test_phase_returns_a_new_workspace_and_leaves_its_input_equal(position: int) -> None:
    rt = make_rt()
    episodes = [episode(s, Action.GO_OAK, need_before={HEAT: 2.5, WATER: 0, ENERGY: 0})
                for s in range(1, 6)]  # fmt: skip
    ws = through(base_ws(rt, remembering(rt, episodes)), rt, position)
    before = ws.model_copy(deep=True)

    result = run(PHASES[position], ws, rt)

    assert isinstance(result, Workspace)
    assert result is not ws
    assert result != ws
    assert ws == before


@pytest.mark.parametrize("position", range(len(PHASES) - 1), ids=[p.__name__ for p in PHASES[:-1]])
def test_only_reconcile_touches_the_agent(position: int) -> None:
    rt = make_rt()
    ws = through(base_ws(rt), rt, position)
    assert run(PHASES[position], ws, rt).agent == ws.agent


# --- Perceive -----------------------------------------------------------------------------------


def test_percept_offers_carry_their_phrases() -> None:
    rt = make_rt()
    ws = base_ws(rt)
    percept = build_percept(ws.observation, ws.agent, PARAMS)
    assert percept.offers == [
        Offer(action=a, phrase=OFFER_PHRASES[(Place.MEADOW, a)]) for a in ws.observation.offers
    ]


def test_percept_scene_is_the_lens_on_the_belief() -> None:
    rt = make_rt()
    agent = agent_with(rt, attention=WATER)
    obs = rt.world.observe()
    percept = build_percept(obs, agent, PARAMS)
    assert percept.scene == felt_scene(
        obs.place, obs.facts, agent.belief, ILO, agent.feeling.arousal, WATER, agent.voiced, 0,
        PARAMS.lens,
    )  # fmt: skip


def test_after_looking_one_more_fact_is_noticed() -> None:
    rt = make_rt()
    agent = agent_with(rt)
    looked = rt.world.step(Action.LOOK).observation
    assert looked.event is Event.LOOKED
    percept = build_percept(looked, agent, PARAMS)
    assert percept.scene == felt_scene(
        looked.place, looked.facts, agent.belief, ILO, agent.feeling.arousal, None, agent.voiced,
        1, PARAMS.lens,
    )  # fmt: skip


def pulled(rt: Runtime, values: dict[tuple[Place, Action], float]) -> AgentState:
    table = {place: dict.fromkeys(Action, 0.0) for place in Place}
    for (place, action), v in values.items():
        table[place][action] = v
    return agent_with(rt, pulls=Pulls(values=table))


@pytest.mark.parametrize(
    ("values", "line"),
    [
        ({}, ""),
        ({(Place.MEADOW, Action.GO_OAK): 0.19}, ""),
        ({(Place.MEADOW, Action.GO_OAK): 0.3}, PULL_PHRASES[(Action.GO_OAK, True)]),
        (
            {(Place.MEADOW, Action.GO_OAK): 0.3, (Place.MEADOW, Action.REST): -0.5},
            PULL_PHRASES[(Action.REST, False)],
        ),
        # a pull at another place, or for an action not offered here, is not spoken
        ({(Place.OAK, Action.REST): 0.9, (Place.MEADOW, Action.GO_MEADOW): 0.9}, ""),
    ],
    ids=["none", "below speak", "leans", "largest magnitude", "elsewhere"],
)
def test_pull_line(values: dict[tuple[Place, Action], float], line: str) -> None:
    rt = make_rt()
    percept = build_percept(rt.world.observe(), pulled(rt, values), PARAMS)
    assert percept.pull_line == line


def test_with_given_offers_only_those_appear() -> None:
    rt = make_rt()
    agent = pulled(rt, {(Place.MEADOW, Action.GO_OAK): 0.9, (Place.MEADOW, Action.REST): 0.3})
    percept = build_percept(rt.world.observe(), agent, PARAMS, offers=[Action.REST])
    assert percept.offers == [
        Offer(action=Action.REST, phrase=OFFER_PHRASES[(Place.MEADOW, Action.REST)])
    ]
    assert percept.pull_line == PULL_PHRASES[(Action.REST, True)]


def test_recall_follows_the_configured_retrieval() -> None:
    rt = make_rt()
    episodes = [
        episode(s).model_copy(update={"salience": v})
        for s, v in enumerate([0.9, 0.2, 0.3, 0.1, 0.4, 0.2], start=1)
    ]
    agent = remembering(rt, episodes)
    obs = rt.world.observe()
    last = build_percept(obs, agent, PARAMS)
    memory = PARAMS.memory.model_copy(update={"retrieval": "salient_k"})
    salient = build_percept(obs, agent, PARAMS.model_copy(update={"memory": memory}))
    assert last.recalled == episodes[2:]
    assert salient.recalled == agent.memory.salient_k(4)


def test_perceive_sets_the_percept() -> None:
    rt = make_rt()
    ws = base_ws(rt)
    assert run(perceive, ws, rt).percept == build_percept(ws.observation, ws.agent, PARAMS)


# --- Appraise -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("belief", "about", "too_much"),
    [
        (ILO_AFTERNOON, ENERGY, True),  # needs: heat 2.52, water 1.32, energy 2.7
        (Levels(heat=-0.5, water=0.8, energy=0.7), HEAT, False),
        (Levels(heat=0.4, water=0.8, energy=0.7), HEAT, True),
        (Levels(heat=-0.1, water=0.6, energy=0.7), WATER, True),
        (Levels(heat=-0.1, water=0.8, energy=0.7), None, True),
        (Levels(heat=0.0, water=1.0, energy=0.9), None, True),  # over-full is no need
    ],
    ids=["ilo afternoon", "cold", "hot", "thirsty", "at set point", "over-full"],
)
def test_appraise_stub_reads_the_loudest_need(
    belief: Levels, about: Channel | None, too_much: bool
) -> None:
    rt = make_rt(belief=belief)
    ws = run(appraise, through(base_ws(rt), rt, 1), rt)
    assert ws.appraisal is not None
    assert ws.appraisal.about is about
    assert ws.appraisal.reading == STUB_READINGS[(about, too_much)]


@pytest.mark.parametrize("phase", [appraise, deliberate], ids=["appraise", "deliberate"])
def test_a_model_backend_is_not_built_yet(phase: Phase) -> None:
    backends = Backends(**{phase.__name__: "llm"})
    rt = make_rt(backends=backends)
    ws = through(base_ws(rt), rt, PHASES.index(phase))
    with pytest.raises(ConfigError, match=f"{phase.__name__} is not built for a model yet"):
        run(phase, ws, rt)


# --- Deliberate ---------------------------------------------------------------------------------

STUBS = ["reactive", "lookahead", "blind"]


@pytest.mark.parametrize("stub", STUBS)
def test_every_stub_chooses_an_offered_action_and_commits_a_forecast_band(stub: str) -> None:
    rt = make_rt(with_stub(stub))
    ws = through(base_ws(rt), rt, 3)
    decision = ws.decision
    assert decision is not None
    assert decision.action in ws.observation.offers
    assert decision.why == STUB_WHY
    e = decision.expectation
    assert e.words is None and e.world is None and e.sureness is Sureness.FAIRLY_SURE
    assert set(e.bands) == set(Channel)
    place = GO_TARGET.get(decision.action)
    pull = heat_pull(ws.observation.facts, place)
    want = forecast(ws.agent.belief, pull, decision.action, MODEL)
    assert e.bands == bands_from_forecast(want, PARAMS.expectation)


@pytest.mark.parametrize("stub", STUBS)
def test_every_stub_keeps_to_restricted_offers(stub: str) -> None:
    rt = make_rt(with_stub(stub))
    ws = base_ws(rt)
    percept = build_percept(ws.observation, ws.agent, PARAMS, offers=[Action.GO_STREAM])
    ws = ws.model_copy(update={"percept": percept})
    ws = run(appraise, ws, rt)
    assert run(deliberate, ws, rt).decision.action is Action.GO_STREAM


def test_reactive_goes_to_the_oak_in_the_hot_meadow() -> None:
    rt = make_rt(belief=Levels(heat=0.74, water=0.8, energy=0.7))
    assert through(base_ws(rt), rt, 3).decision.action is Action.GO_OAK


def test_blind_is_the_same_for_the_same_seed_tick_and_pass() -> None:
    def choice(seed: int, tick: int, pass_index: int) -> Action:
        rt = make_rt(with_stub("blind"), seed=seed)
        ws = base_ws(rt, agent_with(rt, tick=tick)).model_copy(update={"pass_index": pass_index})
        return through(ws, rt, 3).decision.action

    assert choice(3, 5, 0) == choice(3, 5, 0)
    assert choice(3, 5, 1) == choice(3, 5, 1)
    assert len({choice(seed, 0, 0) for seed in range(20)}) > 1


def at_stream(stub: str) -> Workspace:
    # mild thirst (need 0.36, below act_threshold) and heat need 0.6 that the stream's own pull
    # would raise further: reactive leaves for the shade it notices
    belief = Levels(heat=0.1, water=0.74, energy=0.7)
    rt = make_rt(with_stub(stub), place=Place.STREAM, belief=belief)
    agent = agent_with(rt, feeling=initial_agent_state(rt.config).feeling.model_copy(
        update={"arousal": 0.3}))  # fmt: skip
    obs = rt.world.observe().model_copy(
        update={"facts": facts_at(WORLD, Place.STREAM, 780, True, False)}
    )
    return through(Workspace(agent=agent, observation=obs), rt, 3)


def test_lookahead_at_the_stream_drinks_where_reactive_leaves() -> None:
    reactive, lookahead = at_stream("reactive"), at_stream("lookahead")
    assert reactive.decision.action is Action.GO_OAK
    assert lookahead.decision.action is Action.DRINK


# --- Act and Monitor ----------------------------------------------------------------------------


def test_act_steps_the_world_with_the_decided_action() -> None:
    rt = make_rt()
    ws = through(base_ws(rt), rt, 4)
    assert ws.outcome is not None
    assert ws.outcome.observation.last_action is ws.decision.action
    assert rt.world.observe() == ws.outcome.observation


def test_monitor_compares_the_decision_with_what_arrived() -> None:
    rt = make_rt()
    ws = through(base_ws(rt), rt, 5)
    assert ws.comparison == compare(ws.decision, ws.outcome.observation, ILO, PARAMS.monitor)


# --- Evaluate -----------------------------------------------------------------------------------


def evaluated(episodes: list[Episode]) -> tuple[Workspace, Workspace]:
    """The workspace before and after Evaluate, after one LOOK in the meadow."""
    rt = make_rt()
    ws = base_ws(rt, remembering(rt, episodes))
    ws = through(ws, rt, 2)
    decision = run(deliberate, ws, rt).decision.model_copy(update={"action": Action.LOOK})
    ws = ws.model_copy(update={"decision": decision})
    ws = run(monitor, run(act, ws, rt), rt)
    return ws, run(evaluate, ws, rt)


def heat_now() -> float:
    rt = make_rt()
    signal = rt.world.step(Action.LOOK).observation.signal
    return abs(needs(signal, ILO)[HEAT])


def test_with_fewer_than_window_episodes_nothing_is_judged() -> None:
    eps = [episode(s, Action.GO_OAK, need_before={HEAT: 2.5, WATER: 0, ENERGY: 0})
           for s in range(1, 4)]  # fmt: skip
    _, after = evaluated(eps)
    assert after.feedback == Feedback(judged_step=None, score=None, line="")


@pytest.mark.parametrize(("delta", "sign"), [(0.5, 1), (0.1, 0), (-0.5, -1), (3.0, 1), (-3.0, -1)])
def test_the_episode_window_back_is_judged_on_its_served_channel(delta: float, sign: int) -> None:
    now = heat_now()
    before = max(0.0, now + delta)
    eps = [episode(s, Action.REST) for s in range(1, 7)]
    eps[2] = episode(3, Action.GO_OAK, need_before={HEAT: before, WATER: 0.0, ENERGY: 0.0})
    _, after = evaluated(eps)  # 6 episodes, window 4: index 2
    score = clamp((before - now) / PARAMS.evaluate.score_scale, -1, 1)
    assert after.feedback.judged_step == 3
    assert after.feedback.score == pytest.approx(score)
    assert after.feedback.line == REGULATION_LINES[(Action.GO_OAK, sign)]


def test_a_negative_heat_need_is_judged_by_its_magnitude() -> None:
    now = heat_now()
    eps = [episode(1, Action.GO_MEADOW, need_before={HEAT: -(now + 0.5), WATER: 0, ENERGY: 0}),
           *[episode(s) for s in range(2, 5)]]  # fmt: skip
    _, after = evaluated(eps)
    assert after.feedback.judged_step == 1
    assert after.feedback.score == pytest.approx(0.5)


@pytest.mark.parametrize(
    "first",
    [
        episode(1, Action.GO_OAK, need_before={HEAT: 2.5, WATER: 0, ENERGY: 0}, score=0.3),
        episode(1, Action.LOOK, need_before={HEAT: 2.5, WATER: 0, ENERGY: 0}),
    ],
    ids=["already judged", "look"],
)
def test_an_already_judged_episode_or_a_look_is_not_judged(first: Episode) -> None:
    _, after = evaluated([first, *[episode(s) for s in range(2, 5)]])
    assert after.feedback == Feedback(judged_step=None, score=None, line="")


def test_evaluate_changes_only_the_feedback() -> None:
    eps = [episode(s, Action.REST, need_before={HEAT: 0, WATER: 0, ENERGY: 2.0})
           for s in range(1, 6)]  # fmt: skip
    before, after = evaluated(eps)
    assert after.feedback is not None and after.feedback.judged_step == 2
    assert after == before.model_copy(update={"feedback": after.feedback})
    assert after.agent == before.agent
