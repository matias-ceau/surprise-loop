import math
import random
from pathlib import Path
from typing import Any

import pytest
from bodies import (
    HOTTER,
    ILO,
    ILO_AFTERNOON,
    ILO_AFTERNOON_AROUSAL,
    ILO_TEXT,
    TAVI,
    TAVI_AFTERNOON,
    TAVI_AFTERNOON_AROUSAL,
    TAVI_TEXT,
)
from pydantic import BaseModel, ValidationError

from surprise_loop.config import (
    AgentSpec,
    Backends,
    BeliefParams,
    ConfigError,
    DeliberateParams,
    EvaluateParams,
    ExpectationParams,
    ExperimentConfig,
    FeelingParams,
    LensParams,
    LLMParams,
    MemoryParams,
    MetricsParams,
    MonitorParams,
    OverrideParams,
    Params,
    PullParams,
    ReconcileParams,
    RunConfig,
    Runtime,
    StubModelParams,
    derive_rng,
    initial_agent_state,
    load_config,
    resolve_model,
)
from surprise_loop.feeling import feel
from surprise_loop.lens import felt_scene
from surprise_loop.state import AgentState, Feeling, SurpriseNote
from surprise_loop.world import (
    Action,
    BodyState,
    Channel,
    Levels,
    Observation,
    Outcome,
    Place,
    WorldConfig,
    WorldState,
    frozen_noon_facts,
)

ROOT = Path(__file__).resolve().parents[1]
AFTERNOON = ROOT / "experiments" / "clearing_afternoon.yaml"
DAY = ROOT / "experiments" / "clearing_day.yaml"

YAML = """\
name: pilot
seeds: [1, 2]
ticks: 50
world: {start_minute: 780}
start_place: oak
agents:
  - name: ilo
    temperament:
      setpoint: {heat: -0.1, water: 0.8, energy: 0.7}
      tightness: {heat: 3.0, water: 6.0, energy: 6.0}
      trust: {heat: 0.5, water: 0.3, energy: 0.5}
      habit: {burden: 0.6, relief: 0.2, ambient: 0.1}
      baseline: -0.1
    body: {heat: 0.5, water: 0.6, energy: 0.5}
    belief: {heat: 0.5, water: 0.6, energy: 0.5}
  - name: tavi
    temperament:
      setpoint: {heat: 0.2, water: 0.8, energy: 0.7}
      tightness: {heat: 2.5, water: 4.0, energy: 4.0}
      trust: {heat: 0.5, water: 0.5, energy: 0.5}
      habit: {burden: 0.1, relief: 0.1, ambient: 0.6}
      baseline: 0.1
    body: {heat: 0.0, water: 0.8, energy: 0.7}
    belief: {heat: 0.0, water: 0.8, energy: 0.7}
    arousal: 0.4
    voiced: {question: stream_full, actual: false}
params:
  memory: {retrieval_k: 6}
  deliberate: {stub: lookahead, model: {rest_energy: 0.05}}
"""

LLM = {"appraise": "stub", "deliberate": "llm"}


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "exp.yaml"
    path.write_text(text)
    return path


def spec(**changes: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": "a",
        "temperament": ILO.model_dump(),
        "body": {"heat": 0.99, "water": 0.47, "energy": 0.22},
        "belief": ILO_AFTERNOON.model_dump(),
        "arousal": 0.7,
        "voiced": HOTTER.model_dump(),
    }
    return data | changes


def experiment(**overrides: Any) -> ExperimentConfig:
    data: dict[str, Any] = {"name": "exp", "seeds": [3], "agents": [spec()]}
    data.update(overrides)
    return ExperimentConfig.model_validate(data)


def run_config(**overrides: Any) -> RunConfig:
    return experiment(**overrides).runs()[0]


# --- loading ---------------------------------------------------------------------------------


def test_yaml_loads_and_runs_are_agents_times_seeds(tmp_path: Path) -> None:
    config = load_config(write(tmp_path, YAML))
    assert isinstance(config, ExperimentConfig)
    assert config.world == WorldConfig(start_minute=780)
    assert config.start_place is Place.OAK
    assert config.params.memory.retrieval_k == 6
    assert config.params.memory.base_salience == MemoryParams().base_salience
    assert config.params.deliberate.stub == "lookahead"
    assert config.params.deliberate.model.rest_energy == 0.05
    assert config.params.deliberate.model.walk_energy == StubModelParams().walk_energy
    assert config.agents[0].arousal == 0.2
    assert config.agents[0].voiced is None
    assert config.agents[1].voiced == SurpriseNote(question="stream_full", actual=False)
    assert config.backends == Backends(appraise="stub", deliberate="stub")
    assert config.model is None

    runs = config.runs()
    assert [(r.run_id, r.agent.name, r.seed) for r in runs] == [
        ("pilot-ilo-s1", "ilo", 1),
        ("pilot-ilo-s2", "ilo", 2),
        ("pilot-tavi-s1", "tavi", 1),
        ("pilot-tavi-s2", "tavi", 2),
    ]
    for run in runs:
        assert isinstance(run, RunConfig)
        assert run.ticks == 50
        assert run.world == config.world
        assert run.start_place is Place.OAK
        assert run.params == config.params
        assert run.backends == config.backends
        assert run.model is None


def test_world_and_start_place_default() -> None:
    config = experiment()
    assert config.world == WorldConfig()
    assert config.start_place is Place.MEADOW
    assert config.ticks is None


def test_unknown_yaml_key_raises_config_error_naming_the_key_path(tmp_path: Path) -> None:
    path = write(tmp_path, YAML.replace("retrieval_k: 6", "retreival_k: 6"))
    with pytest.raises(ConfigError, match=r"params\.memory\.retreival_k") as excinfo:
        load_config(path)
    assert "not permitted" in str(excinfo.value)


def test_unknown_top_level_key_raises_config_error_naming_it(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"start_plaec"):
        load_config(write(tmp_path, YAML.replace("start_place:", "start_plaec:")))


def test_validation_errors_are_one_line_per_problem(tmp_path: Path) -> None:
    text = YAML.replace("retrieval_k: 6", "retrieval_k: 0, typo: 1")
    with pytest.raises(ConfigError) as excinfo:
        load_config(write(tmp_path, text))
    lines = str(excinfo.value).splitlines()
    assert any(line.strip().startswith("params.memory.retrieval_k:") for line in lines)
    assert any(line.strip().startswith("params.memory.typo:") for line in lines)
    assert len(lines) == 3  # header + one line per problem
    assert "Traceback" not in str(excinfo.value)


def test_list_indices_appear_in_the_key_path(tmp_path: Path) -> None:
    text = YAML.replace("arousal: 0.4", "arousal: 1.5")
    with pytest.raises(ConfigError, match=r"agents\.1\.arousal"):
        load_config(write(tmp_path, text))


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_yaml_syntax_error_raises_one_line_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_config(write(tmp_path, "name: [unclosed\nseeds: 1\n"))
    assert len(str(excinfo.value).splitlines()) == 1


@pytest.mark.parametrize("text", ["", "- 1\n- 2\n", "just a string\n"])
def test_non_mapping_document_raises_config_error(tmp_path: Path, text: str) -> None:
    with pytest.raises(ConfigError, match="mapping"):
        load_config(write(tmp_path, text))


def test_config_file_that_is_not_utf8_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "exp.yaml"
    path.write_bytes(b"name: \xff\xfe\n")
    with pytest.raises(ConfigError, match="not UTF-8"):
        load_config(path)


def test_config_error_is_a_value_error() -> None:
    assert issubclass(ConfigError, ValueError)


# --- the two experiment files ----------------------------------------------------------------


@pytest.mark.parametrize("path", [AFTERNOON, DAY], ids=["afternoon", "day"])
def test_experiment_files_load_with_two_all_stub_runs(path: Path) -> None:
    config = load_config(path)
    assert config.seeds == [1]
    assert config.ticks is None
    assert config.start_place is Place.MEADOW
    assert config.backends == Backends()
    assert config.model is None
    assert config.params == Params()
    assert [a.name for a in config.agents] == ["ilo", "tavi"]
    assert [a.temperament for a in config.agents] == [ILO, TAVI]
    runs = config.runs()
    assert [r.run_id for r in runs] == [f"{config.name}-ilo-s1", f"{config.name}-tavi-s1"]
    assert resolve_model(runs[0]) is None


def test_afternoon_file_is_the_worked_example() -> None:
    config = load_config(AFTERNOON)
    assert config.name == "clearing_afternoon"
    assert config.world == WorldConfig(start_minute=780)
    ilo, tavi = config.agents
    assert ilo.body == BodyState(heat=0.99, water=0.47, energy=0.22)
    assert ilo.belief == ILO_AFTERNOON
    assert ilo.arousal == ILO_AFTERNOON_AROUSAL
    assert ilo.voiced == HOTTER
    assert tavi.body == BodyState(heat=0.02, water=0.84, energy=0.76)
    assert tavi.belief == TAVI_AFTERNOON
    assert tavi.arousal == TAVI_AFTERNOON_AROUSAL
    assert tavi.voiced is None


def test_day_file_starts_at_the_world_default_with_belief_equal_to_body() -> None:
    config = load_config(DAY)
    assert config.name == "clearing_day"
    assert config.world == WorldConfig()
    ilo, tavi = config.agents
    assert ilo.belief == Levels(heat=-0.1, water=0.60, energy=0.50)
    assert ilo.arousal == 0.3
    assert tavi.belief == Levels(heat=0.0, water=0.85, energy=0.80)
    assert tavi.arousal == 0.2
    for agent in config.agents:
        assert agent.body.model_dump(exclude={"gut"}) == agent.belief.model_dump()
        assert agent.body.gut == 0
        assert agent.voiced is None


@pytest.mark.parametrize(("index", "text"), [(0, ILO_TEXT), (1, TAVI_TEXT)], ids=["ilo", "tavi"])
def test_afternoon_initial_states_give_the_fixed_texts(index: int, text: str) -> None:
    run = load_config(AFTERNOON).runs()[index]
    s = initial_agent_state(run)
    scene = felt_scene(
        run.start_place,
        frozen_noon_facts(),
        s.belief,
        s.temperament,
        s.feeling.arousal,
        s.attention,
        s.voiced,
        0,
        run.params.lens,
    )
    assert scene.text() == text


# --- validation ------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["a/b", "", "has space", "../x"])
def test_names_are_restricted_to_safe_characters(name: str) -> None:
    with pytest.raises(ValidationError):
        experiment(name=name)
    with pytest.raises(ValidationError):
        AgentSpec.model_validate(spec(name=name))


def test_experiment_needs_seeds_and_agents() -> None:
    with pytest.raises(ValidationError):
        experiment(seeds=[])
    with pytest.raises(ValidationError):
        experiment(agents=[])


def test_seeds_must_be_unique() -> None:
    with pytest.raises(ValidationError, match="unique"):
        experiment(seeds=[1, 2, 1])


def test_agent_names_must_be_unique() -> None:
    with pytest.raises(ValidationError, match="unique"):
        experiment(agents=[spec(), spec()])


def test_ticks_must_be_positive_when_set() -> None:
    assert experiment(ticks=1).ticks == 1
    with pytest.raises(ValidationError):
        experiment(ticks=0)


def test_backends_are_appraise_and_deliberate_only() -> None:
    assert set(Backends.model_fields) == {"appraise", "deliberate"}
    with pytest.raises(ValidationError):
        Backends.model_validate({"evaluate": "stub"})
    with pytest.raises(ValidationError):
        Backends.model_validate({"deliberate": "model"})


def test_agent_spec_defaults() -> None:
    data = spec()
    del data["arousal"], data["voiced"]
    agent = AgentSpec.model_validate(data)
    assert agent.arousal == 0.2
    assert agent.voiced is None


def test_params_defaults_are_the_contract_values() -> None:
    p = Params()
    assert p.lens == LensParams(
        gain_base=0.5,
        gain_arousal=1.5,
        min_noticed=3,
        drop_per_arousal=4.0,
        attention_bonus=0.3,
        narrow_arousal=0.5,
        body_lines_narrow=2,
        body_lines_wide=3,
        body_speak_grade=2,
        value_edges=(0.25, 1.0, 2.0, 3.0),
        body_edges=(0.5, 1.0, 2.0, 3.0),
    )
    assert p.feeling == FeelingParams(
        tone_gain=0.12,
        phasic_gain=0.3,
        phasic_decay=0.5,
        surprise_gain=0.5,
        strain_gain=0.6,
        strain_full=3.0,
        arousal_inertia=0.5,
        mood_inertia=0.9,
    )
    assert p.expectation == ExpectationParams(
        bit=0.25,
        much=1.0,
        stub_tolerance={Channel.HEAT: 0.05, Channel.WATER: 0.03, Channel.ENERGY: 0.03},
    )
    assert p.monitor == MonitorParams(world_weight=1.0)
    assert p.belief == BeliefParams(
        expectation_weight={Channel.HEAT: 0.5, Channel.WATER: 0.5, Channel.ENERGY: 0.5}
    )
    assert p.memory == MemoryParams(
        retrieval_k=4, base_salience=0.2, salience_gain=0.3, retrieval="last_k"
    )
    assert p.evaluate == EvaluateParams(window=4, score_scale=1.0, helped=0.25)
    assert p.pulls == PullParams(rate=0.5, speak=0.2)
    assert p.override == OverrideParams(enabled=True, urgent_felt=2.0, worse_margin=0.5)
    assert p.deliberate == DeliberateParams(
        stub="reactive",
        act_threshold=0.5,
        horizon=6,
        model=StubModelParams(
            heat_rate=0.25,
            heat_per_bearing=0.5,
            water_drain=0.004,
            sweat_water=0.01,
            drink_water=0.08,
            walk_energy=0.03,
            rest_energy=0.02,
        ),
    )
    assert p.reconcile == ReconcileParams(voice_threshold=0.5)
    assert p.metrics == MetricsParams(comfort_felt=1.0)
    assert p.llm == LLMParams(output_retries=3, temperature=1.0, timeout_s=60.0)


# (params group, field, out-of-range values); NaN and ±inf are added below
BAD_PARAMS: list[tuple[type[BaseModel], str, list[Any]]] = [
    (LensParams, "gain_base", [-0.1]),
    (LensParams, "gain_arousal", [-0.1]),
    (LensParams, "min_noticed", [0]),
    (LensParams, "drop_per_arousal", [-0.1]),
    (LensParams, "attention_bonus", [-0.1]),
    (LensParams, "narrow_arousal", [-0.1, 1.1]),
    (LensParams, "body_lines_narrow", [-1]),
    (LensParams, "body_lines_wide", [-1]),
    (LensParams, "body_speak_grade", [0]),
    (FeelingParams, "tone_gain", [-0.1]),
    (FeelingParams, "phasic_gain", [-0.1]),
    (FeelingParams, "phasic_decay", [-0.1, 1.1]),
    (FeelingParams, "surprise_gain", [-0.1]),
    (FeelingParams, "strain_gain", [-0.1]),
    (FeelingParams, "strain_full", [0.0]),
    (FeelingParams, "arousal_inertia", [-0.1, 1.1]),
    (FeelingParams, "mood_inertia", [-0.1, 1.1]),
    (ExpectationParams, "bit", [0.0, -0.1]),
    (ExpectationParams, "much", [0.0, -0.1]),
    (MonitorParams, "world_weight", [-0.1]),
    (StubModelParams, "heat_rate", [-0.1, 1.1]),
    (StubModelParams, "heat_per_bearing", []),
    (StubModelParams, "water_drain", [-0.1]),
    (StubModelParams, "sweat_water", [-0.1]),
    (StubModelParams, "drink_water", [-0.1]),
    (StubModelParams, "walk_energy", [-0.1]),
    (StubModelParams, "rest_energy", [-0.1]),
    (MemoryParams, "retrieval_k", [0]),
    (MemoryParams, "base_salience", [-0.1, 1.1]),
    (MemoryParams, "salience_gain", [-0.1]),
    (EvaluateParams, "window", [0]),
    (EvaluateParams, "score_scale", [0.0, -1.0]),
    (EvaluateParams, "helped", [-0.1, 1.1]),
    (PullParams, "rate", [-0.1, 1.1]),
    (PullParams, "speak", [-0.1]),
    (OverrideParams, "urgent_felt", [-0.1]),
    (OverrideParams, "worse_margin", [-0.1]),
    (DeliberateParams, "act_threshold", [-0.1]),
    (DeliberateParams, "horizon", [0]),
    (ReconcileParams, "voice_threshold", [-0.1]),
    (MetricsParams, "comfort_felt", [-0.1]),
    (LLMParams, "output_retries", [-1]),
    (LLMParams, "temperature", [-0.1]),
    (LLMParams, "timeout_s", [0.0]),
]

PARAM_CASES = [
    pytest.param(model, field, bad, id=f"{model.__name__}.{field}={bad}")
    for model, field, bads in BAD_PARAMS
    for bad in [*bads, math.nan, math.inf, -math.inf]
]


@pytest.mark.parametrize(("model", "field", "bad"), PARAM_CASES)
def test_params_reject_out_of_range_nan_and_inf(
    model: type[BaseModel], field: str, bad: Any
) -> None:
    with pytest.raises(ValidationError):
        model.model_validate({field: bad})


@pytest.mark.parametrize(
    ("model", "field"),
    [
        (ExpectationParams, "stub_tolerance"),
        (BeliefParams, "expectation_weight"),
    ],
)
def test_per_channel_params_cover_every_channel_and_reject_bad_values(
    model: type[BaseModel], field: str
) -> None:
    full = model().model_dump()[field]
    with pytest.raises(ValidationError, match="water"):
        model.model_validate({field: {k: v for k, v in full.items() if k != Channel.WATER}})
    for bad in (-0.1, math.nan, math.inf):
        with pytest.raises(ValidationError):
            model.model_validate({field: full | {Channel.HEAT: bad}})


def test_expectation_weight_must_be_positive() -> None:
    # k = trust / (trust + weight) would divide by zero for a body with no trust
    weights = {c: 0.5 for c in Channel} | {Channel.ENERGY: 0.0}
    with pytest.raises(ValidationError, match="energy"):
        BeliefParams(expectation_weight=weights)


def test_bit_must_not_exceed_much() -> None:
    ExpectationParams(bit=1.0, much=1.0)
    with pytest.raises(ValidationError, match="bit"):
        ExpectationParams(bit=1.5, much=1.0)


@pytest.mark.parametrize("field", ["value_edges", "body_edges"])
@pytest.mark.parametrize(
    "edges",
    [(), (1.0, 0.5), (0.5, 0.5), (0.0, 1.0), (-0.5, 1.0), (0.5, math.nan), (0.5, math.inf)],
    ids=["empty", "descending", "repeated", "zero", "negative", "nan", "inf"],
)
def test_lens_edges_are_nonempty_strictly_ascending_and_positive(
    field: str, edges: tuple[float, ...]
) -> None:
    with pytest.raises(ValidationError, match=field):
        LensParams.model_validate({field: edges})


def test_lens_edges_accept_a_single_edge() -> None:
    assert LensParams(value_edges=(0.3,), body_edges=(1.0,)).value_edges == (0.3,)


def test_unknown_params_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Params.model_validate({"memory": {"retreival_k": 3}})
    with pytest.raises(ValidationError):
        Params.model_validate({"habits": {}})


# --- resolve_model ---------------------------------------------------------------------------


def test_resolve_model_is_none_when_every_backend_is_a_stub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SURPRISE_LOOP_MODEL", "env-model")
    assert resolve_model(run_config(model="config-model")) is None


@pytest.mark.parametrize("backends", [LLM, {"appraise": "llm", "deliberate": "stub"}])
def test_resolve_model_prefers_config_over_env(
    monkeypatch: pytest.MonkeyPatch, backends: dict[str, str]
) -> None:
    monkeypatch.setenv("SURPRISE_LOOP_MODEL", "env-model")
    assert resolve_model(run_config(backends=backends, model="config-model")) == "config-model"


def test_resolve_model_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SURPRISE_LOOP_MODEL", "env-model")
    assert resolve_model(run_config(backends=LLM)) == "env-model"


@pytest.mark.parametrize("env", [None, ""])
def test_resolve_model_raises_for_llm_backend_without_model(
    monkeypatch: pytest.MonkeyPatch, env: str | None
) -> None:
    if env is None:
        monkeypatch.delenv("SURPRISE_LOOP_MODEL", raising=False)
    else:
        monkeypatch.setenv("SURPRISE_LOOP_MODEL", env)
    with pytest.raises(ConfigError, match="SURPRISE_LOOP_MODEL"):
        resolve_model(run_config(backends=LLM))


# --- initial state, rng, runtime -------------------------------------------------------------


def test_initial_agent_state_comes_from_the_spec() -> None:
    config = run_config()
    s = initial_agent_state(config)
    assert isinstance(s, AgentState)
    assert s.temperament == ILO
    assert s.belief == ILO_AFTERNOON
    assert s.feeling.arousal == 0.7
    assert s.voiced == HOTTER
    assert s.attention is None
    assert s.memory.episodes == []
    assert s.pulls.values == {p: {a: 0.0 for a in Action} for p in Place}
    assert s.tick == 0


def test_initial_feeling_is_feel_from_baseline_with_the_spec_arousal() -> None:
    config = run_config()
    start = Feeling(discomfort=0, valence=ILO.baseline, arousal=0.7, mood=ILO.baseline, phasic=0)
    expected = feel(ILO_AFTERNOON, ILO, start, None, config.params.feeling)
    s = initial_agent_state(config)
    assert s.feeling == expected.model_copy(update={"arousal": 0.7})
    assert s.feeling.discomfort > 0
    assert s.feeling.valence < ILO.baseline


def test_initial_agent_state_without_voiced_note() -> None:
    config = run_config(agents=[spec(voiced=None)])
    assert initial_agent_state(config).voiced is None


def test_derive_rng_is_stable_and_keyed() -> None:
    assert derive_rng(7, 3, 0).random() == 0.816605007653763
    assert derive_rng(7, 3, 0).random() != derive_rng(7, 3, 1).random()
    assert derive_rng(7, "a").random() == random.Random("7:a").random()


class StillWorld:
    """A stand-in for the World protocol; the clearing world comes in a later task."""

    def __init__(self, state: WorldState) -> None:
        self._state = state

    @property
    def done(self) -> bool:
        return True

    def observe(self) -> Observation:
        return self._state.last_observation

    def step(self, action: Action) -> Outcome:
        raise NotImplementedError

    def snapshot(self) -> WorldState:
        return self._state


def test_runtime_is_a_frozen_dataclass() -> None:
    config = run_config()
    observation = Observation(
        step=0,
        minute=360,
        place=Place.MEADOW,
        last_action=None,
        event="start",
        facts=[],
        signal=Levels(heat=0, water=0.5, energy=0.5),
        offers=[Action.REST],
    )
    state = WorldState(
        seed=3,
        step=0,
        minute=360,
        place=Place.MEADOW,
        body=config.agent.body,
        water_history=[0.47, 0.47],
        stream_full=True,
        forced_rest_left=0,
        last_observation=observation,
    )
    rt = Runtime(config=config, world=StillWorld(state))
    assert rt.model is None
    with pytest.raises(AttributeError):
        rt.model = "x"  # type: ignore[misc]
