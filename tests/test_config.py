import math
import random
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from surprise_loop.config import (
    AgentSpec,
    AppraiseParams,
    Backends,
    ConfigError,
    ConflictParams,
    DeliberateParams,
    DispositionParams,
    EvaluateParams,
    ExperimentConfig,
    GoalParams,
    MemoryParams,
    MetricsParams,
    MonitorParams,
    MoodParams,
    Params,
    RunConfig,
    Runtime,
    derive_rng,
    initial_agent_state,
    load_config,
    resolve_model,
)
from surprise_loop.state import GoalName, Mood, Traits
from surprise_loop.world import Action, SelfAdminWorld, WorldConfig

YAML = """\
name: pilot
seeds: [1, 2]
ticks: 50
world:
  active_lever: random
  reward_magnitude: 1.0
  schedule:
    - {kind: fixed_ratio, steps: 30, ratio: 1}
    - {kind: progressive_ratio, steps: 30, pr_start: 1, pr_increment: 2}
    - {kind: extinction, steps: 30}
agents:
  - name: high
    traits: {reward_sensitivity: 2.0, surprise_learning_rate: 0.5, mood_inertia: 0.3}
  - name: low
    traits: {reward_sensitivity: 0.5, surprise_learning_rate: 0.2, mood_inertia: 0.8}
    mood: {valence: 0.1, arousal: 0.2}
    goal_priorities: {seek_reward: 0.8, explore: 0.6, conserve: 0.4}
params:
  goals: {patience: 5}
  deliberate: {action_costs: {press_a: 0.1, press_b: 0.1, explore: 0.0, rest: 0.0}}
"""

SCHEDULE = [{"kind": "extinction", "steps": 10}]


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "exp.yaml"
    path.write_text(text)
    return path


def experiment(**overrides: Any) -> ExperimentConfig:
    data: dict[str, Any] = {
        "name": "exp",
        "seeds": [3],
        "world": {"active_lever": "press_a", "reward_magnitude": 1.0, "schedule": SCHEDULE},
        "agents": [
            {
                "name": "a",
                "traits": {
                    "reward_sensitivity": 1.5,
                    "surprise_learning_rate": 0.4,
                    "mood_inertia": 0.6,
                },
                "mood": {"valence": 0.2, "arousal": 0.1},
                "goal_priorities": {"seek_reward": 1.2, "explore": 0.3, "conserve": 0.7},
            }
        ],
    }
    data.update(overrides)
    return ExperimentConfig.model_validate(data)


def run_config(**overrides: Any) -> RunConfig:
    return experiment(**overrides).runs()[0]


# --- loading -------------------------------------------------------------------------------


def test_yaml_loads_and_runs_are_agents_times_seeds(tmp_path: Path) -> None:
    config = load_config(write(tmp_path, YAML))
    assert isinstance(config, ExperimentConfig)
    assert config.params.goals.patience == 5
    assert config.params.goals.ema_rate == GoalParams().ema_rate
    assert config.params.deliberate.action_costs[Action.PRESS_A] == 0.1
    assert config.agents[0].mood == Mood(valence=0.0, arousal=0.0)
    assert config.backends == Backends(appraise="stub", deliberate="stub", evaluate="stub")
    assert config.model is None

    runs = config.runs()
    assert [(r.run_id, r.agent.name, r.seed) for r in runs] == [
        ("pilot-high-s1", "high", 1),
        ("pilot-high-s2", "high", 2),
        ("pilot-low-s1", "low", 1),
        ("pilot-low-s2", "low", 2),
    ]
    for run in runs:
        assert isinstance(run, RunConfig)
        assert run.ticks == 50
        assert run.world == config.world
        assert run.params == config.params
        assert run.backends == config.backends
        assert run.model is None


def test_unknown_yaml_key_raises_config_error_naming_the_key_path(tmp_path: Path) -> None:
    path = write(tmp_path, YAML.replace("patience: 5", "pateince: 5"))
    with pytest.raises(ConfigError, match=r"params\.goals\.pateince") as excinfo:
        load_config(path)
    assert "not permitted" in str(excinfo.value)


def test_validation_errors_are_one_line_per_problem(tmp_path: Path) -> None:
    text = YAML.replace("patience: 5", "patience: 0, typo: 1")
    with pytest.raises(ConfigError) as excinfo:
        load_config(write(tmp_path, text))
    lines = str(excinfo.value).splitlines()
    assert any(line.strip().startswith("params.goals.patience:") for line in lines)
    assert any(line.strip().startswith("params.goals.typo:") for line in lines)
    assert len(lines) == 3  # header + one line per problem
    assert "Traceback" not in str(excinfo.value)


def test_list_indices_appear_in_the_key_path(tmp_path: Path) -> None:
    text = YAML.replace("mood_inertia: 0.8", "mood_inertia: 1.5")
    with pytest.raises(ConfigError, match=r"agents\.1\.traits\.mood_inertia"):
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


def test_config_error_is_a_value_error() -> None:
    assert issubclass(ConfigError, ValueError)


# --- validation ----------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["a/b", "", "has space", "../x"])
def test_names_are_restricted_to_safe_characters(name: str) -> None:
    with pytest.raises(ValidationError):
        experiment(name=name)
    with pytest.raises(ValidationError):
        AgentSpec(
            name=name, traits=Traits(reward_sensitivity=1, surprise_learning_rate=1, mood_inertia=0)
        )


def test_experiment_needs_seeds_and_agents() -> None:
    with pytest.raises(ValidationError):
        experiment(seeds=[])
    with pytest.raises(ValidationError):
        experiment(agents=[])


def test_agent_names_must_be_unique() -> None:
    agent = experiment().agents[0].model_dump()
    with pytest.raises(ValidationError, match="unique"):
        experiment(agents=[agent, agent])


def test_ticks_must_be_positive_when_set() -> None:
    assert experiment().ticks is None
    assert experiment(ticks=1).ticks == 1
    with pytest.raises(ValidationError):
        experiment(ticks=0)


def test_agent_goal_priorities_default_and_must_cover_every_goal() -> None:
    t = Traits(reward_sensitivity=1.0, surprise_learning_rate=0.5, mood_inertia=0.5)
    spec = AgentSpec(name="x", traits=t)
    assert spec.goal_priorities == {
        GoalName.SEEK_REWARD: 1.0,
        GoalName.EXPLORE: 0.5,
        GoalName.CONSERVE: 0.5,
    }
    with pytest.raises(ValidationError, match="conserve"):
        AgentSpec(name="x", traits=t, goal_priorities={"seek_reward": 1.0, "explore": 0.5})


def test_params_defaults_are_the_tuned_values() -> None:
    # The values tuned against tests/test_behaviour.py; change them only together with that gate.
    p = Params()
    assert p.monitor == MonitorParams(confidence_floor=0.3, event_mismatch_weight=0.5)
    assert p.mood == MoodParams(arousal_gain=1.0)
    assert p.memory == MemoryParams(retrieval_k=20, base_salience=0.1)
    assert p.goals == GoalParams(
        ema_rate=0.2,
        error_threshold=0.05,
        patience=3,
        revision_step=0.1,
        min_priority=0.1,
        max_priority=2.0,
    )
    assert p.dispositions == DispositionParams(rate=0.2, bound=1.0, alignment_weight=0.2)
    assert p.conflict == ConflictParams(enabled=True, habit_threshold=0.1)
    assert p.deliberate == DeliberateParams(
        temperature=0.1,
        prior_expected_reward=0.3,
        prior_weight=1.0,
        action_costs={"press_a": 0.05, "press_b": 0.05, "explore": 0.02, "rest": 0.0},
        approach_weight=0.2,
        vigor_weight=2.0,
        explore_value=0.3,
        rest_value=0.3,
        confidence_halfsat=3.0,
    )
    assert p.appraise == AppraiseParams(reward_gain=0.5)
    assert p.evaluate == EvaluateParams(scale=0.2)
    assert p.metrics == MetricsParams(quit_steps=10)


# (params group, field, out-of-range values); NaN and ±inf are added below
BAD_PARAMS: list[tuple[type[BaseModel], str, list[Any]]] = [
    (MonitorParams, "confidence_floor", [-0.1, 1.1]),
    (MonitorParams, "event_mismatch_weight", [-0.1]),
    (MoodParams, "arousal_gain", [-0.1]),
    (MemoryParams, "retrieval_k", [0]),
    (MemoryParams, "base_salience", [-0.1, 1.1]),
    (GoalParams, "ema_rate", [-0.1, 1.1]),
    (GoalParams, "error_threshold", [-0.1]),
    (GoalParams, "patience", [0]),
    (GoalParams, "revision_step", [-0.1]),
    (GoalParams, "min_priority", [-0.1, 3.0]),
    (GoalParams, "max_priority", [0.05]),
    (DispositionParams, "rate", [-0.1, 1.1]),
    (DispositionParams, "bound", [0.0]),
    (DispositionParams, "alignment_weight", [-0.1]),
    (ConflictParams, "habit_threshold", []),
    (DeliberateParams, "temperature", [0.0, -1.0]),
    (DeliberateParams, "prior_expected_reward", []),
    (DeliberateParams, "prior_weight", [0.0, -1.0]),
    (DeliberateParams, "approach_weight", []),
    (DeliberateParams, "vigor_weight", []),
    (DeliberateParams, "explore_value", []),
    (DeliberateParams, "rest_value", []),
    (DeliberateParams, "confidence_halfsat", [0.0]),
    (AppraiseParams, "reward_gain", [-0.1]),
    (EvaluateParams, "scale", [0.0, -1.0]),
    (MetricsParams, "quit_steps", [0]),
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


def test_prior_weight_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="prior_weight"):
        DeliberateParams(prior_weight=0.0)


def test_action_costs_must_cover_every_action() -> None:
    with pytest.raises(ValidationError, match="rest"):
        DeliberateParams(action_costs={"press_a": 0.0, "press_b": 0.0, "explore": 0.0})
    with pytest.raises(ValidationError):
        DeliberateParams(action_costs={a: math.nan for a in Action})


def test_unknown_params_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Params.model_validate({"goals": {"pateince": 3}})
    with pytest.raises(ValidationError):
        Params.model_validate({"habits": {}})


# --- resolve_model -------------------------------------------------------------------------

LLM = {"appraise": "stub", "deliberate": "llm", "evaluate": "stub"}


def test_resolve_model_is_none_when_every_backend_is_a_stub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SURPRISE_LOOP_MODEL", "env-model")
    assert resolve_model(run_config(model="config-model")) is None


def test_resolve_model_prefers_config_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SURPRISE_LOOP_MODEL", "env-model")
    assert resolve_model(run_config(backends=LLM, model="config-model")) == "config-model"


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


# --- initial state, rng, runtime -----------------------------------------------------------


def test_initial_agent_state_comes_from_the_spec() -> None:
    config = run_config()
    state = initial_agent_state(config)
    assert state.traits == config.agent.traits
    assert state.mood == Mood(valence=0.2, arousal=0.1)
    assert state.goals.priorities == {
        GoalName.SEEK_REWARD: 1.2,
        GoalName.EXPLORE: 0.3,
        GoalName.CONSERVE: 0.7,
    }
    assert state.memory.episodes == []
    assert state.dispositions.biases == {a: 0.0 for a in Action}
    assert state.goals.error_ema == {g: 0.0 for g in GoalName}
    assert state.goals.error_streak == {g: 0 for g in GoalName}
    assert state.goals.revisions == 0
    assert state.tick == 0


def test_derive_rng_is_stable_and_keyed() -> None:
    assert derive_rng(7, 3, 0).random() == 0.816605007653763
    assert derive_rng(7, 3, 0).random() != derive_rng(7, 3, 1).random()
    assert derive_rng(7, "a").random() == random.Random("7:a").random()


def test_derive_rng_matches_the_worlds_active_lever_draw() -> None:
    for seed in range(20):
        config = WorldConfig(active_lever="random", reward_magnitude=1.0, schedule=SCHEDULE)
        drawn = SelfAdminWorld(config, seed).snapshot().active_lever
        expected = derive_rng(seed, "active_lever").choice([Action.PRESS_A, Action.PRESS_B])
        assert drawn == expected


def test_runtime_is_a_frozen_dataclass() -> None:
    config = run_config()
    world = SelfAdminWorld(config.world, config.seed)
    rt = Runtime(config=config, world=world)
    assert rt.model is None
    with pytest.raises(AttributeError):
        rt.model = "x"  # type: ignore[misc]
