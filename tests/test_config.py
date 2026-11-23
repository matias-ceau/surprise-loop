"""The config: parameter defaults and bounds, the experiment file, loading, and the model name."""

import math
from pathlib import Path
from typing import Any

import pytest
from creatures import LAYOUT, load_two_maps, run_config
from pydantic import BaseModel, ValidationError

from surprise_loop.config import (
    MODEL_ENV_VAR,
    AgentSpec,
    AppraiseParams,
    Backends,
    ConfigError,
    ExperimentConfig,
    FeelingParams,
    LensParams,
    LLMParams,
    MemoryParams,
    MetricsParams,
    MonitorParams,
    Params,
    PromptParams,
    ReasonerParams,
    RuleParams,
    RunConfig,
    derive_rng,
    load_config,
    resolve_model,
)
from surprise_loop.state import Sureness
from surprise_loop.world import BodyParams, Channel, Kind

PARAM_GROUPS = [
    LensParams,
    FeelingParams,
    MonitorParams,
    ReasonerParams,
    RuleParams,
    MemoryParams,
    AppraiseParams,
    PromptParams,
    MetricsParams,
    LLMParams,
]


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "exp.yaml"
    path.write_text(text)
    return path


def experiment_text(layout: tuple[str, ...] = LAYOUT, extra: str = "") -> str:
    rows = "\n".join(f'    - "{row}"' for row in layout)
    return f"""\
name: tiny
seeds: [1]
world:
  layout:
{rows}
agents:
  - name: plain
    temperament:
      presumes: {{ground: 0.6, pleasant: 0.06, unpleasant: 0.05, obstacle: 0.12, food: 0.06,
                 toxic: 0.03, water: 0.06, foul: 0.02}}
{extra}"""


# --- the experiment file ---------------------------------------------------------------------


def test_two_maps_has_four_agents_and_twenty_seeds() -> None:
    config = load_two_maps()
    assert config.name == "two_maps"
    assert [a.name for a in config.agents] == ["plain", "wary", "hopeful", "dull"]
    assert config.seeds == list(range(1, 21))
    assert config.ticks is None
    assert config.backends == Backends(deliberate="rule", appraise="reasoner")
    assert config.world.layout == LAYOUT
    assert config.world.body.smell_chance == 0.7
    assert config.world.body.damp_chance == 0.7
    assert config.params == Params()

    runs = config.runs()
    assert len(runs) == 80
    assert runs[0].run_id == "two_maps-plain-s1"
    assert runs[-1].run_id == "two_maps-dull-s20"
    assert resolve_model(runs[0]) is None


def test_two_maps_temperaments_are_the_table() -> None:
    agents = {a.name: a.temperament for a in load_two_maps().agents}
    plain, wary, hopeful, dull = (agents[n] for n in ("plain", "wary", "hopeful", "dull"))
    assert plain.presumes[Kind.GROUND] == 0.60
    assert wary.presumes[Kind.TOXIC] == 0.12
    assert hopeful.presumes[Kind.PLEASANT] == 0.18
    assert plain.gain == dict.fromkeys(Channel, 1.0)
    assert wary.gain[Channel.THIRST] == 1.2
    assert wary.gain[Channel.SICKNESS] == 1.4
    assert wary.gain[Channel.PLEASURE] == 0.7
    assert hopeful.gain[Channel.HUNGER] == 0.85
    assert hopeful.gain[Channel.UNEASE] == 0.7
    assert hopeful.gain[Channel.PLEASURE] == 1.4
    assert dull.presumes == plain.presumes
    assert dull.gain == plain.gain
    assert dull.noise == dict.fromkeys(Channel, 0.08)
    assert dull.blur == 0.5
    for t in (plain, wary, hopeful):
        assert t.noise == dict.fromkeys(Channel, 0.02)
        assert t.blur == 0.0


def test_run_config_helper_picks_the_run_and_applies_changes() -> None:
    run = run_config("wary", 3, ticks=5)
    assert run.run_id == "two_maps-wary-s3"
    assert run.agent.name == "wary"
    assert run.seed == 3
    assert run.ticks == 5
    assert run_config().run_id == "two_maps-plain-s1"


# --- loading ---------------------------------------------------------------------------------


def test_a_small_file_loads_with_defaults(tmp_path: Path) -> None:
    config = load_config(write(tmp_path, experiment_text()))
    assert config.world.body == BodyParams()
    assert config.backends == Backends()
    assert config.model is None
    assert [r.run_id for r in config.runs()] == ["tiny-plain-s1"]


def test_unknown_yaml_key_raises_config_error_naming_the_key(tmp_path: Path) -> None:
    text = experiment_text(extra="params:\n  memory: {recnet_k: 3}\n")
    with pytest.raises(ConfigError, match=r"params\.memory\.recnet_k"):
        load_config(write(tmp_path, text))


def test_a_bad_layout_names_world_layout_and_the_row(tmp_path: Path) -> None:
    layout = (*LAYOUT[:2], ".....X...", *LAYOUT[3:])
    with pytest.raises(ConfigError) as excinfo:
        load_config(write(tmp_path, experiment_text(layout)))
    message = str(excinfo.value)
    assert "world.layout" in message
    assert "row 3" in message


def test_the_world_is_required(tmp_path: Path) -> None:
    text = "name: x\nseeds: [1]\nagents: [{name: a, temperament: {presumes: {}}}]\n"
    with pytest.raises(ConfigError, match=r"world: Field required"):
        load_config(write(tmp_path, text))


def test_missing_file_raises_one_line_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"file not found$"):
        load_config(tmp_path / "nope.yaml")


def test_invalid_yaml_raises_one_line_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_config(write(tmp_path, "name: [unclosed\n"))
    message = str(excinfo.value)
    assert "invalid YAML at line" in message
    assert "\n" not in message


@pytest.mark.parametrize("text", ["- a\n- b\n", "just words\n", ""])
def test_non_mapping_top_level_raises_one_line_config_error(tmp_path: Path, text: str) -> None:
    with pytest.raises(ConfigError, match=r"the top level must be a mapping of config keys$"):
        load_config(write(tmp_path, text))


def test_config_error_is_a_value_error() -> None:
    assert issubclass(ConfigError, ValueError)


# --- experiment shape ------------------------------------------------------------------------


def spec(name: str = "a") -> dict[str, Any]:
    return {"name": name, "temperament": load_two_maps().agents[0].temperament.model_dump()}


def experiment(**overrides: Any) -> ExperimentConfig:
    data: dict[str, Any] = {
        "name": "exp",
        "seeds": [3],
        "world": {"layout": list(LAYOUT)},
        "agents": [spec()],
    }
    return ExperimentConfig.model_validate(data | overrides)


def test_seeds_and_agent_names_must_be_unique() -> None:
    with pytest.raises(ValidationError, match="seeds must be unique"):
        experiment(seeds=[1, 1])
    with pytest.raises(ValidationError, match="agent names must be unique"):
        experiment(agents=[spec("a"), spec("a")])


@pytest.mark.parametrize("name", ["a b", "../up", ""])
def test_names_are_restricted_to_safe_characters(name: str) -> None:
    with pytest.raises(ValidationError):
        experiment(name=name)
    with pytest.raises(ValidationError):
        AgentSpec.model_validate(spec(name))


def test_agent_spec_has_only_name_and_temperament() -> None:
    assert set(AgentSpec.model_fields) == {"name", "temperament"}
    assert "start_place" not in RunConfig.model_fields
    assert "start_place" not in ExperimentConfig.model_fields


def test_backends_accept_only_their_choices() -> None:
    assert Backends() == Backends(deliberate="rule", appraise="reasoner")
    Backends(deliberate="blind", appraise="llm")
    with pytest.raises(ValidationError):
        Backends(deliberate="reasoner")
    with pytest.raises(ValidationError):
        Backends(appraise="rule")


# --- resolve_model ---------------------------------------------------------------------------


def test_resolve_model_is_none_without_an_llm_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MODEL_ENV_VAR, "env-model")
    assert resolve_model(run_config(model="config-model")) is None


@pytest.mark.parametrize(
    "backends", [Backends(deliberate="llm"), Backends(appraise="llm")], ids=["deliberate", "appr"]
)
def test_resolve_model_prefers_config_then_env(
    monkeypatch: pytest.MonkeyPatch, backends: Backends
) -> None:
    monkeypatch.setenv(MODEL_ENV_VAR, "env-model")
    assert resolve_model(run_config(backends=backends, model="config-model")) == "config-model"
    assert resolve_model(run_config(backends=backends)) == "env-model"


@pytest.mark.parametrize("env", [None, ""])
def test_resolve_model_raises_when_llm_has_no_model(
    monkeypatch: pytest.MonkeyPatch, env: str | None
) -> None:
    if env is None:
        monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(MODEL_ENV_VAR, env)
    with pytest.raises(ConfigError, match=MODEL_ENV_VAR):
        resolve_model(run_config(backends=Backends(deliberate="llm")))


# --- derive_rng ------------------------------------------------------------------------------


def test_derive_rng_is_fixed_by_seed_and_key() -> None:
    assert derive_rng(7, "a", 3).random() == 0.2589360035904338
    assert derive_rng(7, "a", 3).random() != derive_rng(7, "a", 4).random()


# --- params ----------------------------------------------------------------------------------


@pytest.mark.parametrize("group", PARAM_GROUPS, ids=lambda g: g.__name__)
def test_every_default_validates(group: type[BaseModel]) -> None:
    default = group()
    assert group.model_validate(default.model_dump()) == default


def test_params_defaults_are_the_contract_values() -> None:
    p = Params()
    assert p.lens.edges == (0.1, 0.3, 0.55, 0.8)
    assert p.feeling.tone_weights == {
        Channel.HUNGER: 0.5,
        Channel.THIRST: 0.5,
        Channel.ACHE: 1.0,
        Channel.SICKNESS: 1.0,
        Channel.UNEASE: 1.0,
        Channel.PLEASURE: 1.0,
    }
    assert p.feeling.tone_edges == (-0.75, -0.3, 0.15, 0.6)
    assert (p.feeling.stir_rate, p.feeling.stir_edges) == (0.5, (0.15, 0.4))
    assert (p.monitor.moved_weight, p.monitor.sense_weight, p.monitor.scale) == (1.0, 0.5, 2.0)
    assert p.monitor.channel_weight == dict.fromkeys(Channel, 1.0)
    assert p.monitor.sure_factor == {
        Sureness.GUESSING: 0.5,
        Sureness.FAIRLY_SURE: 0.75,
        Sureness.SURE: 1.0,
    }
    assert (p.monitor.surprise_edges, p.monitor.worse_edge) == ((0.05, 0.4), 0.1)
    assert (p.reasoner.slip, p.reasoner.ache_when_slight) == (0.05, 0.5)
    assert p.rule == RuleParams(need_at=2, serve_at=0.5, avoid_at=0.5, believe_at=0.5, reach=12)
    assert (p.memory.recent_k, p.appraise.max_changes, p.prompt.decimals) == (6, 12, 2)
    assert p.metrics.comfort_at == {
        Channel.HUNGER: 0.55,
        Channel.THIRST: 0.55,
        Channel.ACHE: 0.3,
        Channel.SICKNESS: 0.3,
        Channel.UNEASE: 0.3,
        Channel.PLEASURE: 1.0,
    }
    assert (p.metrics.wrong_side, p.metrics.rewritten_by) == (0.5, 0.1)
    assert p.llm == LLMParams(output_retries=3, temperature=1.0, timeout_s=60.0)


@pytest.mark.parametrize(
    ("group", "field"),
    [
        (LensParams, "edges"),
        (FeelingParams, "tone_edges"),
        (FeelingParams, "stir_edges"),
        (MonitorParams, "surprise_edges"),
    ],
)
def test_edges_must_be_strictly_ascending(group: type[BaseModel], field: str) -> None:
    edges = getattr(group(), field)
    flipped = (edges[1], edges[0], *edges[2:])
    with pytest.raises(ValidationError, match="strictly ascending"):
        group.model_validate({field: flipped})
    repeated = (edges[0], edges[0], *edges[2:])
    with pytest.raises(ValidationError, match="strictly ascending"):
        group.model_validate({field: repeated})
    with pytest.raises(ValidationError, match="exactly"):
        group.model_validate({field: edges[:-1]})


@pytest.mark.parametrize("value", [-0.1, 1.1, math.nan, math.inf])
@pytest.mark.parametrize(
    ("group", "field"),
    [
        (FeelingParams, "stir_rate"),
        (ReasonerParams, "slip"),
        (ReasonerParams, "ache_when_slight"),
        (RuleParams, "serve_at"),
        (RuleParams, "avoid_at"),
        (RuleParams, "believe_at"),
        (MetricsParams, "wrong_side"),
        (MetricsParams, "rewritten_by"),
    ],
)
def test_unit_params_reject_out_of_range_nan_and_inf(
    group: type[BaseModel], field: str, value: float
) -> None:
    with pytest.raises(ValidationError, match=field):
        group.model_validate({field: value})


@pytest.mark.parametrize(
    ("group", "field", "value"),
    [
        (MonitorParams, "scale", 0.0),
        (MonitorParams, "scale", math.inf),
        (MonitorParams, "moved_weight", math.nan),
        (MonitorParams, "worse_edge", -0.1),
        # strictly between 0 and 1: at either end the reasoner's shares can all become 0
        (ReasonerParams, "slip", 0.0),
        (ReasonerParams, "slip", 1.0),
        (ReasonerParams, "ache_when_slight", 0.0),
        (ReasonerParams, "ache_when_slight", 1.0),
        (RuleParams, "need_at", 5),
        (RuleParams, "reach", 0),
        (MemoryParams, "recent_k", -1),
        (AppraiseParams, "max_changes", 0),
        (PromptParams, "decimals", -1),
        (LLMParams, "timeout_s", 0.0),
        (LLMParams, "temperature", math.nan),
    ],
)
def test_bounded_params_reject_bad_values(group: type[BaseModel], field: str, value: Any) -> None:
    with pytest.raises(ValidationError, match=field):
        group.model_validate({field: value})


@pytest.mark.parametrize(
    ("group", "field", "keys"),
    [
        (FeelingParams, "tone_weights", Channel),
        (MonitorParams, "channel_weight", Channel),
        (MonitorParams, "sure_factor", Sureness),
        (MetricsParams, "comfort_at", Channel),
    ],
)
def test_enum_keyed_params_need_every_key(group: type[BaseModel], field: str, keys: Any) -> None:
    partial = {k.value: 0.5 for k in list(keys)[1:]}
    with pytest.raises(ValidationError, match=f"missing keys: {list(keys)[0].value}"):
        group.model_validate({field: partial})


def test_unknown_params_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="extra"):
        Params.model_validate({"lens": {"gain": 1.0}})
    with pytest.raises(ValidationError, match="extra"):
        Params.model_validate({"override": {}})
