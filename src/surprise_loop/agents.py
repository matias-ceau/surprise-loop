"""What every model call shares: resolving the model by name, and the request settings.

Agents are built without a model: the model is resolved once per run (or per ladder check) and
passed to every `agent.run(..., model=model)`, so all calls share one HTTP client.
"""

from pydantic_ai import ModelSettings, UserError, models

from surprise_loop.config import ConfigError, LLMParams


def load_model(name: str | None) -> models.Model | None:
    """The pydantic-ai model for `name` (e.g. "openrouter:<id>", or "test" for `TestModel`).

    Resolved once per run, so every call of the run shares one model and one HTTP client.
    """
    if name is None:
        return None
    try:
        return models.infer_model(name)
    except UserError as e:
        raise ConfigError(f"model {name!r}: {e}") from None


def model_settings(params: LLMParams) -> ModelSettings:
    """The request settings every model call sends, from `Params.llm`."""
    settings = ModelSettings(timeout=params.timeout_s)
    if params.temperature is not None:
        settings["temperature"] = params.temperature
    return settings
