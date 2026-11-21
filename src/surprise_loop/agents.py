"""Everything a model call needs. For now: the model by name.

The model is resolved once per run and passed to every call, so all calls share one HTTP client.

No import from `surprise_loop.phases`: the phases import this module.
"""

from pydantic_ai import UserError, models

from surprise_loop.config import ConfigError


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
