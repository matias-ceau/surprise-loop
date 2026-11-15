"""Suite-wide test settings."""

from pydantic_ai import models

# No test may reach a real model: any request to a non-test model raises. TestModel and
# FunctionModel are unaffected.
models.ALLOW_MODEL_REQUESTS = False
