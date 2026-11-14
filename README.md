# surprise-loop

An embodied LLM agent driven by prediction error: it predicts what the world will do, acts,
compares, and uses the surprise to update mood, memory, goals and dispositions.

## Research question (v0)

Does a prediction-error-driven agent show stable individual differences in reinforcement
behaviour?

Testbed: a self-administration-style world. The agent can press lever A, press lever B, explore,
or rest. One action delivers a reward under a schedule: fixed ratio, then progressive ratio, then
extinction. Agents differ only in a small set of trait parameters (reward sensitivity, learning
rate on surprise, mood inertia). Measured per agent: response rate, breakpoint under progressive
ratio, persistence in extinction, and the prediction-error trace.

## Cognitive cycle

One tick is one pass through seven phases over a shared typed state:

1. **Perceive**: read the world observation, filter by attention.
2. **Appraise** (LLM): affective reading of the observation given mood and goals.
3. **Deliberate** (LLM): choose an action and emit an explicit prediction of the outcome
   (expected observation and reward, with a confidence).
4. **Act**: send the action to the world and get the actual outcome.
5. **Monitor** (deterministic): prediction error = actual vs predicted.
6. **Evaluate** (LLM): external critique of the action against goals; feedback only.
7. **Reconcile** (deterministic): apply feedback through four channels on different timescales:
   mood (fast), memory (per tick, salience weighted by absolute prediction error), goals (slow,
   only after sustained error), dispositions (slowest). It may send control back to Deliberate
   at most once per tick.

Every LLM phase has a deterministic stub with the same signature, so an experiment can run with
zero API calls.

## Boundaries

- The agent core is the only owner of mental state (mood, memory, goals, dispositions, traits).
- The world is the only owner of environment state. In v0 it is deterministic Python code with a
  seeded RNG.
- The evaluator gives feedback and nothing else.

## Install and test

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 or later.

```sh
uv sync --all-groups
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## Design

The full v0 design, contracts and maths are in [`docs/plan-v0.md`](docs/plan-v0.md).
