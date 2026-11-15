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
7. **Reconcile** (deterministic in v0): apply feedback through four channels on different timescales:
   mood (fast), memory (per tick, salience weighted by absolute prediction error), goals (slow,
   only after sustained error), dispositions (slowest). It may send control back to Deliberate
   at most once per tick.

Every LLM phase (Appraise, Deliberate, Evaluate) has a deterministic stub with the same
signature, chosen per phase under `backends:` in the config (`stub` or `llm`), so an experiment can
run with zero API calls. The configs in `experiments/` other than `llm_*.yaml` are all-stub.

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

## Single run, resume, replay

```sh
# one agent and seed of a config (default: the first agent and the first seed);
# writes runs/<run_id>/config.json and one tick_<n>.json per tick, prints the metrics as JSON
uv run surprise-loop run experiments/stub_smoke.yaml
uv run surprise-loop run experiments/trait_grid.yaml --agent mood_inertia_high --seed 3

# continue an interrupted run from its last tick dump (same config required)
uv run surprise-loop run experiments/trait_grid.yaml --agent mood_inertia_high --seed 3 --resume

# print the metrics of a stored run, or re-run its ticks from the dumps and check every byte
uv run surprise-loop metrics runs/trait_grid-mood_inertia_high-s3
uv run surprise-loop replay runs/trait_grid-mood_inertia_high-s3
```

The run id is `<config name>-<agent>-s<seed>`; `--runs-dir` changes the parent directory (default
`runs`). Replay feeds the recorded LLM outputs back in, so it makes no model calls. A bad config or
run directory exits 2, a replay that differs from the dumps exits 1.

## Run an experiment and plot it

```sh
uv run surprise-loop experiment experiments/trait_grid.yaml --runs-dir runs --jobs 8
uv run --group plot python scripts/plot_results.py runs/trait_grid
```

The first command writes the CSV tables into `runs/trait_grid/`; the second writes
`metrics_by_agent.png`, `pe_trace.png` and `cumulative_record.png` into `runs/trait_grid/plots/`.
Rerunning the same command skips complete runs and resumes partial ones.

## LLM backends

Set a phase's backend to `llm` under `backends:` to have pydantic-ai drive it. The model is a
pydantic-ai model name, `<provider>:<model>`, taken from the config's `model:` key or, when that is
null, from the `SURPRISE_LOOP_MODEL` environment variable. The provider's API key comes from its
usual environment variable, for example for OpenRouter:

```sh
export OPENROUTER_API_KEY=...
export SURPRISE_LOOP_MODEL=openrouter:<provider>/<model>
uv run surprise-loop experiment experiments/llm_traits.yaml --jobs 8
```

`experiment` loads every model it needs before any run starts, so an unknown provider or a missing
key fails at once (exit 2). Regenerating the tables of a finished LLM experiment needs no model.
Request settings (output retries, temperature, timeout) live under `params.llm`. Output that fails
validation (out of range, an action and prediction that do not fit, an `expected_reward` above the
world's reward magnitude) is sent back to the model, up to `output_retries` times; after that the
run stops with an error and no stub fallback. The tests use pydantic-ai's test model only.

A design choice in the Deliberate instructions: the model is told that a reward always comes in
the same fixed amount and, until it has seen one, to keep `expected_reward` small. This fixed text
plays the part of the stub's `deliberate.prior_expected_reward` (a low prior before any reward has
been seen); it is not a config parameter.

## Design

The full v0 design, contracts and maths are in [`docs/plan-v0.md`](docs/plan-v0.md).

## Results

TBD
