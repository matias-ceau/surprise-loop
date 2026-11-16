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

These are first numbers from v0. They show the pipeline works and point in a direction; they are
not a finding. Stability is ICC(1) of each metric across seeds within a trait profile (1 = all the
variance lies between profiles, 0 or below = none).

### Stub phases: `experiments/trait_grid.yaml`

7 profiles × 5 seeds, schedule FR1 60 → PR(1, +2) 150 → FR1 40 → extinction 150 world steps, zero
API calls. No breakpoint or extinction value is censored.

| profile | breakpoint | extinction persistence | PR response rate | extinction response rate |
|---|---|---|---|---|
| baseline | 9.4 ± 0.9 | 17.4 ± 1.8 | 0.31 ± 0.01 | 0.19 ± 0.01 |
| reward_sensitivity_low (0.5) | 7.8 ± 1.1 | 14.6 ± 2.6 | 0.25 ± 0.03 | 0.16 ± 0.01 |
| reward_sensitivity_high (2.0) | 9.8 ± 1.1 | 17.8 ± 3.0 | 0.39 ± 0.01 | 0.22 ± 0.01 |
| surprise_learning_rate_low (0.2) | 13.8 ± 1.1 | 39.0 ± 5.5 | 0.51 ± 0.00 | 0.34 ± 0.02 |
| surprise_learning_rate_high (0.8) | 7.4 ± 0.9 | 11.6 ± 1.3 | 0.24 ± 0.02 | 0.15 ± 0.00 |
| mood_inertia_low (0.1) | 9.4 ± 0.9 | 17.6 ± 1.8 | 0.30 ± 0.01 | 0.19 ± 0.00 |
| mood_inertia_high (0.9) | 11.4 ± 0.9 | 22.2 ± 1.3 | 0.39 ± 0.01 | 0.21 ± 0.01 |

Mean ± SD over seeds 1–5. ICC(1): breakpoint 0.82, extinction persistence 0.91, PR response rate
0.97, extinction response rate 0.98, mean |PE| 0.96, inactive-lever rate 0.87. Total rewards,
FR response rate (at ceiling), re-entries and goal revisions have ICC(1) at or below 0.19.

`tests/test_behaviour.py` checks the three predicted contrasts on 20 seeds: higher reward sensitivity
raises breakpoint, higher surprise learning rate lowers persistence, and higher mood inertia raises
persistence. Each paired difference exceeds 8 standard errors, and they hold on 20 seeds that were
not used for tuning.

Caveats:
- **The habit-versus-goal re-entry almost never fires.** It happens about twice per 400-step run,
  so that channel barely shapes behaviour.
- **Trait effects are not specific to one metric.** Mood inertia also raises breakpoint, and reward
  sensitivity's effect on persistence is not monotonic.
- **The mood-inertia effect depends on a tuned default** (`deliberate.vigor_weight = 2.0`), which also
  lifts the inactive-lever rate to about 0.07.
- **The stub results show only that the deterministic channels can carry trait differences.** In
  stub runs the traits enter through hand-written heuristics.

### Real model: `experiments/llm_traits.yaml`

All three LLM phases ran on one small open-weights model through OpenRouter: 4 profiles × 2 seeds,
150 world steps each (FR1 15 → PR(1, +2) 75 → FR1 10 → extinction 50), about 3,500 model calls,
40 minutes with `--jobs 8`. Every output passed validation, and `replay` verifies the dumps.

| profile | seed | breakpoint | extinction persistence | PR response rate | extinction response rate |
|---|---|---|---|---|---|
| baseline | 1 / 2 | 15* / 15* | 21 / 16 | 0.92 / 0.91 | 0.42 / 0.46 |
| surprise_learning_rate_low | 1 / 2 | 15* / 15* | 31* / 32* | 0.93 / 0.91 | 0.62 / 0.64 |
| surprise_learning_rate_high | 1 / 2 | 11* / 13* | 5 / 15* | 0.64 / 0.72 | 0.14 / 0.30 |
| reward_sensitivity_high | 1 / 2 | 15* / 15* | 25* / 28* | 0.93 / 0.89 | 0.50 / 0.56 |

`*` = censored: the phase ended before the quit criterion was met, so the value is a lower bound.
The model never quit in the 75-step PR phase, so breakpoint measures phase length here, not the
agent. Persistence orders the profiles the same way as the stubs: a low surprise learning rate is
most persistent and a high one least. With two seeds and heavy censoring this is a direction, not a
result. ICC(1) of PR and extinction response rate is 0.92 and 0.88. Longer PR and extinction phases
are the next step.

### Real model, longer phases: `experiments/llm_traits_long.yaml`

This uses the same model, profiles and seeds, with PR lengthened to 200 steps and extinction to
150, giving 375 world steps per run. It made about 9,000 model calls and took 47 minutes with
`--jobs 8`. Every output passed validation, and `replay` verifies the dumps (checked on two runs).

| profile | seed | breakpoint | extinction persistence | PR response rate | extinction response rate | goal revisions |
|---|---|---|---|---|---|---|
| baseline | 1 / 2 | 13 / 19 | 10 / 56* | 0.40 / 0.71 | 0.13 / 0.37 | 8 / 10 |
| surprise_learning_rate_low | 1 / 2 | 23* / 23* | 56* / 36 | 0.81 / 0.81 | 0.37 / 0.38 | 5 / 3 |
| surprise_learning_rate_high | 1 / 2 | 19 / 21 | 17 / 59* | 0.59 / 0.78 | 0.29 / 0.39 | 16 / 17 |
| reward_sensitivity_high | 1 / 2 | 17 / 17 | 12 / 10 | 0.56 / 0.56 | 0.21 / 0.13 | 8 / 9 |

With room to quit, the model now meets the quit criterion in PR in 6 of 8 runs, so breakpoint
measures the agent rather than the phase length. Three findings:

- **The behavioural differences mostly do not hold within a profile.** ICC(1) is 0.60 for
  breakpoint, 0.30 for PR response rate, 0.28 for extinction response rate and −0.09 for extinction
  persistence.
  - `surprise_learning_rate_low` and `reward_sensitivity_high` behave alike on both seeds: one
    presses the most, the other the least.
  - The baseline and `surprise_learning_rate_high` profiles differ more between their two seeds
    than between each other.
  - The short run's ordering, where a high surprise learning rate gave the least persistence, does
    not replicate.
- **The stable numbers are mechanical.** Goal revisions and re-entries have ICC(1) of 0.95 and
  0.81. The deterministic Reconcile scales its goal error average by the surprise learning rate, so
  they mostly track that trait directly, not the model's choices.
- **The answer to the research question here is "not yet".** With one small model, two seeds per
  profile and traits given to the LLM only as numbers in the prompt and through the deterministic
  channels, the LLM-driven agent shows a few consistent profiles, not stable individual
  differences across the board.

The next steps are more seeds per profile and a quit criterion that tolerates short pauses: the
extinction values swing because a single 10-step pause ends the count.

Reproduce:

```sh
uv run surprise-loop experiment experiments/trait_grid.yaml --jobs 8
uv run --group plot python scripts/plot_results.py runs/trait_grid
SURPRISE_LOOP_MODEL=openrouter:<provider>/<model> OPENROUTER_API_KEY=... \
  uv run surprise-loop experiment experiments/llm_traits.yaml --jobs 8
uv run --group plot python scripts/plot_results.py runs/llm_traits
# the same with experiments/llm_traits_long.yaml (and runs/llm_traits_long) for the longer phases
```
