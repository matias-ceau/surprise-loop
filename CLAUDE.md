# surprise-loop

An embodied LLM agent driven by prediction error: it predicts what the world will do, acts, compares, and uses the surprise to update mood, memory, goals and dispositions.

This file is the full brief. If the repo is empty, bootstrap it by following "Build order" from step 1.

## Research question (v0)

Does a prediction-error-driven agent show stable individual differences in reinforcement behaviour?

Testbed: a self-administration-style world. The agent can take actions (e.g. press lever A, press lever B, explore, rest). One action delivers a reward under a schedule (fixed ratio, then progressive ratio, then extinction). Agents differ only in a small set of trait parameters (reward sensitivity, learning rate on surprise, mood inertia). Measure per agent: response rate, breakpoint under progressive ratio, persistence in extinction, and the prediction-error trace.

Everything in the architecture exists to serve this experiment. Do not add a component the experiment does not need.

## Cognitive cycle

One tick = one pass through seven phases. Each phase is a plain async function over a shared typed state.

1. Perceive: read the world observation, filter by attention.
2. Appraise (LLM): affective reading of the observation given mood and goals.
3. Deliberate (LLM): choose an action AND emit an explicit prediction of the outcome (expected observation plus expected reward, with a confidence).
4. Act: send the action to the World, get the actual outcome.
5. Monitor (deterministic): compute prediction error = actual vs predicted. This is the core signal.
6. Evaluate (LLM): external critique of the action against goals. Returns feedback only, never mutates state.
7. Reconcile (deterministic in v0): apply feedback through four channels, each with its own timescale:
   - mood (fast): valence and arousal move with signed prediction error
   - memory (per tick): store the episode, salience weighted by absolute prediction error
   - goals (slow): revise only after sustained error
   - dispositions (slowest): action biases drift with accumulated outcomes

   If Reconcile detects a conflict between channels it may send control back to Deliberate, at most once per tick.

## Boundaries

- CoreAgent is the only owner of mental state (mood, memory, goals, dispositions, traits).
- World is the only owner of environment state. In v0 the World is deterministic Python code with a seeded RNG, not an LLM. An LLM-narrated world is a later option.
- Evaluator gives feedback and nothing else.

## Stack

- Python 3.12+, managed with uv
- pydantic v2 for all state models
- pydantic-ai (current 2.x) for the LLM-backed phases, with structured outputs
- pytest; pydantic-ai's test model for LLM-free tests
- ruff for lint and format

The pydantic-ai API changed a lot in v2 (June 2026). Before writing any agent code, read the current docs at https://ai.pydantic.dev and the installed package source. Do not rely on memory of the 0.x or 1.x API. Do not use `pydantic_graph.persistence`; it was removed.

Orchestration in v0 is a plain Python loop calling the seven phase functions. Introduce pydantic-graph only if the bounded re-entry path makes the plain loop unreadable.

State persistence: dump the full state model to JSON after every tick into `runs/<run_id>/tick_<n>.json`. A run must be resumable and replayable from these files.

## Layout

```
src/surprise_loop/
  state.py        # Mood, Memory, Goals, Dispositions, Traits, AgentState
  world.py        # World protocol + SelfAdminWorld (schedules, seeded RNG)
  phases/         # perceive, appraise, deliberate, act, monitor, evaluate, reconcile
  loop.py         # tick() and run()
  agents.py       # pydantic-ai agent definitions and output models
  metrics.py      # response rate, breakpoint, extinction persistence, PE trace
  cli.py          # run an experiment from a config file
experiments/      # YAML configs: traits, schedule, ticks, seed, model
runs/             # gitignored outputs
tests/
```

## Build order

1. Scaffold: uv project, ruff, pytest, CI workflow, .gitignore, README with the research question.
2. `state.py` and `world.py` with tests. World fully deterministic under a seed.
3. End-to-end loop with every LLM phase stubbed by simple heuristics. It must run 100 ticks, write tick dumps, and compute metrics. Do not go further until this works.
4. Monitor and Reconcile with real logic and tests on the four channels.
5. Replace the Deliberate stub with a pydantic-ai agent. Then Appraise, then Evaluate.
6. Experiment runner: N agents with different traits across the three schedules, results to CSV, one plotting script.
7. Only then: episodic memory retrieval beyond "last K episodes".

Commit after each step with tests passing.

## Rules

- Memory in v0 is a list of episodes with salience, plus last-K retrieval. No decay curves, no semantic graph, no capacity limits until step 7. Over-building memory killed the previous attempt.
- Every LLM phase has a deterministic stub with the same signature, selectable by config, so the whole experiment can run with zero API calls.
- Every number that shapes behaviour (learning rates, thresholds) lives in the config or in Traits, never inline.
- Model name comes from config or an env var, never hard-coded.
- Keep phases pure where possible: take state in, return a new state or a delta.
- Ask before adding a dependency or a component not listed here.
