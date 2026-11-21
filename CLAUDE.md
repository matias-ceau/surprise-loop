# surprise-loop

A creature that learns a hidden world through what its body feels. Each tick it says what it expects to feel after its next step, takes the step, and is corrected by what happens. The size of the miss is the surprise, and the surprise is what redraws its map of the world.

This file is the brief. The full design is `docs/superpowers/specs/2026-11-21-two-maps-design.md`; the build plan is `docs/superpowers/plans/2026-11-21-two-maps.md`. Where they say more than this file, they hold.

## What came before

- **The lever world** (commit `2ec53fd`). An agent pressing levers for reward. Its result with a real model was "not yet": trait numbers in the prompt barely reached behaviour.
- **The clearing** (commit `24aff96`). Two bodies in a forest clearing, told their state in sentences. It showed that a body told to a model in words moves its choice where numbers do not. The owner judged it too small: three inputs, and a surprise that built nothing lasting about the world.

## Research question

When creatures with different temperaments explore the same hidden world, and learn about it only through what their bodies feel, do they end up with different pictures of it? And how much of that difference is the model's own doing, as opposed to what careful arithmetic on the same feelings would give?

Testbed: a small grid of tiles the creature cannot see. Some tiles are ground, some pleasant or unpleasant to arrive on, some stop it: obstacles, food, water, and food and water that bring sickness a few steps later. The creature tries to step north, east, south or west. The world answers only with whether the step succeeded and what the body now feels. It never names a tile.

Each creature keeps a map with five beliefs per tile: it blocks me, it hurts, it feeds, it quenches, it pleases. Code keeps a second map for each creature: what a careful reasoner concludes from the very same feelings. The result is a film of three maps side by side, the world in the middle and a creature's belief on each side, drawn with one colour rule.

Everything in the architecture exists to serve this. Do not add a component it does not need. This is faked embodiment, a working analogue; nothing here licenses the claim that a creature feels.

## Cognitive cycle

One tick is one pass through seven phases. Each phase is a plain async function over a shared typed state.

1. Perceive (code): gather what the creature has to go on: its latest felt record, its own map, and its last few steps with what it expected, what it felt and what it made of it.
2. Deliberate (model): choose one of the four directions and commit to an expectation: whether it will move, and a word for each feeling, for smell and for damp.
3. Act (world): the body's true numbers move.
4. Monitor (code): pass the true numbers through the creature's temperament to get what it feels, in words. Compare with the expectation, field by field: that is the surprise. Set the summary feeling, tone and stir.
5. Evaluate (code): the careful reasoner updates its own map from the felt record. It changes nothing of the creature's.
6. Appraise (model): shown what it expected and what it felt, the creature says in one line what it makes of it and lists the tiles whose beliefs it now changes.
7. Reconcile (code): apply the changes to the creature's own map, count the try, store the step, and take the tick's measures.

What changes at each pace: tone and stir fast; the creature's map and its short memory every tick; temperament not at all in this milestone.

## Temperament

How one creature's senses and presumptions differ from another's. Three parts, all numbers in the experiment file:

- presumption: what an untried tile is presumed to be. The creature's map starts from it.
- loudness: a gain on each body feeling.
- clarity: noise on each feeling, and how far neighbouring feelings blur into each other.

Temperament is applied by code to what the creature feels. It is never written to the model as a personality.

## Boundaries

- The World is the only owner of the layout and of each body's true numbers. It is deterministic Python with a seeded RNG, not an LLM.
- The creature receives only the felt record: whether it moved, a word for each body feeling, smell, damp, tone, stir, and where it is in steps from its start.
- The model is never shown a true body number, a temperament's loudness or clarity, or what kind a tile is. Nothing it reads says game, simulation or experiment. Feelings are words; beliefs about tiles are numbers.
- Nothing of the careful reasoner is ever shown to the model.
- Evaluate changes only the reasoner's own map.

## Stack

- Python 3.12+, managed with uv
- pydantic v2 for all state models
- pydantic-ai (current 2.x) for the model-backed phases, with structured outputs
- pytest; pydantic-ai's test and function models for model-free tests
- ruff for lint and format
- The viewer is exported, self-contained HTML written by the standard library. No web framework, no JavaScript library.

Read the installed pydantic-ai source before writing agent code. Do not rely on memory of the 0.x or 1.x API. Do not use `pydantic_graph.persistence`.

Orchestration is a plain Python loop calling the seven phase functions.

State persistence: dump the full state model to JSON after every tick into `runs/<run_id>/tick_<n>.json`. A run must be resumable and replayable from these files.

## Layout

```
src/surprise_loop/
  world.py        # the grid, the eight kinds of tile, the body's true numbers, the seeded RNG
  state.py        # Temperament, FeltRecord, Expectation, Comparison, the two maps, AgentState, tick records
  lens.py         # true numbers -> what this creature feels, through its temperament
  feeling.py      # tone and stir
  phrases.py      # the word ladders and every fixed text a model reads
  config.py       # every number that shapes behaviour, the experiment files' schema
  reasoner.py     # the careful reasoner: what follows from the felt record alone
  phases/         # perceive, deliberate, act, monitor, evaluate, appraise, reconcile;
                  # heuristics.py holds the rule chooser, the blind chooser and the measures
  loop.py         # tick() and run(), tick files, resume, replay
  agents.py       # pydantic-ai agent definitions, prompts and output models
  metrics.py      # comfort, surprise, map error, the gap between the two maps, distances between runs
  viewer/         # exported HTML: the film of three maps
  cli.py          # run, experiment, metrics, replay, gate2, film
experiments/      # YAML configs: layout, temperaments, seeds, backends, model
runs/             # gitignored outputs
tests/
```

## Build order

1. This brief; the clearing's code retired.
2. The world, the body and the felt record. Tests. The world fully deterministic under a seed.
3. The maps, the careful reasoner, the stand-ins, the loop, tick files, resume and replay, measures. Gate 1: on stand-ins, expectations can fail, the reasoner learns, looking pays, and temperament matters to the arithmetic.
4. Deliberate and Appraise on a model. Gate 2: can the model keep a map at all? The film is not built before this.
5. The film and its key, built on stored runs.
6. The results and the README: what was shown, and what was not.

Commit after each step with tests passing.

## Rules

- Memory is the last few episodes. No decay curves, no semantic graph. Over-building memory killed an earlier attempt.
- Every model phase has a deterministic stand-in with the same signature, selectable by config, so the whole experiment can run with zero API calls: a rule chooser and a blind chooser for Deliberate, the reasoner's appraiser for Appraise.
- Every number that shapes behaviour lives in the config or in a temperament, never inline.
- Model name comes from config or an env var, never hard-coded.
- Keep phases pure where possible: take state in, return a new state.
- A claim about the model is reported as the model run set against the stand-in run on the same seeds, and next to the share of its choices the rule chooser would also have made. A difference between temperaments that the blind chooser shows too comes from the code and is not a finding.
- A failed model call stops the run. There is no silent fallback to a stand-in.
- Ask before adding a dependency or a component not listed here.
