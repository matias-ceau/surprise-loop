# surprise-loop

An embodied LLM agent driven by prediction error. It has a simulated body it must keep in balance. Each tick it reads its world as that body lives it, says how it expects its body to feel after its next action, acts, and is corrected by what happens. The size of the miss is the surprise.

This file is the brief. The full design is `docs/superpowers/specs/2026-11-17-lived-twice-design.md`; the build plan for the first milestone is `docs/superpowers/plans/2026-11-17-lived-twice.md`. Where they say more than this file, they hold.

The first version of this project (a lever-pressing world, "v0") is at commit `2ec53fd`. Its result with a real model was "not yet": trait numbers in the prompt barely reached behaviour. This brief replaces it.

## Research question

Do bodies that differ in state and temperament come to perceive, predict and act differently in the same objective world, and does it matter how the model is told its body?

Testbed: one small forest clearing with three places (meadow, oak, stream), a daytime clock, and a body with three numbers (heat, water, energy). Two bodies live the same seeded day, each in its own copy of the world, and are watched side by side. Measure per body: time outside its comfortable range, the surprise trace, how often its expectations held, and how much of its behaviour a ten-line rule predicts.

A test on one frozen scene has already shown that a body told to the model as numbers barely moves its choice, while a body told in words does. So body state reaches the model in words only.

Everything in the architecture exists to serve this. Do not add a component it does not need. This is faked embodiment, a working analogue; nothing here licenses the claim that an agent feels.

## Cognitive cycle

One tick = one pass through seven phases. Each phase is a plain async function over a shared typed state.

1. Perceive (code): start from the belief about the body left by the last tick. Run the lens: plain code that turns the world's facts plus this body's belief into a felt scene (which facts this body notices, what each is worth to it, and the words for that).
2. Appraise (LLM): one line of the agent's own reading of the scene, and what the moment is about (heat, thirst, tiredness, nothing). The reading is kept and steers the next tick's attention.
3. Deliberate (LLM): choose one of the offered actions AND commit an expectation: how each body channel will feel afterwards (a phrase from a fixed ladder), and, where the action will settle an open fact about the world, a yes or no.
4. Act: send the action to the World; the clock and the true body move.
5. Monitor (code): three comparisons. Body signal against the expected band, per channel. The world expectation against the fact. Better or worse for the body than expected.
6. Evaluate (code for now): a regulation check. Did an action taken a few ticks ago ease the need it serves, now that its effect has arrived? Returns feedback only, never mutates state.
7. Reconcile (code): apply everything through four channels, each with its own timescale:
   - feeling (fast): the belief about the body, the tone of feeling, arousal
   - memory (per tick): store the episode, weighted by the size of the surprise
   - pulls (slow): a gut pull per place and action, from Evaluate's judgements
   - temperament (slowest): set points, how much a deviation is minded, attention habits. Switched off in the first milestone.

   If the body's loudest need is urgent, came out clearly worse than expected, and the action did not serve it, Reconcile may send control back to Deliberate, at most once per tick, with the choices cut to actions that serve that need.

## Boundaries

- The World is the only owner of environment state, and that includes each body's true numbers. It is deterministic Python with a seeded RNG, not an LLM.
- The agent owns its belief about its body, its temperament, feeling, mood, memory and pulls. It receives only a noisy body signal, the facts of the place it is in, and the actions on offer.
- The model is never shown a body number, a temperament number, a clock time or a variable name. Nothing it reads says game, simulation or experiment.
- Evaluate gives feedback and nothing else.

## Stack

- Python 3.12+, managed with uv
- pydantic v2 for all state models
- pydantic-ai (current 2.x) for the LLM-backed phases, with structured outputs
- pytest; pydantic-ai's test model for LLM-free tests
- ruff for lint and format
- The viewer is exported, self-contained HTML written by the standard library. No web framework, no JavaScript library.

Read the installed pydantic-ai source before writing agent code. Do not rely on memory of the 0.x or 1.x API. Do not use `pydantic_graph.persistence`.

Orchestration is a plain Python loop calling the seven phase functions.

State persistence: dump the full state model to JSON after every tick into `runs/<run_id>/tick_<n>.json`. A run must be resumable and replayable from these files.

## Layout

```
src/surprise_loop/
  world.py        # World protocol + ClearingWorld (places, clock, facts, true body, signal)
  state.py        # Temperament, Feeling, Expectation, Comparison, FeltScene, Memory, AgentState
  feeling.py      # the definitions of felt strength, discomfort, tone and arousal, in one place
  lens.py         # facts + belief -> FeltScene
  phrases.py      # every string a model can read
  config.py       # every number that shapes behaviour, the experiment files' schema, the seeded RNG
  phases/         # perceive, appraise, deliberate, act, monitor, evaluate, reconcile;
                  # heuristics.py holds the rules the code phases and the stubs share
  loop.py         # tick() and run(), tick files, resume, replay
  agents.py       # pydantic-ai agent definitions and output models
  ladder.py       # the frozen-noon sweep that checks the phrase table against a model
  metrics.py      # comfort time, surprise trace, expectation hit rates, rule agreement
  viewer/         # exported HTML: the lens page and the film of two runs
  cli.py          # run, experiment, metrics, replay, lens-page, ladder-check, scene
experiments/      # YAML configs: bodies, world, seeds, backends, model
runs/             # gitignored outputs
tests/
```

## Build order (first milestone)

1. The lens and its phrase table, with the forest example as fixed text. The lens page.
2. The ladder check on the lens as built, against a real model. If it fails, fix the wording before building on it.
3. The world and the body, with tests. World fully deterministic under a seed.
4. State, stubs, the loop on stubs, tick files, resume and replay, metrics. Defaults tuned until a stub day is alive: expectations can fail, and looking ahead pays.
5. The film: two stored runs as one exported page.
6. Appraise and Deliberate on a model. One model-driven afternoon for two bodies.
7. The README: what was shown, and what was not.

Commit after each step with tests passing.

## Rules

- Memory is a list of episodes with salience, plus last-K retrieval. No decay curves, no semantic graph, no capacity limits. Over-building memory killed an earlier attempt.
- Every LLM phase has a deterministic stub with the same signature, selectable by config, so the whole experiment can run with zero API calls.
- Every number that shapes behaviour (rates, thresholds, set points) lives in the config or in a body's temperament, never inline.
- Model name comes from config or an env var, never hard-coded.
- Keep phases pure where possible: take state in, return a new state or a delta.
- A claim about the model is reported as model run minus stub run on the same seeds, and next to the share of its choices a ten-line rule predicts. Differences that survive a chooser that reads nothing come from the update code and are not findings.
- Ask before adding a dependency or a component not listed here.
