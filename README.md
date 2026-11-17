# surprise-loop

An embodied LLM agent driven by prediction error. It has a simulated body it must keep in
balance. Each tick it reads its world as that body lives it, says how it expects its body to feel
after its next action, acts, and is corrected by what happens. The size of the miss is the
surprise.

**Status: being rebuilt.** The project was redirected on 2026-11-17. The tree currently holds the
new brief, the design and the build plan; the code lands task by task.

- The brief: [`CLAUDE.md`](CLAUDE.md)
- The design, with the test that motivated it:
  [`docs/superpowers/specs/2026-11-17-lived-twice-design.md`](docs/superpowers/specs/2026-11-17-lived-twice-design.md)
- The build plan:
  [`docs/superpowers/plans/2026-11-17-lived-twice.md`](docs/superpowers/plans/2026-11-17-lived-twice.md)

## Research question

Do bodies that differ in state and temperament come to perceive, predict and act differently in
the same objective world, and does it matter how the model is told its body?

## The first version (v0)

v0 was a lever-pressing world: an agent pressed levers for reward under changing schedules, and
the question was whether agents with different trait numbers showed stable differences in
behaviour. All of it is at commit `2ec53fd`:

    git show 2ec53fd:README.md      # the full v0 write-up, with commands and results
    git checkout 2ec53fd            # the v0 code and its 741 tests

Its results in one paragraph. With hand-written rules in place of the model, trait profiles
separated cleanly (stability across seeds between 0.82 and 0.98 on the four measures). With a
real model deciding, they mostly did not (0.60 for breakpoint, about 0.3 for response rates,
−0.09 for persistence in extinction), and the measures that stayed stable were made by the
deterministic update code, not by the model's choices. The answer was "not yet". A later test on
one frozen scene suggests why: a state given to the model as numbers barely moves its choice,
while the same state given in words does. That test is in section 6 of the design.

## Install and test

    uv sync --all-groups
    uv run ruff check . && uv run ruff format --check . && uv run pytest
