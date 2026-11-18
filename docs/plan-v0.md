# surprise-loop v0 — implementation plan

> **Kept for history.** This is the plan of the first version, the lever-pressing world. Its code is
> no longer in the tree: it is at commit `2ec53fd`, and the commands below name files that exist only
> there. The current plan is `docs/superpowers/plans/2026-11-17-lived-twice.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. One fresh implementer per task, one task review after each, a whole-branch review at the end. Steps use checkbox (`- [ ]`) syntax.

**Goal:** all seven build-order steps of the brief implemented, tested, committed and pushed, with a draft PR and green CI, a stub-only experiment that runs end to end with zero API calls, and one small real-model experiment through OpenRouter.

**Architecture:** a plain async loop calls seven phase functions over one frozen `Workspace`; `AgentState` (mood, memory, goals, dispositions, traits) is changed only by Reconcile, `WorldState` only by `World.step`. The three LLM phases each have a deterministic stub and a pydantic-ai variant with the same signature, selected by config. Full state is dumped to JSON after every tick; metrics are pure functions over those dumps.

**Tech stack:** Python ≥3.12, uv, pydantic v2, pydantic-ai 2.x (latest 2.68.0), pyyaml, pytest, ruff; matplotlib in an optional `plot` group.

**Spec:** `/home/user/surprise-loop/CLAUDE.md` (the brief). It is the authority; this plan argues from it.

## Context

The repo holds only the brief and a one-line README. The brief asks for a prediction-error-driven agent in a self-administration world, to answer: *does such an agent show stable individual differences in reinforcement behaviour?* You asked me to plan it, then drive Opus subagents through as much of the build as possible without further input.

The brief leaves open the contracts between modules, the maths of the deterministic phases, and how re-entry interacts with world time. This plan fixes those so that independent subagents produce code that fits together. An independent reviewer simulated the maths below on all three schedules before this plan was finalised; four of its findings changed the design and are marked ★.

## Decided by you (answers of 2026-11-14)

- **Re-entry is a second action in the same tick.** When Reconcile flags a conflict, the loop runs Deliberate → Act → Monitor → Evaluate → Reconcile once more. A tick holds one or two *passes*; each pass is one world step. Schedule lengths and metrics count **world steps**; `ticks` counts cognitive cycles.
- **Dependencies:** `pyyaml` (runtime) and `matplotlib` (optional `plot` group).
- **Step 7:** minimal salience-based retrieval behind a config flag, off by default.
- **Real-model verification:** OpenRouter key, model `glm-5.3-flash` (OpenRouter id `z-ai/glm-5.3-flash`, pydantic-ai name `openrouter:z-ai/glm-5.3-flash`; I confirmed it is listed and supports tools and structured output).

## My calls (not yet agreed by you; say so if any is wrong)

1. ★ **The default schedule is FR → PR → FR (re-baseline) → extinction.** Run literally as FR → PR → extinction, the agent has already stopped being rewarded by the end of PR, so extinction starts from an extinguished state and persistence measures almost nothing (simulated: about 5 presses, tracking PR length rather than the agent). A short FR phase before extinction fixes that. The schedule is a list in YAML, so the literal three-phase version is a one-line change.
2. ★ **Breakpoint and extinction persistence use a quit criterion** (a run of `quit_steps` steps with no active-lever press) and carry a `censored` flag when the phase ended first. Without it, breakpoint is just a function of PR phase length.
3. ★ **Mood inertia acts through arousal.** Arousal (driven by unsigned prediction error) adds vigor to pressing; valence biases approach. With valence alone, mood inertia had no measurable effect on any metric, because signed error averages to zero.
4. ★ **The prior on expected reward fades as evidence accumulates, and a goal's error average resets when the goal is revised.** Otherwise goal revision either never fires or ratchets to its limit.
5. **One conflict rule: habit versus goal.** Conflict = on this pass the goals channel *lowered* the priority of the goal the action serves, while the disposition bias for that action is still above a threshold. On re-entry Deliberate is told, and the stub ignores dispositions for that pass.
6. **No fifth mental-state component for learned values.** The stub Deliberate predicts reward from recalled episodes only, recency-weighted by `surprise_learning_rate`.
7. **Confidence is used as precision.** Prediction error is scaled by `floor + (1 − floor)·confidence`.
8. **Three goals:** `seek_reward` (both presses), `explore`, `conserve` (rest). Revision moves the priority of the goal the action served. In stub runs only `seek_reward` is ever revised, because explore and rest have no reward to mispredict.
9. **A run ends when the schedule ends.** `ticks` in the config is an optional cap (the 100-tick smoke run uses it); a capped run is marked `truncated`. This gives every agent the same exposure in world steps even though re-entry makes ticks and steps differ.
10. **Replay means:** re-execute the deterministic phases against the recorded outputs of the three LLM phases and check every rebuilt dump equals the stored one. Works for stub and LLM runs alike.
11. **Randomness without stored RNG state.** World and stub draws use a generator seeded from a string of `(seed, …key)`, so dumps stay plain data and resume is exact across processes.
12. **LLM failure stops the run.** No silent fallback to the stub; the run resumes from the last tick dump.
13. **Two files beyond the brief's layout:** `config.py` and `phases/heuristics.py` (helpers shared by the stub phases). The experiment sweep lives in `cli.py`, dump/load/replay in `loop.py`, the plot script in `scripts/`.
14. **"CoreAgent" is `AgentState` plus the phase functions**, not a separate class. **Attention in v0 is a fixed filter:** the percept carries event, reward and last action (not the world's step counter) plus the `retrieval_k` recalled episodes.
15. **Stability statistic:** ICC(1) of each metric across seeds within trait profile, one pure function in `metrics.py`.
16. **Run directories follow the brief literally:** `runs/<run_id>/tick_<n>.json`, `run_id = <experiment>-<agent>-s<seed>`, `<n>` zero-padded to six digits. Experiment-level CSVs and plots go in `runs/<experiment>/`.
17. **Async tests use `asyncio.run`**, no pytest plugin. **This plan is committed as `docs/plan-v0.md`.**

## Global constraints (every task inherits these)

- Every behaviour-shaping number lives in `config.py` defaults / YAML or in `Traits`. No numeric literals in phase code other than structural 0, 1, −1 and clamps to documented ranges.
- Model name only from config `model` or env `SURPRISE_LOOP_MODEL`. Never hard-coded.
- Mental state in `AgentState` is changed only by Reconcile (`tick()` alone advances the `tick` counter). World state only by `World.step`. Evaluate returns `Feedback` and nothing else.
- The agent never sees `StepInfo` (phase kind, ratio, which lever is active). Only `Observation` crosses Perceive.
- Memory is a list of episodes plus last-K retrieval until Task 15. No decay, graph, or capacity limit.
- Every LLM phase has a stub with the same signature, chosen by config. The full experiment must run with zero API calls.
- All models are pydantic v2 with `extra="forbid"` and `allow_inf_nan=False`. Every phase is `async def phase(ws: Workspace, rt: Runtime) -> Workspace` and returns a copy; the input is never mutated.
- All randomness goes through `derive_rng`. No `hash()`, no global `random`, no wall-clock or timestamps in dumps.
- pydantic-ai: read the current docs and the installed source before writing agent code. No 0.x/1.x idioms, no `pydantic_graph.persistence`.
- No new dependency or component beyond this plan.
- `uv run ruff check . && uv run ruff format --check . && uv run pytest` green before every commit. Tests make no network calls.

## Contracts

### `world.py`

```python
class Action(StrEnum):    PRESS_A="press_a"; PRESS_B="press_b"; EXPLORE="explore"; REST="rest"
class Event(StrEnum):     START="start"; REWARD="reward"; NO_REWARD="no_reward"; EXPLORED="explored"; RESTED="rested"
class PhaseKind(StrEnum): FIXED_RATIO="fixed_ratio"; PROGRESSIVE_RATIO="progressive_ratio"; EXTINCTION="extinction"

class Observation: step: int; last_action: Action | None; event: Event; reward: float
class StepInfo:    step: int; phase_index: int; phase_kind: PhaseKind; ratio_required: int | None
                   active_press: bool; ratio_completed: int | None      # set on a reinforced press
class Outcome:     observation: Observation; info: StepInfo
class PhaseSpec:   kind: PhaseKind; steps: int; ratio: int | None; pr_start: int | None; pr_increment: int | None
class WorldConfig: active_lever: Literal["press_a","press_b","random"]; reward_magnitude: float; schedule: list[PhaseSpec]
class WorldState:  seed: int; step: int; active_lever: Action; phase_index: int; step_in_phase: int
                   presses_toward_ratio: int; reinforcers_in_phase: int; last_observation: Observation

class WorldDone(RuntimeError)
class World(Protocol):
    @property
    def done(self) -> bool: ...
    def observe(self) -> Observation: ...          # START (step 0) before any step; the last observation afterwards
    def step(self, action: Action) -> Outcome: ...  # raises WorldDone once the schedule has ended
    def snapshot(self) -> WorldState: ...
class SelfAdminWorld:    # __init__(config: WorldConfig, seed: int); classmethod restore(config, state) -> SelfAdminWorld
```

Steps are 1-based. `Observation.step` is the number of steps taken so far. `StepInfo` describes the step in which the action was taken (its phase, the requirement in force for that press). FR: every `ratio`-th active press is rewarded. PR: the requirement is `pr_start + n·pr_increment` after `n` reinforcers in the phase. Extinction: never rewarded. `presses_toward_ratio` and `reinforcers_in_phase` reset on phase change. Inactive lever, explore and rest never reward. `PhaseSpec` validates that each kind carries exactly its own parameters. The only random draw is `active_lever="random"`, made from the seed.

### `state.py`

```python
class GoalName(StrEnum): SEEK_REWARD="seek_reward"; EXPLORE="explore"; CONSERVE="conserve"
GOAL_OF_ACTION: dict[Action, GoalName]   # presses -> SEEK_REWARD, explore -> EXPLORE, rest -> CONSERVE

class Traits(frozen):   reward_sensitivity: float (>0); surprise_learning_rate: float (0,1]; mood_inertia: float [0,1)
class Mood:             valence: float [-1,1]; arousal: float [0,1]
class Prediction:       expected_event: Event; expected_reward: float; confidence: float [0,1]
class PredictionError:  raw_signed: float; signed: float; unsigned: float; event_mismatch: bool
class Appraisal:        valence: float [-1,1]; arousal: float [0,1]; note: str
class Decision:         action: Action; prediction: Prediction; rationale: str
class Feedback:         alignment: float [-1,1]; critique: str
class Conflict:         kind: Literal["habit_vs_goal"]; action: Action; detail: str
class Episode:          tick; step; action; prediction; event; reward; prediction_error; alignment; critique; salience [0,1]
class Memory:           episodes: list[Episode];  last_k(k: int) -> list[Episode]
class Goals:            priorities: dict[GoalName, float]; error_ema: dict[GoalName, float]
                        error_streak: dict[GoalName, int]   # signed: +n / −n consecutive over-threshold passes of that sign
                        revisions: int
class Dispositions:     biases: dict[Action, float]
class AgentState:       traits; mood; memory; goals; dispositions; tick: int      # tick = completed ticks

class Percept:          event; reward; last_action; recalled: list[Episode]
class PassRecord:       tick; pass_index; decision; outcome: Outcome; prediction_error; feedback
                        goal_revised: GoalName | None; conflict: Conflict | None
class TickRecord:       tick; observation; appraisal; passes: list[PassRecord]    # length 1 or 2
class TickDump:         run_id; tick; agent: AgentState; world: WorldState; record: TickRecord
class Workspace(frozen): agent; observation; pass_index: int = 0; percept; appraisal; conflict; decision; outcome
                        prediction_error; feedback; passes: tuple[PassRecord, ...] = ()
```

### `config.py`

`Params` groups, each field with its default here and overridable from YAML:
`monitor` (confidence_floor, event_mismatch_weight) · `mood` (arousal_gain) · `memory` (retrieval_k, base_salience) · `goals` (ema_rate, error_threshold, patience, revision_step, min_priority, max_priority) · `dispositions` (rate, bound, alignment_weight) · `conflict` (enabled, habit_threshold) · `deliberate` (temperature, prior_expected_reward, prior_weight > 0, action_costs, approach_weight, vigor_weight, explore_value, rest_value, confidence_halfsat) · `appraise` (reward_gain) · `evaluate` (scale) · `metrics` (quit_steps).

```python
class Backends:         appraise | deliberate | evaluate: Literal["stub", "llm"]
class AgentSpec:        name: str; traits: Traits; mood: Mood; goal_priorities: dict[GoalName, float]
class RunConfig:        run_id: str; seed: int; ticks: int | None; world: WorldConfig; agent: AgentSpec
                        params: Params; backends: Backends; model: str | None
class ExperimentConfig: name; seeds: list[int]; ticks: int | None; world; agents: list[AgentSpec]; params; backends; model
                        def runs(self) -> list[RunConfig]         # agents × seeds; run_id = f"{name}-{agent.name}-s{seed}"
class ConfigError(ValueError)
def load_config(path: Path) -> ExperimentConfig
def resolve_model(config: RunConfig) -> str | None   # config.model, else env SURPRISE_LOOP_MODEL;
                                                     # ConfigError if any backend is "llm" and neither is set
def initial_agent_state(config: RunConfig) -> AgentState
def derive_rng(seed: int, *key: int | str) -> random.Random   # random.Random(":".join(map(str, (seed, *key))))

@dataclass(frozen=True)
class Runtime: config: RunConfig; world: World; model: object | None = None   # resolved pydantic-ai Model when a backend is "llm"
```

### `phases/` and `loop.py`

```python
# phases/<name>.py each export  async def <name>(ws: Workspace, rt: Runtime) -> Workspace
# appraise / deliberate / evaluate also export <name>_stub and (from Tasks 10-12) <name>_llm; <name> dispatches on rt.config.backends
# phases/perceive.py also exports  build_percept(observation, agent, params) -> Percept

class RunResult: config: RunConfig; records: list[TickRecord]; agent: AgentState; world: WorldState
class ReplayMismatch(RuntimeError)                         # names the first differing tick
async def tick(agent: AgentState, rt: Runtime) -> tuple[AgentState, TickRecord]
async def run(config: RunConfig, run_dir: Path | None = None, *, resume: bool = False,
              stop_after_tick: int | None = None) -> RunResult
async def replay(run_dir: Path) -> int                      # ticks verified
def write_dump(run_dir: Path, dump: TickDump) -> Path       # tick_000001.json; temp file then atomic rename
def load_dumps(run_dir: Path) -> list[TickDump]              # globs tick_*.json only, sorted by tick
def load_result(run_dir: Path) -> RunResult
```

`tick`: ticks are 1-based; the tick in progress is `agent.tick + 1`. Perceive, Appraise, then pass 0 = Deliberate, Act, Monitor, Evaluate, Reconcile. If Reconcile set `ws.conflict` and the world is not done, `tick` builds the pass-1 workspace (`pass_index=1`, the pass-0 `conflict` kept, `observation = outcome.observation`, `percept = build_percept(...)` on the updated agent so it includes the pass-0 episode, `appraisal` kept, `decision/outcome/prediction_error/feedback = None`, `passes = (pass0,)`) and runs Deliberate → Reconcile once more. Reconcile on pass 1 never sets a conflict. Reconcile never changes `tick`; `tick()` sets `agent.tick` after the last pass.
`run`: builds or restores world and agent; before each tick, stops if `world.done`, or `config.ticks` ticks are complete, or `stop_after_tick` is reached. One dump per tick when `run_dir` is given. Start: refuse a non-empty `run_dir` unless `resume=True`; write `config.json` containing the `RunConfig` with the resolved model name. Resume: `config.json` must equal the given config (with its model resolved); restore from the last dump and continue. `RunResult.records` includes the ticks loaded on resume. A tick that died half-way left no dump and is simply run again.
`replay`: from `config.json`, re-run every tick with the three LLM-backed phases replaced by the recorded `appraisal`, `decision` and `feedback`, and compare each rebuilt `TickDump` with the stored one.

### Deterministic maths

Traits: `RS` reward_sensitivity, `α` surprise_learning_rate, `m` mood_inertia. `a` is the action taken, `g = GOAL_OF_ACTION[a]`.

- **Monitor.** `raw = reward − expected_reward`; `w = floor + (1 − floor)·confidence`; `signed = w·raw`; `unsigned = w·(|raw| + event_mismatch_weight·[event ≠ expected_event])`.
- **Mood (fast).** `valence ← m·valence + (1 − m)·tanh(RS·signed)`; `arousal ← m·arousal + (1 − m)·min(1, arousal_gain·unsigned)`.
- **Memory (per pass).** Append an `Episode`; `salience = min(1, base_salience + unsigned)`. The critique text is stored here; that is how textual feedback reaches later deliberation.
- **Goals (slow).** `ema[g] ← (1 − β)·ema[g] + β·signed`, `β = ema_rate·α`. Let `s = sign(ema[g])` if `|ema[g]| > error_threshold`, else 0. If `s = 0`: `streak[g] = 0`. If `streak[g]` has sign `s`: `streak[g] += s`. Otherwise `streak[g] = s`. When `|streak[g]|` reaches `patience`: `priority[g] += revision_step·s`, clamped to `[min_priority, max_priority]`; `streak[g] = 0`; `ema[g] = 0`; `revisions += 1`; `goal_revised = g`. Passes serving another goal leave `g` untouched.
- **Dispositions (slowest).** `v = RS·reward + alignment_weight·alignment`; `bias[a] ← clamp(bias[a] + rate·α·(v − bias[a]), ±bound)`. Only `bias[a]` moves.
- **Conflict.** On pass 0, with `conflict.enabled`: this pass revised `g` downward (`s < 0`) and `bias[a] ≥ habit_threshold` → `Conflict("habit_vs_goal", a, …)`. Otherwise `None`.
- **Stub Deliberate.** For a press `x` with `n` recalled episodes of `x`, rewards `rᵢ`, `wᵢ = (1 − α)^i` where `i = 0` is the most recent: `E[x] = (p·P + Σ wᵢ·rᵢ) / (p + Σ wᵢ)` with `P = prior_expected_reward` and `p = prior_weight·(1 − α)^n`. Utilities: press `priority[seek_reward]·RS·E[x] − cost[x] + approach_weight·appraisal.valence + vigor_weight·appraisal.arousal`; explore `priority[explore]·explore_value − cost`; rest `priority[conserve]·rest_value − cost`; each plus `bias[x]` unless `ws.conflict` is set. Sample by softmax at `temperature` using `derive_rng(config.seed, tick, pass_index)`. Prediction: `expected_reward = E[x]` for presses and 0 otherwise; `expected_event` = most frequent recalled event for that action (with none recalled: `NO_REWARD` for presses, `EXPLORED`, `RESTED`); `confidence = n / (n + confidence_halfsat)`.
- **Stub Appraise.** `valence = clamp(mood.valence + reward_gain·RS·priority[seek_reward]·observation.reward, −1, 1)`; `arousal = mood.arousal`.
- **Stub Evaluate.** `alignment = tanh((U(a) − mean_b U(b)) / scale)` where `U` is the utility above without the bias, valence and arousal terms.

### `metrics.py` (pure; `passes` = flattened `PassRecord`s in step order)

```python
def flatten(records: list[TickRecord]) -> list[PassRecord]
class PhaseMetrics: phase_index; phase_kind; steps; active_presses; response_rate; rewards; mean_abs_pe
def phase_table(passes) -> list[PhaseMetrics]                    # one row per phase the run reached
def response_rate(passes, phase_index: int | None = None) -> float   # active presses per step; 0.0 if no steps
class Censored: value: int; censored: bool
def breakpoint(passes, quit_steps: int) -> Censored            # last ratio_completed in the PR phase before the quit criterion
def extinction_persistence(passes, quit_steps: int) -> Censored # active presses in extinction before the quit criterion
def pe_trace(passes) -> list[PEPoint]                           # step, tick, pass_index, phase_index, phase_kind, action, active_press, raw_signed, signed, unsigned
def summarize(result: RunResult) -> RunMetrics                  # flat, CSV-ready
def icc1(groups: list[list[float]]) -> float                    # one-way ICC; nan if undefined or zero total variance
```

Quit criterion: the first run of `quit_steps` consecutive steps within the phase with no active press; `censored=True` when the phase ended first or was never reached. `RunMetrics`: steps, ticks, truncated, rewards, response_rate_fr (FR phases before the first PR phase), response_rate_pr, response_rate_ext, inactive_rate, breakpoint, breakpoint_censored, extinction_persistence, extinction_censored, mean_abs_pe, reentries, goal_revisions.

## Review focus (conditions the brief implies that are easy to miss)

1. **Interrupted run.** A crash mid-tick leaves no partial tick file; a resumed stub run is byte-identical to an uninterrupted one, including when ticks with two passes fall on both sides of the cut (Tasks 5, 8).
2. **Agent never earns a reward, or the run ends before a phase starts.** Every metric returns a defined value; no division by zero (Task 6).
3. **LLM returns out-of-range or malformed output.** Validation, bounded retry, then a loud stop; never a silent stub substitution (Tasks 10–12).
4. **Hidden world information leaking into percepts or prompts.** Asserted absent (Tasks 4, 10–12).
5. **Misconfiguration.** LLM backend with no model, unknown YAML key, non-empty run directory, resume with a different config or model: all fail before tick 1 with a message naming the problem (Tasks 3, 5, 7).

Known cost, accepted because the brief requires full state per tick: each dump contains all episodes so far, so run size grows with the square of run length. Default experiments stay in the low hundreds of steps.

## Tasks

Strictly sequential. Each task ends with ruff and pytest green and a commit; I push after every task.

### Task 1 — Scaffold (build step 1)
Files: `pyproject.toml`, `uv.lock`, `.gitignore`, `.github/workflows/ci.yml`, `README.md`, `docs/plan-v0.md`, `src/surprise_loop/__init__.py`, `tests/test_smoke.py`.
- [ ] src-layout uv project `surprise-loop`, `requires-python = ">=3.12"`; dependencies `pydantic>=2`, `pydantic-ai>=2`, `pyyaml`; dependency groups `dev = [pytest, ruff]`, `plot = [matplotlib]`; script `surprise-loop = "surprise_loop.cli:main"`; ruff configured (line length 100, target py312, rules E, F, I, UP, B, SIM).
- [ ] `.gitignore`: `runs/`, `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `.superpowers/`.
- [ ] CI: on push and pull_request; `astral-sh/setup-uv`; matrix Python 3.12 and 3.13; `uv sync --locked --all-groups`, `ruff check`, `ruff format --check`, `pytest`.
- [ ] README: research question, the seven-phase cycle, boundaries, how to install and test. `docs/plan-v0.md` is a verbatim copy of this plan.
- [ ] `tests/test_smoke.py::test_package_imports` imports `surprise_loop`. Commit `step 1: scaffold`.

### Task 2 — `world.py` (build step 2)
Files: `src/surprise_loop/world.py`, `tests/test_world.py`. Produces everything under Contracts → `world.py`.
- [ ] Tests first: FR5 rewards exactly active presses 5, 10, 15 and nothing else; the inactive lever, explore and rest never reward and return their own `Event`; PR with `pr_start=1, pr_increment=2` requires 1, 3, 5, 7 presses and reports each in `ratio_completed`; extinction never rewards; a phase change happens exactly after `steps` steps, resets `presses_toward_ratio`, and the `StepInfo` of the last step of a phase names that phase; `StepInfo.active_press` is true only for the active lever; the same seed and action script give an identical `Outcome` sequence; `active_lever="random"` depends only on the seed and takes both values across seeds 0–19; `snapshot()` then `restore()` mid-run continues identically to an uninterrupted world; `observe()` returns `START` with step 0 before any step and the last observation afterwards, including after the final step; `step()` after the schedule ends raises `WorldDone`; `PhaseSpec` rejects FR without `ratio`, PR without `pr_start`/`pr_increment`, extinction with any of them, and `steps < 1`.
- [ ] Implement. Commit `step 2: world`.

### Task 3 — `state.py` and `config.py` (build step 2)
Files: `src/surprise_loop/state.py`, `src/surprise_loop/config.py`, `tests/test_state.py`, `tests/test_config.py`. Consumes `world.py`. Produces Contracts → `state.py`, `config.py`.
- [ ] Tests first: every bounded field rejects out-of-range, NaN and infinite values; unknown fields are rejected; `AgentState` and `TickDump` survive a JSON round-trip unchanged; `Memory.last_k` returns the last k in order, all of them when k exceeds the length, and `[]` for an empty memory; `Workspace` is frozen; a YAML file loads to `ExperimentConfig` and `.runs()` yields agents × seeds with the documented `run_id`; an unknown YAML key raises `ConfigError` naming the key; `prior_weight ≤ 0` is rejected; `resolve_model` prefers config over env, returns `None` for all-stub configs, and raises `ConfigError` for an `llm` backend with neither set; `initial_agent_state` has empty memory, zero biases, ema and streak for every key, and the spec's mood and priorities; `derive_rng(7, 3, 0).random()` equals a hard-coded value and `derive_rng(7, 3, 0)` differs from `derive_rng(7, 3, 1)`.
- [ ] Implement, with a one-line comment on each `Params` default saying what it controls. Commit `step 2: state and config`.

### Task 4 — Stub phases (build step 3)
Files: `src/surprise_loop/phases/{__init__,perceive,appraise,deliberate,act,monitor,evaluate,reconcile,heuristics}.py`, `tests/test_phases_stub.py`.
Scope: the three stubs exactly as under Deterministic maths; Monitor sets `raw_signed`, and for now `signed = raw`, `unsigned = |raw|`; Reconcile only appends the `Episode` and the `PassRecord` and never sets a conflict (Task 8 replaces both).
- [ ] Tests first: each phase returns a new `Workspace` and leaves its input equal to a deep copy taken before the call; `Percept` has no `step` and no `StepInfo` field; with nothing recalled `E[x]` equals the prior and confidence is 0; under repeated reward 1 `E[x]` rises monotonically toward 1 and exceeds 0.95 within `retrieval_k` episodes for α = 0.3 and for α = 0.9; under repeated reward 0 it falls toward 0; stub Deliberate's draw is identical for the same `(seed, tick, pass_index)`; with `ws.conflict` set the bias term is omitted; Monitor on three hand cases including a negative error; Act is the only phase that calls `world.step`; Evaluate leaves `ws.agent` equal; stub Appraise and Evaluate outputs stay in range for extreme inputs.
- [ ] Implement. Commit `step 3: stub phases`.

### Task 5 — The loop, dumps, resume and replay (build step 3)
Files: `src/surprise_loop/loop.py`, `tests/test_loop.py`, `tests/test_persistence.py`. Produces Contracts → `loop.py`.
- [ ] Tests first: `tick` returns one pass and `agent.tick + 1`; `run` with `ticks=100` and a 200-step schedule returns 100 records; with `ticks=None` it stops when the world is done without raising; with `run_dir` it writes `tick_000001.json` … and `config.json`; a run with `stop_after_tick=40` then `resume=True` yields files byte-identical to an uninterrupted run, and `RunResult.records` covers all ticks; a stray `*.tmp` file is ignored on resume; a non-empty directory without `resume` raises; `resume` with a changed config raises; `replay` returns the tick count on an untouched run and raises `ReplayMismatch` naming the tick after one dump's `agent.mood` is edited.
- [ ] Implement. Commit `step 3: loop, dumps, resume, replay`.

### Task 6 — `metrics.py` (build step 3)
Files: `src/surprise_loop/metrics.py`, `tests/test_metrics.py`. Produces Contracts → `metrics.py`.
- [ ] Tests first, on hand-built pass lists: `phase_table` rows and response rates per `phase_index` with two FR phases kept separate; breakpoint equals the last `ratio_completed` before a run of `quit_steps` non-press steps, later completions ignored, `censored=True` when the criterion is never met; extinction persistence likewise; an agent that never presses gives rate 0.0, breakpoint 0, persistence 0; a run with no PR or no extinction phase gives value 0 with `censored=True` and does not raise; `pe_trace` has one point per pass in step order; `summarize` on a real 100-tick stub run matches values computed by hand from its records; `icc1` is 1.0 for groups with no within-group variance and distinct means, about 0 for identical group means, `nan` for a single group, single-member groups, or zero total variance.
- [ ] Implement. Commit `step 3: metrics`.

### Task 7 — CLI and the 100-tick end-to-end run (build step 3, gate)
Files: `src/surprise_loop/cli.py`, `experiments/stub_smoke.yaml` (`ticks: 100`, schedule totalling at least 200 steps, stub backends), `tests/test_cli.py`.
Produces: `main(argv: list[str] | None = None) -> int`; `surprise-loop run CONFIG [--agent NAME] [--seed N] [--runs-dir DIR] [--resume]` (defaults: first agent, first seed, `DIR = runs`; run directory `DIR/<run_id>`; prints `RunMetrics`); `surprise-loop metrics RUN_DIR`; `surprise-loop replay RUN_DIR`.
- [ ] Tests first: `run` on `stub_smoke.yaml` writes 100 tick files plus `config.json` and exits 0; `metrics RUN_DIR` prints the same values as the in-memory `summarize`; `replay RUN_DIR` exits 0; a non-empty run directory without `--resume`, `--resume` with a changed config, an unknown agent name, and a config with an unknown key each exit non-zero with a one-line message and no traceback.
- [ ] Implement. Commit `step 3: cli and end-to-end run`. **Nothing further until this passes.**

### Task 8 — Monitor and Reconcile: the four channels and bounded re-entry (build step 4)
Files: `phases/monitor.py`, `phases/reconcile.py`, `loop.py` (re-entry wiring), `tests/test_monitor.py`, `tests/test_reconcile.py`, `tests/test_reentry.py`.
- [ ] Tests first, each on a hand-built workspace with explicit `Params`. Monitor: the formula for confidence 0, 0.5 and 1, with and without event mismatch. Mood: one pass moves valence to exactly `m·v + (1 − m)·tanh(RS·signed)` and arousal to its formula; higher `m` moves less; values stay in range for extreme PE. Memory: one episode per pass with the specified salience, clamped at 1. Goals: priority unchanged for `patience − 1` consecutive over-threshold same-sign passes and changed by exactly `revision_step` on the next; a sub-threshold pass or a sign flip restarts the streak; after a revision `ema` and `streak` are 0; passes serving another goal do not touch this goal; clamping at both bounds; `revisions` and `goal_revised` recorded. Dispositions: only `bias[a]` moves, by exactly the formula, clamped at `±bound`. Timescales: after one large PE with default `Params`, the relative change of mood exceeds that of the goal priority and of the bias. Conflict: fires only when a downward revision and `bias[a] ≥ habit_threshold` coincide on pass 0; not on an upward revision, not below the threshold, not when disabled, never on pass 1. Re-entry: a tick whose pass 0 conflicts has exactly two passes and two world steps, even if pass 1 would conflict again; pass-1 Deliberate receives the conflict and a percept that includes the pass-0 episode; `agent.tick` advances by one; a conflict on the world's last step gives one pass; an interrupted-and-resumed run with two-pass ticks on both sides of the cut is byte-identical to an uninterrupted one, and `replay` passes on it.
- [ ] Implement. Commit `step 4: monitor and reconcile`.

### Task 9 — Default parameters and the behavioural gate (build step 4)
Files: `config.py` (defaults only), `experiments/stub_smoke.yaml`, `experiments/trait_grid.yaml` (FR → PR → FR re-baseline → extinction, `ticks: null`; a baseline agent plus a low and a high profile for each of the three traits, 5 seeds), `tests/test_behaviour.py`.
- [ ] Write the gate first, on stubs over 20 fixed seeds with the `trait_grid.yaml` schedule. Baseline agent: active presses exceed inactive presses in the first FR phase; PR response rate in the last quarter is below the first quarter; breakpoint is above 0 and censored in at most 2 of 20 runs; extinction response rate in the first third exceeds the last third; extinction persistence is censored in at most 2 of 20 runs; at least one goal revision in at least 10 runs, and no priority at a clamp at the end of any run; at least one two-pass tick in at least 10 runs, and two-pass ticks are under 25% of ticks in every phase; 300 steps of steady FR1 produce at most one revision.
- [ ] Trait gate, same 20 seeds, low versus high profile differing by more than two standard errors of the seed-paired difference: reward_sensitivity high → higher breakpoint; surprise_learning_rate high → lower extinction persistence; mood_inertia high → higher extinction persistence. Breakpoint and persistence are each censored in at most 4 of 20 runs for every low and high profile, so the contrasts are not artefacts of phase length.
- [ ] Tune defaults and schedule lengths until the gate passes. If no setting can meet a criterion, stop and report which one and why; do not weaken the criterion and do not change a formula.
- [ ] Report the tuned values and the 20-seed mean and standard deviation of each metric per profile. Commit `step 4: defaults and behavioural gate`.

### Task 10 — Deliberate on pydantic-ai (build step 5)
Files: `src/surprise_loop/agents.py`, `phases/deliberate.py`, `config.py` (`Params.llm`: output_retries, temperature, timeout_s), `loop.py` (resolve the model once per run into `Runtime.model`), `tests/conftest.py` (disable real model requests for the whole suite), `tests/test_agents_deliberate.py`.
- [ ] First read https://ai.pydantic.dev (agents, output, testing, models, migration) and the installed package source, and confirm each call below against them.
- [ ] `agents.py`: `DeliberateOutput(reasoning, action, expected_event, expected_reward, confidence)` with field descriptions and ranges; `deliberate_agent` built with no model, static instructions describing the task and output, and a prompt builder `deliberate_prompt(ws: Workspace) -> str` covering observation, mood, appraisal, goal priorities, dispositions, traits, recalled episodes (with salience and critique) and the conflict note when set.
- [ ] `deliberate_llm(ws, rt)` runs the agent with `rt.model` and maps the output to `Decision`.
- [ ] Tests, using only `TestModel` / `FunctionModel`: the prompt contains each state element above and none of `phase_kind`, `ratio_required`, `active_press`, `ratio_completed`, `active_lever`; a valid structured response maps to the right `Decision`; out-of-range confidence or an unknown action triggers a retry and, when it persists, raises; `backends.deliberate = "llm"` with `model: test` runs 20 ticks, writes valid dumps, and `replay` passes on them; an `llm` backend with no model fails before tick 1 with `ConfigError`.
- [ ] Commit `step 5: deliberate agent`.

Checked against the live docs and the source at tag v2.68.0 (re-confirm against the installed version):
- `Agent(model=None, *, output_type=..., instructions=..., deps_type=..., retries=...)`; the model may be omitted and passed per run. `instructions` is preferred over `system_prompt`.
- `result = await agent.run(prompt, model=..., model_settings=...)`; the structured value is `result.output`; `result.usage` is a property.
- One `retries=` argument (int or `{'output': N, 'tools': N}`); `output_retries` no longer exists. Exhausted output retries raise `UnexpectedModelBehavior`.
- Tool output is the default structured-output mode; the output tool is named `final_result`.
- Model strings need a provider prefix (`openrouter:`, `anthropic:`, `openai:`, `google:`). The string `test` resolves to `TestModel`. OpenRouter reads `OPENROUTER_API_KEY`.
- `from pydantic_ai.models.test import TestModel`; `from pydantic_ai.models.function import FunctionModel, AgentInfo`; `agent.override(model=...)`; `models.ALLOW_MODEL_REQUESTS = False`. A `FunctionModel` returns structured output with `ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {...})])` (composed from source; there is no docs example).
- For hundreds of sequential calls: resolve the `Model` object once per run (`models.infer_model(name)`), run the whole loop in one event loop, and keep the HTTP client open for the run. A model *string* passed per call opens and closes a client each time.

### Task 11 — Appraise on pydantic-ai (build step 5)
Files: `agents.py`, `phases/appraise.py`, `tests/test_agents_appraise.py`. Same pattern as Task 10: `AppraiseOutput(valence, arousal, note)`, `appraise_prompt(ws)`, `appraise_llm`. Tests: prompt content and the same hidden-field exclusions; mapping to `Appraisal`; out-of-range values retried then raised; a 20-tick run with `appraise: llm`, `model: test`. Commit `step 5: appraise agent`.

### Task 12 — Evaluate on pydantic-ai (build step 5)
Files: `agents.py`, `phases/evaluate.py`, `experiments/llm_smoke.yaml` (`ticks: 40`, all three backends `llm`, `model: null`), `tests/test_agents_evaluate.py`. Same pattern: `EvaluateOutput(alignment, critique)`, `evaluate_prompt(ws)` covering goals, the action, the prediction and the actual outcome, `evaluate_llm`. Tests: as Task 11, plus `ws.agent` equal before and after the phase, plus a 20-tick run with all three backends `llm` and `model: test` on which `replay` passes. Commit `step 5: evaluate agent`.
Then, by me, if the key was provided: the 40-tick real-model smoke run. I read the dumps, check that predictions and feedback are in range and sensible, and send back to an implementer whatever the real model exposes (prompt wording, retries, thinking setting).

### Task 13 — Experiment runner and CSVs (build step 6)
Files: `cli.py`, `metrics.py` (CSV writers), `experiments/llm_traits.yaml` (4 profiles × 2 seeds, short schedule, `llm` backends, `model: null`), `tests/test_experiment.py`.
Produces: `surprise-loop experiment CONFIG [--runs-dir DIR] [--jobs N]`; one run directory `DIR/<run_id>/` per (agent, seed); in `DIR/<experiment name>/`: `results.csv` (one row per run: agent, the three traits, seed, every `RunMetrics` field), `phases.csv` (agent, seed, then `PhaseMetrics` fields), `pe_trace.csv` (agent, seed, then `PEPoint` fields), `stability.csv` (metric, icc1, n_agents, n_seeds).
- [ ] Tests first: a 2-agent × 2-seed stub experiment writes the four files with the documented columns and 4 result rows; a rerun produces identical CSVs; `--jobs 4` equals `--jobs 1`; a run directory that is already complete is not rerun, and a partial one is resumed.
- [ ] Implement; runs execute concurrently in one event loop bounded by `--jobs`. Commit `step 6: experiment runner`.
Then, by me, if the key was provided: `experiments/llm_traits.yaml` on the real model with `--jobs 8`.

### Task 14 — Plotting script (build step 6)
Files: `scripts/plot_results.py`, `tests/test_plot.py` (`pytest.importorskip("matplotlib")`, Agg backend).
`python scripts/plot_results.py EXPERIMENT_DIR` writes into `EXPERIMENT_DIR/plots/`: `metrics_by_agent.png` (each behavioural metric per trait profile, seeds as points, mean marked), `pe_trace.png` (unsigned PE over steps per agent, phase boundaries marked), `cumulative_record.png` (cumulative active presses over steps per agent). Test: on a small stub experiment it writes three non-empty PNG files. Commit `step 6: plots`.

### Task 15 — Salience-based retrieval (build step 7, minimal)
Files: `state.py` (`Memory.salient_k`), `config.py` (`memory.retrieval: Literal["last_k", "salient_k"] = "last_k"`), `phases/perceive.py`, `experiments/trait_grid_salient.yaml`, tests.
`salient_k(k)`: the most recent `k // 2` episodes plus the `k − k // 2` highest-salience episodes outside that window (ties to the more recent), returned in chronological order. Tests: the selection and ordering on a hand-built memory; identical to `last_k` when there are at most k episodes; with the default config a run's dumps are byte-identical to before this task. Nothing else: no decay, graph, or capacity limit. Commit `step 7: salience-based retrieval`.

### Task 16 — Whole-branch review and wrap-up (by me)
Fresh Opus reviewer over the full diff against the brief and this plan, given the code and not my conclusions; one fix wave. Run `experiments/trait_grid.yaml` and the plot script for real; put the resulting numbers, the stability table and the reproduction commands in the README, with the real-model results beside them if they exist. CI green on the draft PR.

## Execution

- Subagent-driven: every implementer and reviewer runs on Opus; I orchestrate, check each result against the contracts myself before moving on, keep a progress ledger so nothing is redone after context compaction, and continue through Task 16 without checking in. Plan defects found on the way are decided by me against the brief and listed in the final report.
- Branch `ccr-563e5fbe-fxxlri`; pushed after every task; draft PR after the first push; I watch CI and fix failures.
- **OpenRouter key.** Immediately after you approve this plan I open a secure form for an OpenRouter key for host `openrouter.ai`. It is stored on this session's environment ("full access", your personal one) and injected by the network proxy; I never see it and it is never written to the repo. If you dismiss the form I continue with test models only and say so at the end.
- **Real-model runs I will make with it**, all with `SURPRISE_LOOP_MODEL=openrouter:z-ai/glm-5.3-flash`, none in the test suite or CI:
  1. after Task 12, the 40-tick smoke run (about 120 calls);
  2. after Task 13, one small experiment: 4 trait profiles × 2 seeds × about 150 ticks (about 3,600 calls).
  At the listed price ($0.04 per million input tokens, $0.08 per million output) this is well under $1 in total. I stop real-model runs if spend looks like exceeding that by an order of magnitude, and make no other real-model calls.

## Verification

- `uv run ruff check . && uv run ruff format --check . && uv run pytest` at every commit; CI green on Python 3.12 and 3.13.
- `uv run surprise-loop run experiments/stub_smoke.yaml` → 100 tick dumps and printed metrics; interrupt and `--resume` gives identical files; `surprise-loop replay` passes.
- `uv run surprise-loop experiment experiments/trait_grid.yaml` → four CSVs; `uv run --group plot python scripts/plot_results.py runs/trait_grid` → three PNGs that I open and check.
- With the key: `SURPRISE_LOOP_MODEL=openrouter:z-ai/glm-5.3-flash uv run surprise-loop experiment experiments/llm_smoke.yaml`, then `experiments/llm_traits.yaml --jobs 8`; I read the dumps and report what the real model did, including anything that looks wrong.
