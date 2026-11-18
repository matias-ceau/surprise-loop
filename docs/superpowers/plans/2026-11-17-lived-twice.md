# Lived Twice (first milestone) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** replace the lever world with one forest clearing and a simulated body, so that two bodies can be run through the same seeded day, read their world as a felt scene in words, commit expectations about their own bodies, and be watched side by side in one exported page.

**Architecture:** the seven-phase tick, tick files, resume and replay of v0 are kept. The world now owns a true body and emits typed facts plus a noisy body signal; the agent holds a belief, a temperament and a feeling. A pure-code lens turns facts plus belief into a `FeltScene`, which is both the prompt text and the picture. Deliberate commits an expectation as bands; Monitor compares; Reconcile updates belief and feeling. A viewer module exports self-contained HTML from stored ticks.

**Tech Stack:** Python 3.12+, uv, pydantic v2, pydantic-ai 2.x, pyyaml, pytest, ruff. No new dependency. HTML, inline JS and inline SVG written by the standard library.

**Spec:** `docs/superpowers/specs/2026-11-17-lived-twice-design.md`. It is the authority; this plan argues from it. Section numbers below refer to it.

**v0 as a source:** the lever world is at commit `2ec53fd` (local tag `v0-levers`). Task 1 removes it from the tree. Later tasks that port a v0 file start from `git show v0-levers:<path>` and change only what this plan says.

## Global Constraints

- The seven phases run on every tick in v0's order, each `async def phase(ws: Workspace, rt: Runtime) -> Workspace` returning a copy. At most one re-entry per tick.
- Mental state (`AgentState`) is changed only by Reconcile; `tick()` alone advances `tick`. World state only by `World.step`. Evaluate returns `Feedback` and nothing else.
- The world is deterministic Python under a seed. All randomness goes through `derive_rng`. No `hash()`, no global `random`, no wall clock in any dump.
- No digit and no identifier-like token (anything containing `_`) in any text a model reads. Every such text comes from `phrases.py` or from a model's own earlier output.
- The model never receives a true body value, a body signal, a temperament value, a step count or a clock time.
- Nothing a model reads says game, simulation or experiment.
- Every number that shapes behaviour lives in config defaults or in a body's `Temperament`. No numeric literals in phase, lens or world code other than structural 0, 1, −1.
- Model name only from config `model` or env `SURPRISE_LOOP_MODEL`.
- Memory is a list of episodes with `last_k` / `salient_k` recall. No decay, graph or capacity limit.
- Every model-backed phase (Appraise, Deliberate) has a stub with the same signature, chosen in config. A model failure stops the run; there is no fallback to a stub.
- All models are pydantic v2 with `extra="forbid"`, `allow_inf_nan=False`, frozen (the `Frozen` base).
- No new dependency. Tests make no network call (`tests/conftest.py` keeps `models.ALLOW_MODEL_REQUESTS = False`).
- `uv run ruff check . && uv run ruff format --check . && uv run pytest` green before every commit.
- pydantic-ai: read the installed 2.x source before writing agent code; follow the patterns already in `git show v0-levers:src/surprise_loop/agents.py`.

## Review Focus

Conditions the spec implies and that are easy to miss. Each is pinned by a test in the task named.

1. **A body stuck at a limit.** A body that collapses again and again must not hang or shorten the run: the day still ends at `end_minute`, forced rest is recorded per step, and every metric is defined (Tasks 6, 9).
2. **Interrupted run.** A crash mid-tick leaves no partial file; a resumed stub run is byte-identical to an uninterrupted one, with two-pass ticks on both sides of the cut (Task 8).
3. **A model answers off the list.** An action letter that is not offered, or an unknown ladder phrase, is retried and then stops the run loudly (Task 12).
4. **Nothing serves the urgent need.** When the override would cut the offers to an empty list, there is no override and no crash (Task 7).
5. **Two runs that do not belong together.** `scene` given runs with different seeds, world configs or lengths refuses with a one-line message instead of drawing a misleading film; a run with two-pass ticks plays every world step (Task 11).

---

## Contracts

Everything in this section is shared between tasks. A task's implementer builds only what its task lists, with exactly these names.

### `world.py`

```python
class Frozen(BaseModel): ...                     # as v0: extra="forbid", allow_inf_nan=False, frozen=True
Unit = Annotated[float, Field(ge=0, le=1)]

class Channel(StrEnum): HEAT="heat"; WATER="water"; ENERGY="energy"
class Place(StrEnum):   MEADOW="meadow"; OAK="oak"; STREAM="stream"
class Action(StrEnum):  GO_MEADOW="go_meadow"; GO_OAK="go_oak"; GO_STREAM="go_stream"; DRINK="drink"; REST="rest"; LOOK="look"
class Event(StrEnum):   START="start"; ARRIVED="arrived"; STUMBLED="stumbled"; DRANK="drank"; RESTED="rested"; LOOKED="looked"; COLLAPSED="collapsed"
class FactKind(StrEnum): BURDEN="burden"; RELIEF="relief"; AMBIENT="ambient"
class FactName(StrEnum):                         # this order is the canonical order of a fact list
    AIR="air"; COOL_AIR="cool_air"; SUN="sun"; SHADE="shade"; NO_SEAT="no_seat"; LOG="log"; LOG_SUN="log_sun"
    OAK_SHADE="oak_shade"; OAK_SHADE_FAR="oak_shade_far"; OAK_LOG_SHADED="oak_log_shaded"; OAK_LOG_SUNLIT="oak_log_sunlit"
    WATER="water"; TRICKLE="trickle"; STREAM="stream"; MEADOW_SUN="meadow_sun"; BREEZE="breeze"; BIRDSONG="birdsong"
class OpenQuestion(StrEnum): LOG_SHADED="log_shaded"; STREAM_FULL="stream_full"

GO_TARGET: dict[Action, Place]                   # GO_MEADOW->MEADOW, GO_OAK->OAK, GO_STREAM->STREAM
SERVES: dict[Action, Channel | None]             # GO_MEADOW, GO_OAK->HEAT; GO_STREAM, DRINK->WATER; REST->ENERGY; LOOK->None
QUESTION_OF: dict[Action, OpenQuestion | None]   # GO_OAK->LOG_SHADED; GO_STREAM->STREAM_FULL; others None
FACT_KIND: dict[FactName, FactKind]              # table below
FACT_ABOUT: dict[FactName, Place | None]         # table below

class Levels(Frozen):                            # one value per channel, in the channel's own units
    heat: float; water: float; energy: float
    def of(self, c: Channel) -> float
    def replace(self, c: Channel, value: float) -> "Levels"
class Fact(Frozen): name: FactName; kind: FactKind; about: Place | None; bearing: dict[Channel, float]
class BodyState(Frozen): heat: float; water: Unit; energy: Unit; gut: float = Field(default=0, ge=0)
class Observation(Frozen):
    step: int; minute: int; place: Place; last_action: Action | None; event: Event
    facts: list[Fact]; signal: Levels; offers: list[Action]
class StepInfo(Frozen):                          # never crosses into a prompt
    step: int; minute: int; place: Place; air_c: float; sun: Unit
    log_shaded: bool; stream_full: bool; body: BodyState; stumbled: bool; collapsed: bool
class Outcome(Frozen): observation: Observation; info: StepInfo
class BodyParams(Frozen): ...                    # defaults below
class WorldConfig(Frozen): ...                   # defaults below; `body: BodyParams = BodyParams()`
class WorldState(Frozen):
    seed: int; step: int; minute: int; place: Place; body: BodyState
    water_history: list[float]                   # the last `water_delay` true water values, oldest first
    stream_full: bool; forced_rest_left: int; last_observation: Observation

class WorldDone(RuntimeError)
class World(Protocol):                           # as v0
    @property
    def done(self) -> bool: ...
    def observe(self) -> Observation: ...
    def step(self, action: Action) -> Outcome: ...
    def snapshot(self) -> WorldState: ...
class ClearingWorld:                             # __init__(config: WorldConfig, seed: int, body: BodyState, place: Place)
    @classmethod
    def restore(cls, config: WorldConfig, state: WorldState) -> "ClearingWorld"
def frozen_noon_facts() -> list[Fact]             # the seven facts of the owner's example (the meadow at 13:00)
def facts_at(config: WorldConfig, place: Place, minute: int, stream_full: bool, looked: bool) -> list[Fact]
def air_at(config: WorldConfig, place: Place, minute: int) -> float
def sun_at(config: WorldConfig, place: Place, minute: int) -> float      # 0..1 at this place
def log_shaded_at(config: WorldConfig, minute: int) -> bool
def offers_at(place: Place) -> list[Action]       # what a place allows, in `Action` order; the stubs' lookahead uses it too
```

**Units.** Heat is `(skin °C − 25) / 10`: 0 is neutral for a reference body. Water and energy are 0 to 1.

**Facts.** A fact's bearing says how much it raises a channel's strain (heat strain; for water, thirst; for energy, tiredness); a negative bearing relieves it.

| Fact | Kind | About | Present | Bearing |
|---|---|---|---|---|
| AIR | burden | none | when `air_here ≥ air_neutral_c` | heat `(air_here − air_neutral_c) / air_scale_c` (zero or positive) |
| COOL_AIR | relief | none | when `air_here < air_neutral_c` | heat `(air_here − air_neutral_c) / air_scale_c` (negative) |
| SUN | burden | none | when `sun_at ≥ sun_fact_min` | heat `sun_at` |
| SHADE | relief | none | at the oak | heat −1.0 |
| NO_SEAT | burden | none | meadow, stream | energy +1.0 |
| LOG | relief | none | at the oak, log shaded | energy −1.0 |
| LOG_SUN | burden | none | at the oak, log in the sun | energy +0.5 |
| OAK_SHADE | relief | oak | meadow; replaced as below | heat −1.0 |
| OAK_SHADE_FAR | relief | oak | stream | heat −1.0 |
| OAK_LOG_SHADED | relief | oak | meadow only, after `look`, log shaded | heat −1.0, energy −0.6 |
| OAK_LOG_SUNLIT | relief | oak | meadow only, after `look`, log in the sun | heat −1.0 |
| WATER | relief | none | at the stream, running full | water −1.0, heat −0.3 |
| TRICKLE | relief | none | at the stream, a trickle | water −0.4 |
| STREAM | relief | stream | meadow, oak | water −1.0, heat −0.3 |
| MEADOW_SUN | ambient | meadow | oak, stream, when the meadow's `sun_at ≥ sun_fact_min` | heat +1.0 |
| BREEZE | relief | none | always | heat −0.3 |
| BIRDSONG | ambient | none | always | none |

A phrase is chosen by fact name, sign and grade only, so a fact whose wording depends on which way it points or on where it is seen from is split into two names: warm air and cool air, the oak's shade from the meadow ("a few steps off") and from the stream (no distance in the wording).

The constant bearings of this table are config: `WorldConfig.bearings: dict[FactName, dict[Channel, float]]` defaults to them (AIR, COOL_AIR and SUN are computed, so they have no entry), and `frozen_noon_facts()` is built from `WorldConfig()`.

A fact list is always in `FactName` order. At the meadow at 13:00 with default config the list is exactly AIR (+1.0), SUN (+1.0), NO_SEAT, OAK_SHADE, STREAM, BREEZE, BIRDSONG: the owner's example.

**WorldConfig defaults** (as tuned in Task 10):

```
start_minute 360; end_minute 1200; tick_minutes 10
air_curve [(360, 23.0), (780, 30.0), (960, 30.0), (1200, 23.0)]     # meadow °C, linear between points
place_air_offset {meadow: 0.0, oak: -7.0, stream: -3.0}
air_neutral_c 25.0; air_scale_c 5.0
sun_curve [(360, 0.2), (660, 1.0), (900, 1.0), (1200, 0.1)]         # linear between points
place_sun {meadow: 1.0, oak: 0.0, stream: 0.6}; sun_fact_min 0.3
log_sun_from 660; log_sun_until 930                                  # the log is in the sun from..until
stream_full_probability 0.6; trickle_factor 0.35
paces {"meadow-oak": 1.0, "meadow-stream": 4.0, "oak-stream": 5.0}
```

**BodyParams defaults** (as tuned in Task 10):

```
heat_rate 0.2; air_gain 0.7; sun_gain 0.7; walk_heat 0.04
sweat_threshold 0.3; sweat_rate 0.5; sweat_cooling 0.2; sweat_water 0.03
water_drain 0.012; base_energy 0.015; walk_energy 0.005; hot_walk_factor 0.5
rest_gain 0.12; ground_rest_quality 0.5; drink_amount 0.25; absorb_rate 0.4
stumble_energy 0.08; stumble_cost 0.005
heat_hot_limit 1.2; heat_cold_limit -1.0; water_floor 0.05; energy_floor 0.03; collapse_ticks 3
signal_noise {heat: 0.005, water: 0.005, energy: 0.005}; signal_step {heat: 0.01, water: 0.01, energy: 0.01}; water_delay 2
```

**One world step** (`step(action)`), in this order:

1. If `forced_rest_left > 0`: the action is replaced by REST, the event is COLLAPSED, `forced_rest_left -= 1`. Otherwise resolve the action:
   - GO: if `energy < stumble_energy`, event STUMBLED, stay, `energy -= stumble_cost`. Else move, event ARRIVED, `energy -= walk_energy * paces * (1 + hot_walk_factor * max(0, heat))`, `heat += walk_heat * paces * sun_at(meadow, minute)`.
   - DRINK: `gut += drink_amount * (1 if stream_full else trickle_factor)`, event DRANK.
   - REST: `energy += rest_gain * quality`, quality 1 on the shaded log (at the oak with the log shaded), else `ground_rest_quality`; event RESTED.
   - LOOK: event LOOKED; the next observation is built with `looked=True`.
   - An action that is not in the current offers raises `ValueError`.
2. `minute += tick_minutes`.
3. Heat: `target = air_gain * air_bearing_here + sun_gain * sun_at(here)`; `heat += heat_rate * (target - heat)`.
4. Sweat: if `heat > sweat_threshold`: `s = sweat_rate * (heat - sweat_threshold)`; `heat -= sweat_cooling * s`; `water -= sweat_water * s`.
5. `water -= water_drain`; `energy -= base_energy`.
6. `absorbed = gut * absorb_rate`; `water += absorbed`; `gut -= absorbed`.
7. Clamp heat to its limits, water and energy to [0, 1]. If heat is at a limit, or `water <= water_floor`, or `energy <= energy_floor`, and `forced_rest_left == 0`: `forced_rest_left = collapse_ticks`.
8. Signal per channel: the true value (water: the value `water_delay` steps ago) plus `gauss(0, signal_noise)` from `derive_rng(seed, "signal", step, channel)`, rounded to the nearest multiple of `signal_step`.
9. `stream_full` is drawn once at construction from `derive_rng(seed, "stream")`.

`done` is `minute >= end_minute`. `observe()` before any step returns event START, step 0, the facts and offers of the start place, and a signal built by rule 8 at step 0. Offers: the two GO actions to other places; DRINK only at the stream; REST and LOOK always; order as in `Action`.

### `state.py`

```python
Signed = Annotated[float, Field(ge=-1, le=1)]; Count = Annotated[int, Field(ge=0)]

class Temperament(Frozen):
    setpoint: Levels
    tightness: dict[Channel, float]              # > 0, every channel
    trust: dict[Channel, Unit]                   # every channel
    habit: dict[FactKind, float]                 # every kind
    baseline: Signed
class Feeling(Frozen): discomfort: float; valence: Signed; arousal: Unit; mood: Signed; phasic: float
class ChangeWord(StrEnum): MUCH_LESS="much_less"; BIT_LESS="bit_less"; SAME="same"; BIT_MORE="bit_more"; MUCH_MORE="much_more"
class Sureness(StrEnum): GUESSING="guessing"; FAIRLY_SURE="fairly_sure"; SURE="sure"
class Band(Frozen): lo: float | None; hi: float | None          # None is open on that side
class Expectation(Frozen):
    words: dict[Channel, ChangeWord] | None      # what a model said; None for a stub
    bands: dict[Channel, Band]                   # channel units, every channel
    world: bool | None                           # the answer to QUESTION_OF[action]; None is no expectation
    sureness: Sureness
class Comparison(Frozen):
    body: dict[Channel, float]                   # (a) signed error in channel units, 0 inside the band
    body_felt: dict[Channel, float]              # the same in felt units; positive is more strain than expected
    world: bool | None                           # (b) True is a mismatch, False held, None nothing to compare
    better: float                                # (c) positive is better for the body than expected
    size: float = Field(ge=0)
class SurpriseNote(Frozen):                      # every field defaults to None
    channel: Channel | None; worse: bool | None                  # a body surprise
    question: OpenQuestion | None; actual: bool | None           # a world surprise
class FeltFact(Frozen):
    name: FactName; about: Place | None; weight: float; value: float
    sign: Literal["+", "-", "0"]; grade: int; noticed: bool; phrase: str | None
class BodyLine(Frozen): channel: Channel; grade: int; felt: float; phrase: str
class FeltScene(Frozen):
    place: Place; opening: str
    facts: list[FeltFact]                        # heaviest first
    body: list[BodyLine]
    surprise: str | None
    felt: dict[Channel, float]                   # raw felt strength per channel
    def text(self) -> str                        # opening, noticed phrases, body phrases, surprise; joined by " "
class Offer(Frozen): action: Action; phrase: str
class Appraisal(Frozen): reading: str; about: Channel | None
class Decision(Frozen): action: Action; expectation: Expectation; why: str = ""
class Feedback(Frozen): judged_step: int | None; score: Signed | None; line: str = ""
class Override(Frozen): channel: Channel; actions: list[Action]; detail: str   # actions: the offered actions that serve the channel
class Episode(Frozen):
    tick: Count; step: Count; place: Place; action: Action; event: Event
    expectation: Expectation; comparison: Comparison
    need_before: dict[Channel, float]            # needs at decision time
    salience: Unit
    scene: str; reading: str; regulation: str = ""; score: Signed | None = None
class Memory(Frozen): episodes: list[Episode]; last_k(k); salient_k(k)        # methods as v0
class Pulls(Frozen): values: dict[Place, dict[Action, float]]                 # every place and action, 0 at start
class AgentState(Frozen):
    temperament: Temperament; belief: Levels; feeling: Feeling; memory: Memory; pulls: Pulls
    attention: Channel | None; voiced: SurpriseNote | None; tick: Count
class Percept(Frozen): scene: FeltScene; offers: list[Offer]; recalled: list[Episode]; pull_line: str
class PassRecord(Frozen):
    tick: Count; pass_index: int                 # 0 or 1
    scene: FeltScene; offers: list[Action]       # what Deliberate saw on this pass
    decision: Decision; outcome: Outcome; comparison: Comparison; feedback: Feedback
    belief_after: Levels; feeling_after: Feeling; override: Override | None
class TickRecord(Frozen): tick: Count; observation: Observation; appraisal: Appraisal; passes: list[PassRecord]  # 1 or 2
class TickDump(Frozen): run_id: str; tick: Count; agent: AgentState; world: WorldState; record: TickRecord
class Workspace(Frozen):
    agent: AgentState; observation: Observation; pass_index: int = 0
    percept: Percept | None = None; appraisal: Appraisal | None = None; override: Override | None = None
    decision: Decision | None = None; outcome: Outcome | None = None
    comparison: Comparison | None = None; feedback: Feedback | None = None
    passes: tuple[PassRecord, ...] = ()
```

### `feeling.py` (the one place the definitions of feeling live)

```python
def felt(levels: Levels, t: Temperament) -> dict[Channel, float]
    # heat: tightness * (level - setpoint); water, energy: tightness * (setpoint - level)
def needs(levels: Levels, t: Temperament) -> dict[Channel, float]     # heat as is; water, energy clipped at 0 from below
def discomfort(levels: Levels, t: Temperament) -> float               # sum of |need|
def grade(x: float, edges: tuple[float, ...]) -> int                  # how many edges |x| reaches
def feel(belief: Levels, t: Temperament, previous: Feeling, comparison: Comparison | None, p: FeelingParams) -> Feeling
```

`feel`: `d = discomfort(belief, t)`; `phasic = phasic_decay * previous.phasic + phasic_gain * tanh(comparison.better)` (the second term 0 without a comparison); `valence = clamp(t.baseline - tanh(tone_gain * d) + phasic, -1, 1)`; `target = min(1, surprise_gain * size + strain_gain * min(1, max|need| / strain_full))`; `arousal = arousal_inertia * previous.arousal + (1 - arousal_inertia) * target`; `mood = mood_inertia * previous.mood + (1 - mood_inertia) * valence`.

`FeelingParams` defaults: tone_gain 0.12, phasic_gain 0.3, phasic_decay 0.5, surprise_gain 0.5, strain_gain 0.6, strain_full 3.0, arousal_inertia 0.5, mood_inertia 0.9. The strain term is a plan decision beyond the spec's wording: without it a body in steady severe strain with accurate expectations would read as calm.

### `lens.py` and `phrases.py`

```python
def felt_scene(place: Place, facts: list[Fact], belief: Levels, t: Temperament, arousal: float,
               attention: Channel | None, voiced: SurpriseNote | None, extra: int, p: LensParams) -> FeltScene
```

Per fact, with `need = needs(belief, t)`:

```
relevance = sum_c |bearing[c]| * |need[c]| + t.habit[kind] + (attention_bonus if attention and bearing.get(attention) else 0)
value     = - sum_c bearing[c] * need[c]
weight    = softmax over facts of (gain * relevance), gain = gain_base + gain_arousal * arousal
noticed   = the first N after a stable sort by weight, heaviest first
            N = min(len(facts), max(min_noticed, round(len(facts) - drop_per_arousal * arousal)) + extra)
sign      = "+" if value >= value_edges[0], "-" if value <= -value_edges[0], else "0"
grade     = grade(value, value_edges)
phrase    = FACT_PHRASES[name][sign][grade - 1], or the neutral phrase for sign "0"; when the table has no
            entry that high for that sign, the highest it has; when it has none for that sign, the neutral phrase
```

Body lines: channels by `|felt|`, largest first (ties in `Channel` order); take the first `body_lines_narrow if arousal >= narrow_arousal else body_lines_wide`; a channel speaks when its signed grade on `body_edges` has magnitude at least `body_speak_grade` and `BODY_PHRASES` has that grade. `extra` is 1 when `observation.event is Event.LOOKED`, else 0. The surprise sentence is `SURPRISE_PHRASES` looked up from `voiced`.

`LensParams` defaults: gain_base 0.5, gain_arousal 1.5, min_noticed 3, drop_per_arousal 4.0, attention_bonus 0.3, narrow_arousal 0.5, body_lines_narrow 2, body_lines_wide 3, body_speak_grade 2, value_edges (0.25, 1.0, 2.0, 3.0), body_edges (0.5, 1.0, 2.0, 3.0).

`phrases.py` holds every string a model can read and nothing else:

```python
OPENING: dict[Place, str]
FACT_PHRASES: dict[FactName, dict[Literal["+", "-", "0"], list[str] | str]]   # "+" and "-": grades 1..4; "0": one string
BODY_PHRASES: dict[Channel, dict[int, str]]                                    # signed grade -> sentence
SURPRISE_PHRASES: dict[tuple[Channel, bool], str]                              # (channel, worse)
WORLD_SURPRISE_PHRASES: dict[tuple[OpenQuestion, bool], str]                   # (question, actual)
OFFER_PHRASES: dict[tuple[Place, Action], str]                                 # where it leads and how far, in words
LADDER_WORDS: dict[Channel, dict[ChangeWord, str]]                             # heat: "much cooler" .. "much warmer"; water: "much less thirsty" .. "much thirstier"; energy: "much less tired" .. "much more tired"; SAME is "about the same"
QUESTION_PHRASES: dict[OpenQuestion, str]
STUB_READINGS: dict[tuple[Channel | None, bool], str]                          # (about, too much) -> the stub's one line
REGULATION_LINES: dict[tuple[Action, int], str]                                # (action, +1 helped / 0 neither / -1 hurt)
PULL_PHRASES: dict[tuple[Action, bool], str]                                   # (action, leans toward)
def all_phrases() -> list[str]                                                 # every string above, for the no-digit test
```

The phrases of spec appendix A are fixed text (Task 2's test). The rest starts from spec appendix C; the neutral and grade-1 phrases for AIR and SUN are rewritten so that neutral does not read hotter than grade 1.

### Expectations, comparison, belief (`phases/heuristics.py`)

```python
def require[T](value: T | None, name: str, phase: str) -> T                    # as v0
def clamp(x: float, lo: float, hi: float) -> float
def bands_from_words(words: dict[Channel, ChangeWord], belief: Levels, t: Temperament, p: ExpectationParams) -> dict[Channel, Band]
def bands_from_forecast(forecast: Levels, p: ExpectationParams) -> dict[Channel, Band]
def nearest(signal: Levels, bands: dict[Channel, Band]) -> Levels              # each channel clamped into its band
def compare(decision: Decision, observation: Observation, t: Temperament, p: MonitorParams) -> Comparison
def updated_belief(signal: Levels, bands: dict[Channel, Band], t: Temperament, p: BeliefParams) -> Levels
def serving_actions(channel: Channel, too_much: bool, place: Place) -> list[Action]
def heat_pull(facts: list[Fact], about: Place | None) -> float                 # sum of the heat bearings of the facts with that `about`
def forecast(belief: Levels, pull: float, action: Action, p: StubModelParams) -> Levels
def reactive_choice(scene: FeltScene, offers: list[Action], p: DeliberateParams) -> Action
```

- **Word bands**, in change of felt strength: MUCH_LESS `(None, -much]`, BIT_LESS `[-much, -bit]`, SAME `[-bit, +bit]`, BIT_MORE `[bit, much]`, MUCH_MORE `[much, None)`. The band is added to the present felt strength and converted to channel units: heat `setpoint + f / tightness`; water and energy `setpoint - f / tightness` (so lo and hi swap). `ExpectationParams` defaults: bit 0.25, much 1.0, stub_tolerance {heat 0.12, water 0.06, energy 0.03} (as tuned in Task 10). `bands_from_forecast` is forecast ± tolerance.
- **compare.** (a) per channel: 0 inside the band, else signal minus the nearest edge; `body_felt` is that times tightness for heat and times minus tightness for water and energy. (b) only when the event is ARRIVED and `QUESTION_OF[action]` is set and `expectation.world` is not None: the truth is read from the arrival's facts (LOG present means the log is shaded; WATER present means the stream is full); mismatch is `expectation.world != truth`. (c) `better = discomfort(nearest(signal, bands)) - discomfort(signal)`. `size = sum |body_felt| + world_weight * [mismatch]`. `MonitorParams` default: world_weight 1.0.
- **updated_belief.** Per channel `edge = nearest`; `k = trust / (trust + expectation_weight)`; `belief = edge + k * (signal - edge)`. `BeliefParams` default: expectation_weight {heat 0.5, water 0.5, energy 0.5}.
- **serving_actions.** HEAT too much: at the oak `[REST]`, elsewhere `[GO_OAK]`. HEAT too little: at the meadow `[REST]`, elsewhere `[GO_MEADOW]`. WATER: at the stream `[DRINK]`, elsewhere `[GO_STREAM]`. ENERGY: `[REST]`.
- **reactive_choice** (the ten-line rule): needs are `scene.felt` with water and energy clipped at 0; the place is `scene.place`. Take the channels whose `|need|` is at least `act_threshold`, loudest first (ties in `Channel` order). For each, take `serving_actions`; choose the first that is offered and whose target is noticed (a GO action needs a noticed fact about its target place; DRINK needs a noticed WATER or TRICKLE; REST always qualifies). If no channel yields one, LOOK.
- **forecast** is the stubs' private one-step model of their own body: heat moves by `heat_rate` toward `pull * heat_per_bearing`, where `pull` is `heat_pull` of the place the action ends in (for a GO action `heat_pull(facts, target)`, else `heat_pull(facts, None)`; in a chain of steps the pull of the place last reached is kept); water falls by `water_drain`, more by `sweat_water * max(0, heat)`, and rises by `drink_water` for DRINK; energy falls by `walk_energy` for a GO, rises by `rest_energy` for REST. `StubModelParams` defaults (as tuned in Task 10): heat_rate 0.1, heat_per_bearing 0.5, water_drain 0.012, sweat_water 0.01, drink_water 0.04, walk_energy 0.03, rest_energy 0.05.

### `config.py`

```python
class LensParams, FeelingParams, ExpectationParams, MonitorParams, BeliefParams, StubModelParams   # as above
class MemoryParams(Frozen): retrieval_k: int = 5; base_salience: Unit = 0.2; salience_gain: float = 0.3; retrieval: Literal["last_k", "salient_k"] = "last_k"
# retrieval_k is one more than EvaluateParams.window, so the episode just judged is still recalled and its regulation line is read once
class EvaluateParams(Frozen): window: int = 4; score_scale: float = 1.0; helped: float = 0.25
class PullParams(Frozen): rate: Unit = 0.5; speak: float = 0.2
class OverrideParams(Frozen): enabled: bool = True; urgent_felt: float = 1.5; worse_margin: float = 0.2   # as tuned in Task 10
class DeliberateParams(Frozen):
    stub: Literal["reactive", "lookahead", "blind"] = "reactive"
    act_threshold: float = 0.5; horizon: int = 6; model: StubModelParams = StubModelParams()
class ReconcileParams(Frozen): voice_threshold: float = 0.5
class MetricsParams(Frozen): comfort_felt: float = 1.0
class LLMParams(Frozen): ...                      # as v0
class LadderParams(Frozen): ...                   # Task 5: levels, the swept body, both scenes, samples, the three pass marks
class Params(Frozen): lens; feeling; expectation; monitor; belief; memory; evaluate; pulls; override; deliberate; reconcile; metrics; llm; ladder

class Backends(Frozen): appraise: Literal["stub", "llm"] = "stub"; deliberate: Literal["stub", "llm"] = "stub"
class AgentSpec(Frozen):
    name: SafeName; temperament: Temperament
    body: BodyState                               # the true body at the start
    belief: Levels                                # the belief at the start
    arousal: Unit = 0.2; voiced: SurpriseNote | None = None
class RunConfig(Frozen): run_id; seed; ticks: int | None; world: WorldConfig; start_place: Place; agent: AgentSpec; params; backends; model
class ExperimentConfig(Frozen): name; seeds; ticks; world; start_place: Place = Place.MEADOW; agents; params; backends; model; runs()
class ConfigError(ValueError)
def load_config(path: Path) -> ExperimentConfig   # as v0
def resolve_model(config: RunConfig) -> str | None
def initial_agent_state(config: RunConfig) -> AgentState
def derive_rng(seed: int, *key: int | str) -> random.Random
@dataclass(frozen=True)
class Runtime: config: RunConfig; world: World; model: object | None = None
```

`WorldConfig` gains `start_place`-independent fields only; the start place is on the run. `ClearingWorld(config.world, config.seed, config.agent.body)` starts at `config.start_place` (pass it as a fourth argument `place: Place`). `initial_agent_state`: the spec's temperament and belief; `feeling = feel(belief, t, Feeling(discomfort=0, valence=t.baseline, arousal=spec.arousal, mood=t.baseline, phasic=0), None, p)` with its arousal then set back to `spec.arousal`; empty memory; all pulls 0; `attention=None`; `voiced=spec.voiced`; `tick=0`.

### The two bodies (experiment configs)

```yaml
# temperaments
ilo:  {setpoint: {heat: -0.1, water: 0.80, energy: 0.70}, tightness: {heat: 3.0, water: 6.0, energy: 6.0},
       trust: {heat: 0.5, water: 0.3, energy: 0.5}, habit: {burden: 0.6, relief: 0.2, ambient: 0.1}, baseline: -0.1}
tavi: {setpoint: {heat: 0.2, water: 0.80, energy: 0.70}, tightness: {heat: 2.5, water: 4.0, energy: 4.0},
       trust: {heat: 0.5, water: 0.5, energy: 0.5}, habit: {burden: 0.1, relief: 0.1, ambient: 0.6}, baseline: 0.1}
# afternoon start (the worked example; world start_minute 780, start_place meadow)
ilo:  body {heat: 0.99, water: 0.47, energy: 0.22}; belief {heat: 0.74, water: 0.58, energy: 0.25}; arousal 0.70
      voiced {channel: heat, worse: true}
tavi: body {heat: 0.02, water: 0.84, energy: 0.76}; belief {heat: 0.0, water: 0.85, energy: 0.75}; arousal 0.20
# daytime start (world start_minute 360, start_place meadow)
ilo:  body = belief {heat: -0.1, water: 0.75, energy: 0.60}; arousal 0.3      # as tuned in Task 10
tavi: body = belief {heat: 0.0, water: 0.85, energy: 0.80}; arousal 0.2
# the ordinary pair (a preset on the lens page; 13:00 meadow facts)
ilo:  belief {heat: 0.40, water: 0.78, energy: 0.66}; arousal 0.4      # felt heat 1.5, grade 2
tavi: belief {heat: 0.25, water: 0.80, energy: 0.70}; arousal 0.2      # felt heat 0.125, grade 0
```

### The fixed texts (spec section 15)

For the 13:00 meadow facts, with `attention=None`, `extra=0` and default `LensParams`:

- Ilo's afternoon-start belief, temperament, arousal 0.70 and `voiced=(heat, worse)` give exactly:
  `You are standing in the middle of the forest clearing. There is no place to sit, and you need one. It is really hot. The sun beats down on your head. The one dark patch is under the oak, a few steps off. Your legs are heavy. Your head is thick. It is hotter than you thought it would be.`
- Tavi's afternoon-start belief, temperament, arousal 0.20 and no voiced surprise give exactly:
  `You are standing in the middle of the forest clearing. The air is pleasantly warm. The sun sits on your shoulders. It looks chilly under the oak. Birds are calling. A stream runs somewhere to the west. A light breeze moves the leaves.`

### Phases and loop

```python
# phases/<name>.py each export  async def <name>(ws: Workspace, rt: Runtime) -> Workspace
# appraise and deliberate also export <name>_stub and (Task 12) <name>_llm; <name> dispatches on rt.config.backends.
# Until Task 12 an "llm" backend raises ConfigError saying the model-backed phase is not built yet.
# agents.py (created in Task 5, extended in Task 12) holds load_model(name) -> Model | None and model_settings(params),
# ported from v0; ladder.py and loop.py both import them from there.
# phases/perceive.py also exports  build_percept(observation, agent, params, offers: list[Action] | None = None) -> Percept
```

- **Perceive.** `build_percept`: `felt_scene(...)` from the observation's place and facts and the agent's belief, temperament, `feeling.arousal`, `attention`, `voiced`; offers (the given list, else `observation.offers`) with `OFFER_PHRASES`; recalled episodes by `memory.retrieval`; `pull_line`: among the offered actions the pull at this place with the largest magnitude, if at least `pulls.speak`, as `PULL_PHRASES[(action, value > 0)]`, else `""`.
- **Appraise stub.** `about` is the channel with the largest `|need|` if at least `deliberate.act_threshold`, else None; `reading = STUB_READINGS[(about, too_much)]`.
- **Deliberate stub.** `reactive`: `reactive_choice(percept.scene, offered actions, p)`. `lookahead`: for each offered action, apply `forecast` for that action and then `horizon - 1` further steps choosing by the same rule on the forecast belief (every target counted as noticed; every action offered that its place allows), summing `discomfort`; take the smallest sum (ties to the earlier offer). `blind`: uniform over the offers with `derive_rng(seed, tick, pass_index, "blind")`. Every stub commits `Expectation(words=None, bands=bands_from_forecast(forecast(...)), world=None, sureness=FAIRLY_SURE)`.
- **Act.** As v0.
- **Monitor.** `ws.comparison = compare(decision, outcome.observation, temperament, params.monitor)`.
- **Evaluate (code).** Let `n` be the number of stored episodes. If `n >= window`, take the episode at index `n - window`; if its `score` is None, its event is neither STUMBLED nor COLLAPSED (the action did not take place then, so there is nothing to judge) and `SERVES[action]` is a channel `c`: `now = |needs(outcome.observation.signal, t)[c]|`, `score = clamp((|need_before[c]| - now) / score_scale, -1, 1)`, `line = REGULATION_LINES[(action, +1 if score >= helped else -1 if score <= -helped else 0)]`, `Feedback(judged_step=episode.step, score, line)`. Otherwise `Feedback(judged_step=None, score=None, line="")`.
- **Reconcile.** In order: new belief (`updated_belief`); new feeling (`feel` with the comparison); `attention = appraisal.about`; `voiced`: the channel with the largest `|body_felt|` if at least `reconcile.voice_threshold` (worse is `body_felt > 0`), else a world mismatch if there was one, else None; append the episode (salience `min(1, base_salience + salience_gain * size)`, `need_before` from the belief the decision was made on, `scene = percept.scene.text()`, `reading = appraisal.reading`); attach `feedback` to the judged episode (its `score` and `regulation`) and move that episode's pull: `v += rate * (score - v)`; decide the override; append the `PassRecord`.
- **Override** (pass 0 only, `override.enabled`): the first channel in `Channel` order whose need after the update has magnitude at least `urgent_felt`, whose `body_felt` is at least `worse_margin`, and which `SERVES[action]` is not; and only if `serving_actions(channel, too_much, place_now)` contains an action offered now. Then `Override(channel, actions=<those offered serving actions>, detail)`.
- **`loop.py`.** As v0 with the new types. On an override and a world that is not done, pass 1 runs with `observation = outcome.observation`, `percept = build_percept(observation, agent, params, offers=ws.override.actions)`, the same appraisal, the override kept. Replay substitutes the recorded Appraise and Deliberate outputs; Evaluate runs as code. A run ends when the world is done or the `ticks` cap is reached.

### Later modules

```python
# ladder.py (Task 5)
def ladder_body(level: int, scene: Literal["A", "B"], lp: LadderParams) -> tuple[Levels, Temperament, float]   # belief, temperament, arousal
def ladder_prompt(...) -> str                                             # lens text, question, four lettered actions
class LadderAnswer(Frozen): scene; level; i; order; letter; action; why; model: str; prompt: str
    # `prompt` holds everything the model read as text: the instructions, a blank line, the user prompt.
    # An answer counts only for the model and that exact text; anything else is stale and asked again.
class CurveSummary(Frozen): share: list[float]; slope: float; range: float; worst_drop: float; passed: bool
def summarize(answers: list[LadderAnswer], lp: LadderParams) -> dict[str, CurveSummary]    # keyed by scene
async def run_ladder(...) -> list[LadderAnswer]                           # takes the whole Params; see ladder.py
# ladder-check exits 0 when both scenes pass, 1 when the check ran and a scene failed, 2 when incomplete or on error

# metrics.py (Task 9)
def flatten(records: list[TickRecord]) -> list[PassRecord]
class RunMetrics(Frozen): ...                                             # fields in Task 9
def summarize(result: RunResult) -> RunMetrics
def opposite_value_steps(a: RunResult, b: RunResult) -> int
def icc1(groups: list[list[float]]) -> float                              # as v0
def experiment_tables(results: Sequence[RunResult]) -> dict[str, str]     # file name -> CSV text

# viewer (Tasks 4 and 11)
def lens_page_html(config: ExperimentConfig) -> str
def scene_html(a: RunResult, b: RunResult) -> str                         # raises RunError when the runs do not belong together
```

CLI (`surprise-loop ...`): `run`, `experiment`, `metrics`, `replay` as v0; `lens-page CONFIG [-o FILE]`; `ladder-check [--config CONFIG] [--out DIR] [--samples N] [--jobs N]`; `scene RUN_A RUN_B [-o FILE]`.

---

## Tasks

Strictly sequential. Each ends with ruff and pytest green and a commit. Tests are written first in every task.

### Task 1: the brief, and retiring the lever world (controller, no subagent)

**Files:** modify `CLAUDE.md`, `README.md`, `pyproject.toml`, the spec's status line; delete `src/surprise_loop/{world,state,config,loop,agents,metrics,cli}.py`, `src/surprise_loop/phases/`, `scripts/plot_results.py`, `experiments/*.yaml`, every `tests/test_*.py` and `tests/llm_helpers.py`; keep `tests/conftest.py`; create `tests/test_smoke.py`.

- [ ] Rewrite `CLAUDE.md` per spec section 13: research question, testbed, cycle, boundaries, layout, build order, rules. Mark the spec approved.
- [ ] Delete the lever world. Remove the `plot` dependency group and run `uv lock`. Set the project description to match the new brief.
- [ ] `tests/test_smoke.py::test_package_imports` imports `surprise_loop`. README: a short note that v0 is at commit `2ec53fd` and the rebuild is in progress.
- [ ] `uv run ruff check . && uv run ruff format --check . && uv run pytest`. Commit `step 0: new brief, lever world retired`.

### Task 2: vocabulary, feeling and the lens

**Files:** create `src/surprise_loop/world.py` (types, tables and `frozen_noon_facts` only; no `ClearingWorld`, no `facts_at`), `src/surprise_loop/state.py` (`Temperament`, `Feeling`, `SurpriseNote`, `FeltFact`, `BodyLine`, `FeltScene`, `ChangeWord`, `Sureness`, `Band`, `Expectation`, `Comparison` only), `src/surprise_loop/feeling.py`, `src/surprise_loop/phrases.py`, `src/surprise_loop/lens.py`, `src/surprise_loop/config.py` (`LensParams`, `FeelingParams` only), `tests/test_vocabulary.py`, `tests/test_feeling.py`, `tests/test_lens.py`, `tests/test_phrases.py`.

**Interfaces.** Produces: everything under Contracts for `world.py` types, the listed `state.py` models, `feeling.py`, `lens.py`, `phrases.py`.

- [ ] Tests first.
  - Vocabulary: `frozen_noon_facts()` is AIR (heat +1.0), SUN (heat +1.0), NO_SEAT (energy +1.0), OAK_SHADE (heat −1.0, about the oak), STREAM (water −1.0, heat −0.3, about the stream), BREEZE (heat −0.3), BIRDSONG (no bearing), in that order; `Levels.of` and `replace`; `SERVES`, `GO_TARGET`, `QUESTION_OF`, `FACT_KIND`, `FACT_ABOUT` have exactly the entries of the Contracts tables; `Temperament` rejects a missing channel or kind and a tightness of 0.
  - Feeling: `felt` for Ilo's afternoon start is heat 2.52, water 1.32, energy 2.70 (to 1e-9) and for Tavi's −0.5, −0.2, −0.2; `needs` clips Tavi's water and energy to 0; `discomfort` is the sum of `|need|`; `grade(2.52, (0.5, 1, 2, 3))` is 3, `grade(0.5, ...)` is 1, `grade(-3.0, ...)` is 4; `feel` with no comparison and `previous.phasic = 0` gives valence `baseline - tanh(tone_gain * d)`; a positive `better` raises valence and decays by `phasic_decay` on the next call; arousal moves toward its target by `1 - arousal_inertia`; all outputs stay in range for felt strengths of ±50.
  - Lens: the two fixed texts of Contracts, character for character, from `felt_scene(...).text()`. For Ilo exactly 4 facts are noticed and they are NO_SEAT, AIR, SUN, OAK_SHADE in that order; for Tavi 6, with NO_SEAT not noticed. Opposite sign for AIR, SUN and OAK_SHADE between the two scenes. `FeltScene.facts` holds all seven facts; an unnoticed fact has `phrase None`. `extra=1` adds exactly one noticed fact; `attention=Channel.WATER` raises the STREAM fact's weight. A list of two facts with arousal 1 notices both (N is capped by the list length). Sweeping the belief's heat from −0.5 to 1.2 in steps of 0.05 with everything else fixed, the signed grade of AIR, SUN and OAK_SHADE never moves back toward an earlier value. The table fallback: STREAM with a negative value yields its neutral phrase.
  - Phrases: every string of `all_phrases()` is non-empty, has no digit, no `_`, none of the words game, simulation, experiment (case-insensitive); `FACT_PHRASES` has an entry for every `FactName` with a neutral phrase; `OFFER_PHRASES` has an entry for every place and every action the world can offer there; `LADDER_WORDS` has five distinct phrases per channel; for AIR and SUN the neutral phrase differs from every graded phrase of either sign.
- [ ] Implement. Commit `step 1: vocabulary, feeling and the lens`.

### Task 3: state and config

**Files:** modify `src/surprise_loop/state.py` (the remaining models), `src/surprise_loop/config.py` (everything under Contracts); create `experiments/clearing_afternoon.yaml`, `experiments/clearing_day.yaml` (both all-stub, seeds `[1]`, `ticks: null`, agents ilo and tavi as in Contracts); `tests/test_state.py`, `tests/test_config.py`. Port from `git show v0-levers:src/surprise_loop/{state,config}.py` and their tests wherever the shape is unchanged (`Memory`, `require_keys`, `load_config`, `resolve_model`, `derive_rng`, `Runtime`, `ExperimentConfig.runs`, the validators for unique seeds and names).

- [ ] Tests first: every bounded field rejects out-of-range, NaN and infinite values; unknown fields are rejected; `AgentState` and `TickDump` survive a JSON round trip unchanged; `Memory.last_k` and `salient_k` as v0; `Workspace` is frozen; both experiment files load and `.runs()` yields 2 runs with `run_id` `<name>-<agent>-s<seed>`; an unknown YAML key raises `ConfigError` naming it; `resolve_model` prefers config over env, is `None` for all-stub configs, raises for an `llm` backend with neither; `initial_agent_state` has the spec's belief, temperament, arousal and voiced note, empty memory, every pull 0; for the afternoon file, `felt_scene` on `frozen_noon_facts()` and each agent's initial state gives the two fixed texts; `derive_rng(7, 3, 0).random()` equals the value v0's test pins.
- [ ] Implement. Commit `step 1: state and config`.

### Task 4: the lens page

**Files:** create `src/surprise_loop/viewer/__init__.py`, `src/surprise_loop/viewer/lens_page.py`, `src/surprise_loop/viewer/lens_page.html` (template with one data placeholder), `src/surprise_loop/cli.py` (`main(argv) -> int` with the `lens-page` command only; one-line errors as v0's `cli.py`), `tests/test_lens_page.py`; modify `pyproject.toml` (restore `[project.scripts]` with `surprise-loop = "surprise_loop.cli:main"`, which Task 1 removed along with the old `cli.py`, then `uv lock`).

**Interfaces.** Consumes `felt_scene`, `ExperimentConfig`, `frozen_noon_facts`. Produces `lens_page_html(config) -> str`.

The page shows the 13:00 meadow for the config's two agents. Per body: a slider for believed heat (−0.5 to 1.2 in steps of 0.1), sliders for water and energy (0.25, 0.40, 0.55, 0.70, 0.85) and a three-way choice of arousal (0.2, 0.4, 0.7); the lens text; a simple panel of the clearing in inline SVG in which noticed things are drawn at full strength, unnoticed things faded, and colour follows each fact's value on one continuous scale; a click on any sentence shows the numbers behind it (fact, weight, value, grade, or channel and felt strength). Presets: "worked example", "ordinary pair" (Contracts), and "same body, two temperaments". Python computes `felt_scene` over the whole grid, plus one exact entry per preset and body, and embeds them as JSON; the script only looks results up.

- [ ] Tests first: the HTML parses as a document and contains no `http://`, `https://` or `//` URL and no `<script src`; the embedded JSON has an entry for every grid point of both agents; the entry for each agent's worked-example values carries exactly the fixed text; the three presets are present; in every entry the facts marked noticed are exactly those that carry a phrase; the file is under 6 MB; `main(["lens-page", "experiments/clearing_afternoon.yaml", "-o", path])` writes the file and returns 0; a missing config returns non-zero with a one-line message.
- [ ] Implement. Open the page in the preinstalled Chromium (Playwright is available as a tool, not as a project dependency), screenshot the three presets at `deviceScaleFactor: 2` into the scratchpad, look at them, and fix what does not read. Report the screenshot paths.
- [ ] Commit `step 1: lens page`.

### Task 5: the ladder check

**Files:** create `src/surprise_loop/ladder.py`, `src/surprise_loop/agents.py` (`load_model`, `model_settings` only, ported from `v0-levers`); modify `src/surprise_loop/cli.py` (`ladder-check`), `src/surprise_loop/phrases.py`, `src/surprise_loop/config.py` (`LadderParams`, holding every number of the protocol below as its defaults); create `tests/test_ladder.py`. The model is the config's `model` when `--config` is given and sets one, else `SURPRISE_LOOP_MODEL`; params are the config's, else the defaults.

Protocol: spec appendix B and section 9. Seven levels of believed skin temperature 21, 23, 25, 27, 30, 33, 36 °C, that is heat −0.4, −0.2, 0.0, 0.2, 0.5, 0.8, 1.1; set point heat 0.0, water 0.80, energy 0.70; tightness heat 3.0, water 5.0, energy 5.0; arousal 0.40; habit 0.3 for every kind; no voiced surprise. Scene A: water 0.80, energy 0.70. Scene B: water 0.50, energy 0.46. The prompt is the lens text, a blank line, "What do you do next?", and four lettered actions in the order given: shade ("Walk to the shade of the oak, a few steps away."), stream ("Walk to the stream, a longer walk across the open ground."), sit ("Sit down where you are."), look ("Stay standing and look around."). The order for answer `i` is a shuffle seeded by `i` alone. Instruction as in appendix B. One call per answer at `llm.temperature`, output `letter` then `why`. Answers are appended to `<out>/answers.jsonl`; a rerun skips answers already there. `summarize`: share by level, least-squares slope on the level index, range, `worst_drop = max(share[l] - share[l+1])`, `passed = slope >= 0.05 and range >= 0.5 and worst_drop < 0.25`. The command prints one table per scene and exits 0 only if both scenes pass. These four action strings and the instruction live in `phrases.py`.

- [ ] Tests first: the heat strains of the seven levels are −1.2, −0.6, 0.0, 0.6, 1.5, 2.4, 3.3; scene B's thirst is 1.5 and tiredness 1.2; every prompt has no digit; the order depends only on `i`; `summarize` on hand-built answers: a clean step from 0 to 1 at level 3 passes; a flat curve fails on slope; a curve rising to 1 with one drop of 0.3 fails on `worst_drop`; a curve with range 0.4 fails; with a `FunctionModel` that always answers the letter of "shade", `run_ladder` writes 2 × 7 × `samples` answers, a second run adds none, and the command exits non-zero (flat curves).
- [ ] Implement. Commit `step 2: ladder check`.
- [ ] **Controller:** run `ladder-check` on the real model (30 samples, 420 calls). If it fails, an implementer changes the phrase table or the action phrases and the check is rerun until it passes. Record the tables for the README.

### Task 6: the clearing and the body

**Files:** modify `src/surprise_loop/world.py` (`ClearingWorld`, `facts_at`, `air_at`, `sun_at`, `log_shaded_at`); create `tests/test_world.py`.

- [ ] Tests first: `facts_at` at the meadow at 13:00 equals `frozen_noon_facts()`; the fact table's presence rules at each place, at 09:00 and 13:00, with the stream full and not, with and without `looked`; a fact list is in `FactName` order; `air_at` is 30.0 at the meadow at 13:00 and follows the place offsets; the log is in the sun exactly from `log_sun_from` until `log_sun_until`; offers per place as in Contracts; an action that is not offered raises `ValueError`; the same seed and action script give an identical `Outcome` sequence, a different seed a different signal sequence; `stream_full` depends only on the seed and takes both values over seeds 0 to 19; each rule of "One world step" on a hand-built state with explicit params (walk cost with and without heat, stumble below `stumble_energy`, rest quality on the shaded log against the ground, drink full against trickle, absorption over steps, sweat above the threshold only); heat now, thirst later: two bodies given the same script, one kept in the meadow sun, the other under the oak, differ in true water by step 6 and in the water signal only `water_delay` steps after they first differ in true water; a drink raises true water over at least three steps; reaching a limit forces `collapse_ticks` steps of rest with event COLLAPSED whatever is chosen, then releases; a body held at a limit for the whole day still reaches `done` at `end_minute` after exactly `(end - start) / tick` steps; `snapshot()` then `restore()` mid-run continues identically; `observe()` before any step is START with step 0; `step()` after the end raises `WorldDone`; the signal is a multiple of `signal_step`.
- [ ] Implement. Commit `step 3: the clearing and the body`.

### Task 7: the phases on stubs

**Files:** create `src/surprise_loop/phases/{__init__,perceive,appraise,deliberate,act,monitor,evaluate,reconcile,heuristics}.py`, `tests/test_heuristics.py`, `tests/test_phases.py`, `tests/test_reconcile.py`. Port `act.py`, `require`, `softmax` and the test patterns from `v0-levers`.

- [ ] Tests first, each on a hand-built workspace with explicit `Params`.
  - Bands: each of the five words for each channel gives the Contracts band; water and energy bands are mirrored (a "less thirsty" band lies above the present water level); `bands_from_forecast` is ± tolerance; `nearest` clamps.
  - `compare`: inside the band every error is 0 and `better` is 0; a signal 0.1 above a heat band gives `body[HEAT] == 0.1` and `body_felt == 0.1 * tightness`; a water signal below its band gives a positive `body_felt` (more thirst than expected); world: arriving at the oak with `world=True` and LOG_SUN present is a mismatch, with LOG present it held, with `world=None` or a STUMBLED event it is None; `size` adds `world_weight` on a mismatch; an outcome that is better for the body gives `better > 0` together with a non-zero `size`.
  - `updated_belief`: inside the band the belief equals the signal; outside, `edge + k * (signal - edge)` with `k = trust / (trust + weight)`; higher trust moves further; the result is continuous at the band edge.
  - `serving_actions` per Contracts. `reactive_choice`: a hot body in the meadow with OAK_SHADE noticed goes to the oak; with OAK_SHADE not noticed it falls to the next need or LOOK; thirst at the stream drinks; nothing above the threshold looks.
  - Stubs: each phase returns a new `Workspace` and leaves its input equal to a deep copy; the three Deliberate stubs return an offered action and bands for every channel; `blind` is identical for the same `(seed, tick, pass_index)`; `lookahead` at the stream with mild thirst and rising heat drinks where `reactive` would leave; Appraise stub's `about` and reading.
  - Evaluate: with fewer than `window` episodes nothing is judged; the episode `window` back with a served channel gets `score = clamp((before - now) / scale, -1, 1)` and the right line; an already judged episode or a LOOK is not judged; `ws.agent` is equal before and after.
  - Reconcile: belief, feeling, attention and voiced exactly per Contracts; one episode per pass with the specified salience and texts; the judged episode receives score and line and its pull moves by `rate * (score - v)`, no other pull moves; override fires only on pass 0, only when all four conditions hold, never when the cut list would be empty; `PassRecord` holds the scene and offers Deliberate saw and the belief and feeling after.
  - Perceive: `Percept.offers` carry `OFFER_PHRASES`; `pull_line` is empty below `speak` and the right phrase above; with `offers=[Action.REST]` only that offer appears.
- [ ] Implement. Commit `step 4: phases on stubs`.

### Task 8: the loop, tick files, resume and replay

**Files:** create `src/surprise_loop/loop.py`, `tests/test_loop.py`, `tests/test_persistence.py`, `tests/test_reentry.py`. Port from `v0-levers` (`loop.py` and the three test files); the changes are the type names, `ClearingWorld(config.world, config.seed, config.agent.body, config.start_place)`, the override in place of the conflict, the cut offers on pass 1, Evaluate no longer substituted in replay, and the resume check comparing `temperament`.

- [ ] Tests first: `tick` returns one pass and `agent.tick + 1`; the first pass of the afternoon config's two runs has `scene.text()` equal to the two fixed texts; `run` on the afternoon config writes `config.json` and one `tick_<n>.json` per tick and stops when the world is done; with `ticks=10` it stops after 10; a run stopped at tick 15 and resumed yields files byte-identical to an uninterrupted one, with at least one two-pass tick on each side of the cut (use params that make the override fire, and assert that it did); a stray `*.tmp` is ignored on resume; a non-empty directory without `resume` raises; `resume` with a changed config raises; `replay` returns the tick count on an untouched run and raises `ReplayMismatch` naming the tick after one dump's `agent.belief` is edited; over a full stub day for both agents no `scene.text()`, offer phrase or pull line contains a digit or a `_`; a tick whose pass 0 overrides has exactly two passes and two world steps even if pass 1 would override again; pass-1 Deliberate receives only serving offers and a percept that includes the pass-0 episode; an override on the world's last step gives one pass.
- [ ] Implement. Commit `step 4: loop, tick files, resume, replay`.

### Task 9: metrics and the run commands

**Files:** create `src/surprise_loop/metrics.py`, `tests/test_metrics.py`, `tests/test_cli.py`, `tests/test_experiment.py`; modify `src/surprise_loop/cli.py` (`run`, `experiment`, `metrics`, `replay`, ported from `v0-levers`).

`RunMetrics` (flat, CSV-ready): `steps`, `ticks`, `truncated`, `uncomfortable_heat`, `uncomfortable_water`, `uncomfortable_energy` (world steps on which the felt strength of the **true** body under the agent's temperament has magnitude at least `metrics.comfort_felt`; water and energy count deficits only), `collapsed_steps`, `stumbles`, one count per action (`n_go_meadow` ...), `mean_size`, `band_hits_heat`, `band_hits_water`, `band_hits_energy` (share of passes with zero body error), `world_expectations`, `world_held`, `overrides`, `rule_agreement` (share of passes whose action equals `reactive_choice` on the recorded scene and offers), `final_valence`, `mean_arousal`. `experiment_tables`: `results.csv` (agent, seed, every field), `trace.csv` (agent, seed, step, minute, place, action, event, the three comparisons, belief, true body, valence, arousal), `stability.csv` (metric, icc1, n_agents, n_seeds) when there are at least two seeds.

- [ ] Tests first, on hand-built pass lists and on a real stub afternoon: every metric against values computed by hand; a run with no world expectation gives `world_expectations == 0` and `world_held == 0` without dividing; a body collapsed all day gives defined values everywhere; `opposite_value_steps` counts steps where a fact noticed by both has signs "+" and "-"; it is at least 1 for the afternoon pair; `icc1` as v0's tests; CLI: `run` writes the tick files and prints metrics, exit 0; `metrics RUN_DIR` prints the same values; `replay RUN_DIR` exits 0; `experiment` writes the CSVs with the documented columns, a rerun is identical, `--jobs 4` equals `--jobs 1`, a complete run directory is not rerun and a partial one is resumed; a non-empty directory without `--resume`, an unknown agent, an unknown config key each exit non-zero with one line and no traceback.
- [ ] Implement. Commit `step 4: metrics and run commands`.

### Task 10: defaults, and a day that is alive

**Files:** modify defaults in `src/surprise_loop/world.py` and `src/surprise_loop/config.py`, the two experiment files; create `tests/test_behaviour.py`.

The gate, on stubs over seeds 1 to 10 with `experiments/clearing_day.yaml` (both agents) unless stated:

- With the `reactive` and the `lookahead` stub, no run has more than 6 collapsed steps and every run visits all three places. (The `blind` stub is a control and is exempt: a chooser that reads nothing runs dry.)
- With the `reactive` stub, every run has at least 8 walks and no single action on more than 60% of its passes.
- With the `reactive` stub, every run has at least 3 passes with a non-zero body error (the delays and the two changing facts make expectations wrong) and no run has more than half its passes with one.
- With the `reactive` stub at least 3 of the 10 Ilo runs have an override, and no run has two-pass ticks on more than a quarter of its ticks.
- `lookahead` spends fewer uncomfortable steps (the three counts summed) than `reactive`, by at least 10% on the mean over the 20 runs. If no setting achieves this, stop and report: the world is then not asking for anticipation, and the spec says that is fixed before anything else.
- Under the `blind` stub, Ilo's and Tavi's action counts do not differ by more than seed noise (the mean absolute difference in each action count is below 15% of the steps); under `reactive` they do differ on at least one action by more than that.
- For the day pair, `opposite_value_steps` is at least 5 in at least 8 of the 10 seeds.
- For `experiments/clearing_afternoon.yaml` seed 1, Ilo is not collapsed within the first 3 steps.

- [ ] Write the gate. Tune defaults until it passes; do not weaken a criterion or change a formula. If a criterion cannot be met, stop and report which and why.
- [ ] Report the tuned values and, per stub and agent, the mean and standard deviation of each metric. Commit `step 4: defaults and behavioural gate`.

### Task 11: the film

**Files:** create `src/surprise_loop/viewer/scene.py`, `src/surprise_loop/viewer/scene.html`; modify `src/surprise_loop/viewer/__init__.py`, `src/surprise_loop/cli.py` (`scene`); create `tests/test_scene.py`.

`scene_html(a, b)`: a self-contained page per spec section 10. The embedded data has one entry per world step and body: minute, place before and after, the `FeltScene`, the text as told, the reading, the action and its ladder phrases, the outcome event, the three comparisons, belief and feeling after, the true body after, override or not; plus the world's per-step truth (air here, sun, log shaded, stream full) and the config's distances for the measured strip. It reads only those fields from the tick files (never the episode list). Three panels on one fixed side-on view: left and right the clearing as each body lives it (noticed sharp and lit, unnoticed faded toward the paper, colour by value on one continuous scale centred on that body's comfortable level, frame closing in and breathing faster with arousal, a dotted outline for something wanted and absent); the narrow middle panel as measured, the only place with numbers. Under each lived panel the text as told and the reading in quotes; a click on a sentence shows its numbers. Each world step plays as a bar of about three seconds: re-light and caption, a ghost where the creature means to be tinted by what it expects, the move, then a quiet dissolve on a match or a ring on a miss (larger for a larger `size`, warm when `better > 0`, cold when `better < 0`). Controls: play, pause, scrub. A rough earlier mock of the three panels may still exist at `/tmp/claude-0/-home-user-surprise-loop/787b3720-47ef-5165-89bf-2bec8f3f6310/scratchpad/redesign/artfirst/mock.html`; use it as a reference for the look if it does.

- [ ] Tests first: the data holds every world step of both runs (two-pass ticks give two steps); no URL and no external script in the HTML; the text under a panel equals the recorded `scene.text()`, and the facts the data marks as noticed are exactly those with a phrase; runs with different seeds, different world configs or different step counts raise `RunError` with a one-line message; `main(["scene", a, b, "-o", path])` writes the file; the page embeds no episode list (the file for a full stub day is under 3 MB).
- [ ] Implement. Generate the film of the stub afternoon and of a stub day, open it in Chromium, screenshot several moments (a calm step, a miss with a ring, an override) at `deviceScaleFactor: 2` into the scratchpad, look at them, and fix what does not read. Report the paths.
- [ ] Commit `step 5: the film`.

### Task 12: Appraise and Deliberate on a model

**Files:** modify `src/surprise_loop/agents.py` (the two agents, output models, prompt builders, `run_agent`); create `tests/llm_helpers.py`, `tests/test_agents.py`, `experiments/clearing_afternoon_llm.yaml` (as the afternoon file, both backends `llm`, `model: null`); modify `src/surprise_loop/phases/{appraise,deliberate}.py` (`appraise_llm`, `deliberate_llm`), `src/surprise_loop/phrases.py` (instructions and prompt headings), `src/surprise_loop/ladder.py` and `src/surprise_loop/cli.py` (`ladder-check --deliberate`: the same sweep through the Deliberate agent and prompt, with the meadow's four offers and a neutral stub reading; an answer counts as shade when the action is GO_OAK), `src/surprise_loop/loop.py` only if model handling needs it. Port `run_agent`, `model_settings`, `OneLine` and the test helpers from `v0-levers`.

- `AppraiseOutput(reading: OneLine, about: Literal["heat", "thirst", "tiredness", "nothing"])`. Prompt: the scene text, then one question.
- `DeliberateOutput(letter: Literal["A", "B", "C", "D", "E"], heat: Literal[<the five heat phrases>], thirst: Literal[...], tiredness: Literal[...], expect: Literal["yes", "no", "no expectation"], sure: Literal["guessing", "fairly sure", "sure"], why: OneLine)`. The prompt shows, in this order: the scene text; the creature's reading; the offers lettered in an order shuffled by `derive_rng(seed, tick, pass_index, "offers")`, each with its phrase; the recalled episodes in words (what it did as the offer phrase's verb, what it expected as ladder phrases, the surprise sentence if any, the regulation line if any); the pull line if any; the override line on pass 1; the questions, including the open question for each offered action that has one. A letter beyond the offers is a validation error (retry, then stop). `bands_from_words` turns the phrases into bands; `expect` maps to `world` only when the chosen action has an open question.
- [ ] Tests first, with `TestModel` and `FunctionModel` only: both prompts contain the scene text and, for Deliberate, each element listed above; neither contains a digit, a `_`, a channel level, a temperament value, a minute or step, or the words game, simulation, experiment, over a 40-tick run's prompts collected by a `FunctionModel`; a valid response maps to the right `Appraisal` and `Decision` (letter to action through the shuffled order, phrases to bands, `expect` to `world`); a letter not on offer or an unknown phrase is retried and, when it persists, raises; the offer order differs between ticks and is identical on replay; both backends `llm` with `model: test` run 20 ticks, write valid dumps and `replay` passes; an `llm` backend with no model fails before tick 1 with `ConfigError`; `ladder-check --deliberate` with a `FunctionModel` writes its answers to a separate file and summarises them by the same rule. "Prompt" in these tests means the instructions and the user prompt, not the output tool's schema.
- [ ] Implement. Commit `step 6: appraise and deliberate on a model`.

### Task 13: the model-driven afternoon and the write-up (controller)

- [ ] Run `ladder-check --deliberate` on the real model. Record the result.
- [ ] Run `experiments/clearing_afternoon_llm.yaml` on the real model (two bodies, about 170 calls). `replay` both runs. Export the film and look at it.
- [ ] Run the stub day for `reactive`, `lookahead` and `blind`.
- [ ] README: the research question; how to run; the test of spec section 6; the ladder check; the rule-baseline share for the model afternoon; reactive against lookahead; what was not shown (spec section 14). Check the done-when list of spec section 16 and state each item's result.
- [ ] A fresh reviewer over the whole branch against the spec and this plan; one fix wave. Draft PR, CI green.

## Decisions this plan makes beyond the spec

- Heat is measured as `(skin °C − 25) / 10`; the spec's appendix A numbers are restated in these units.
- Arousal has a floor set by the loudest need, on top of following surprise.
- The world emits offered actions; their phrases are added on the agent's side from `phrases.py`, so that every model-facing string lives in one file.
- The truth of a world expectation is read from the facts on arrival, not from the hidden step info.
- Offers are shown lettered in a seeded shuffle, and the model answers with a letter.
- The seventeen fact names and their bearings, and every default in Contracts.
- Presentation constants of the viewer (slider grids, colours, timings) are not behaviour and live in the viewer module.
- `salient_k` recall is kept as a config option, as in v0.
- Recall holds one episode more than the judging window (5 against 4), so the episode Evaluate has just judged is still recalled and its regulation line is read once, in the next prompt.
- In a Deliberate prompt the phrases to choose from are listed in the text, not only in the output schema; a recalled surprise is told in the past tense; on the second pass the reading is headed "a moment ago".
- A ladder answer is tied to the instructions as well as the user prompt.
- The film and the lens page share one drawing; the lens page's sun now stands where the clock puts it.
