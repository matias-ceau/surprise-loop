# Lived Twice: one clearing, two bodies

Design for the redirect of surprise-loop. Status: **approved by the owner on 2026-11-17**.
The build plan is `docs/superpowers/plans/2026-11-17-lived-twice.md`.

Terms used throughout:

- **Tick**: one pass through the seven phases, as in v0.
- **Stub**: a simple hand-written rule that stands in for the language model in a phase, so the
  whole thing runs with zero API calls.
- **Lens**: plain code that turns the world's facts plus one body's state into the scene that body
  lives. It calls no model.
- **Felt scene**: the lens's output for one body at one tick. It is stored once and used twice: as
  the picture and as the text the model reads.

## 1. Why this exists

v0 answered the question it was given (stable individual differences in lever pressing) and the
answer with a real model was "not yet". The owner's reading of it was "either dumb or obvious", and
an audit agreed with the owner's deeper doubt: v0 is reward learning written in predictive-brain
vocabulary. It has no body, it predicts only the reward of its own lever press, and feeling never
changes what is perceived.

What the owner asked for instead, in their words:

- "agents with a real inner life", with internal states that are "biological inner feelings/basic
  perceptions (think valence, arousal, lisa feldman barret or damasio)";
- "The real interest is how to input that 'objective' ... numerical value in the llms narrative":
  the same fact, "you are standing in the forest, its 30°C", becoming "you are in the middle of a
  forest, no place to sit, and it's really hot" for one body and something else for another;
- "experiences can of course also shape the weights of the perceptions";
- stay "on track with the name, and make the whole 'tick' always";
- few actions, "coherent with the world";
- "faked embodiment and not only LLM text", "a visually appealing scene that one can observe, a
  bit more artistic".

One part of v0 is faithful to the theory and is kept: the agent commits a prediction before it
acts, and plain code, not the model, measures how wrong it was.

## 2. What gets built, in one paragraph

One small forest clearing, shown three times on one screen. In the middle, the clearing as
measured. Left and right, the same clearing as each of two bodies lives it. The two bodies go
through the same seeded day, each in its own copy of the world; they never meet. The world owns
each body's true numbers (heat, water, energy). Each agent receives only noisy signals from its
body and holds its own belief about it. Every tick the lens turns the facts and that belief into a
felt scene. The model reads the scene in words, says how it expects its body to feel after its
next action, and acts. Code compares what it expected with what arrived. The size of the miss is
the surprise, and it is drawn on screen.

## 3. Goals, non-goals, anti-goals

**Goals for the first milestone ("one noon you can play with, one afternoon you can watch").**

1. A lens that reproduces the owner's forest example as fixed text for two bodies, with no digit
   and no variable name in anything the model reads.
2. A page where the owner can move a body's numbers and watch the two texts and two simple
   pictures change, and click any word to see the numbers that produced it.
3. A world with a body in it, a full seven-phase tick on stubs, tick files that replay byte for
   byte, exactly as v0 does.
4. One command that turns two run folders into one self-contained page: two lived panels and the
   measured strip, played from the stored ticks.
5. One afternoon (noon to dusk) for the two bodies with a real model deciding, so that the first
   film contains expectations nobody wrote, meeting a world where they can be wrong.
6. An honest write-up: how much of the model's behaviour a ten-line rule predicts, and whether
   looking ahead pays in this world.

**Non-goals for this milestone** (each is a later step, switched on one at a time):

- a second creature in the same world, speech, anything social;
- slow learning of temperament (set points, how much a deviation is minded, attention habits);
- mood feeding back into perception;
- night, death, food, weather noise;
- more than three body channels, three places or six verbs;
- any change to memory beyond three text fields per episode;
- a narrator model between the lens and the deciding mind;
- the running-loop comparison of ways of telling (numbers, plain words, lens) over whole days.

**Anti-goals** (it fails if any of these is true, even if it runs):

- The model is shown a body number, a trait number or a variable name.
- The picture is driven by the mood of the model's prose instead of the stored body numbers.
- Code stamps an emotion word on a figure.
- A result is reported that comes from the update code alone. v0 did this: its stable measures
  were made by the deterministic Reconcile. Every claim about the model is reported as model run
  minus stub run on the same seeds.
- The write-up says the agents feel, or says "allostasis" for behaviour that only corrects after
  the fact. The owner's own phrase, faked embodiment, is the accurate one.
- Memory grows beyond a list of episodes with last-K recall. The brief records that over-built
  memory killed the previous attempt.

## 4. Constraints

- The seven phases run on every tick, in v0's order, as plain async functions over one frozen
  workspace. The bounded re-entry (at most once per tick) stays.
- The world is deterministic seeded Python. No model narrates it.
- Every model phase has a stub with the same signature, chosen in config. A whole experiment can
  run with zero API calls.
- Every number that shapes behaviour lives in config or in a body's temperament. The model name
  comes from config or an environment variable.
- Full state is written after every tick to `runs/<run_id>/tick_<n>.json`; runs resume and replay
  from those files.
- Python 3.12+, uv, pydantic v2, pydantic-ai 2.x, pytest, ruff. **No new dependency**: the viewer
  is one HTML file with inline script and inline SVG, written by Python's standard library.
- The brief requires asking before adding a component. This design adds three (section 13).

## 5. Approaches considered

Three directions were laid out to the owner on 2026-11-17 after a research and design pass.

| Direction | What you watch | Question | Why not first |
|---|---|---|---|
| **1. Side by side** (chosen) | Two bodies live the same day in separate copies of the clearing | Do the model's choices follow the body, and does the way of telling matter? | |
| 2. Shared clearing | Two creatures, one shaded seat, one spring | At what strain does a model stop being polite and take the seat? | Heaviest build; no shared film for about three weeks; needs all of direction 1 first |
| 3. Twins | Identical bodies, a week of days, different luck | Does experience alone make them perceive differently? | Its headline effects did not survive probing (tiny at ordinary values, or all-or-none) |

**The owner picked direction 1**, approved a throwaway test before any project code, and left one
question to me: if the test showed that the way of telling makes no difference to the model,
should the picture still be built? My call: yes. The piece to watch is what the owner asked for;
the test decides what is claimed and whether the model keeps the deciding seat. As it turned out
(section 6), the way of telling matters a great deal.

Within direction 1, four ways of getting a number into the narrative were on the table:

1. raw numbers in the prompt (v0's way);
2. one plain sentence per need ("You feel very hot.");
3. the lens: code selects which facts this body notices, gives each a value for this body, and
   looks the wording up in a hand-written phrase table;
4. a second model call that narrates the scene, so the deciding mind never sees the lens.

Way 4 is dropped: it puts two model calls between a number and an action, so a disappointing
result cannot be traced, and it doubles the cost. Ways 1 to 3 were tested against each other.

## 6. What the test showed

One frozen noon, 30 °C, direct sun, the oak's shade 15 m away, a stream 60 m away. The body's
believed heat was swept over seven levels, from chilled to burning. The model chose one of four
actions. 30 answers per cell, 1,890 calls in all on the cheap OpenRouter model, no failed call.
The protocol and pass marks were written down before the run and checked by three reviewing
agents; the numbers below were recomputed from the raw file by a separate agent, which matched
them.

Share of 30 answers that walk to the oak's shade, by heat level (coolest on the left):

**Scene A, heat is the only need**

| Way of telling | chilled | cool | neutral | warm | hot | very hot | burning | follows the body? |
|---|---|---|---|---|---|---|---|---|
| No body told | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 1.00 | 1.00 | (base rate) |
| Numbers | 0.93 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | no |
| Plain adjectives | 0.07 | 0.13 | 0.23 | 1.00 | 1.00 | 0.97 | 1.00 | yes |
| Lens text only | 0.00 | 0.47 | 0.93 | 0.50 | 1.00 | 1.00 | 0.97 | yes |
| Facts + lens text | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | yes |

**Scene B, heat competes with a fixed moderate thirst and tiredness**

| Way of telling | chilled | cool | neutral | warm | hot | very hot | burning | follows the body? |
|---|---|---|---|---|---|---|---|---|
| Numbers | 0.67 | 0.17 | 0.37 | 0.57 | 0.90 | 1.00 | 1.00 | no |
| Plain adjectives | 0.17 | 0.07 | 0.07 | 0.73 | 0.87 | 0.77 | 0.97 | yes |
| Lens text only | 0.00 | 0.00 | 0.20 | 0.03 | 0.37 | 0.60 | 0.43 | no |
| Facts + lens text | 0.00 | 0.20 | 0.50 | 0.20 | 0.63 | 0.90 | 0.97 | yes |

"Follows the body" was fixed in advance as: the share rises by at least 0.05 per level and by at
least 0.5 from the coolest level to the hottest. In scene B nearly every answer that is not the
shade is the stream.

By the pass marks written beforehand, the verdict is **mixed**. What that means in plain terms:

1. **With no body, the answer is fixed.** At 30 °C in direct sun the model walks to the shade
   209 times out of 210. The world facts alone decide.
2. **Numbers do not carry the body.** Told "heat -1.20 (below 0 is too cold)", the model still
   walks to the shade 28 times out of 30, and in 27 of those 28 its stated reason is the heat
   ("I'm overheated (-1.20), so moving to nearby shade is the quickest way to cool down."). With
   competing needs the share does rise from cool (0.17) to burning (1.00), but at the coldest
   level two thirds still go to the shade, again citing the heat. This is the way v0 told its
   agents their mood and traits. It is a likely reason v0's traits barely reached behaviour,
   though the test does not prove that.
3. **Words carry it.** One plain sentence ("You feel chilly.") moves the model almost completely.
4. **The lens steers as well as a plain sentence, not better.** Nothing here supports a claim that
   the lens's wording is special to the model.
5. **The lens text alone loses what an action needs.** It never says how far anything is, and at
   high heat it makes the water louder too, so with competing needs half the answers still go to
   the stream at the hottest levels. With the plain facts added back, the same text follows the
   body in both scenes.
6. **The hand-written wording has a step that runs backwards.** In three of the four lens curves
   the neutral level sends more answers to the shade than the next warmer level: "The clearing is
   in full sun." reads hotter than "The sun is bright."
7. **A ten-line rule ("serve the loudest need") predicts 72 to 80% of the choices** in scene B
   for every way of telling, and 86% for plain adjectives in scene A. The model is not pure
   decoration here, and it is not far from a rule either.

Limits: one model, one scene, hand-written phrases, four actions, no running loop. The lens differs
from the other versions in three ways at once (felt wording, a value put on each thing one can
walk to, no metric facts), so "lens against numbers" is a comparison of packages. "Adjectives
against numbers" is the only clean words-against-numbers contrast, and it is the clearest result.
Scene B was added after a pilot showed scene A at ceiling, and was calibrated on the adjectives
version. About 5% of calls were billed 36 more input tokens than identical prompts in the same
cell, so the provider did not serve every call the same way; those answers do not differ
noticeably from the rest. The scripts and raw answers are session scratch files, not in the repo.

**What the test changes in the design.**

- Body state reaches the model in words only. That was the plan; it is now also the evidence.
- The lens earns its place as the one object that drives both the picture and the text, and as
  the thing that decides what a body notices. It is not claimed to steer the model better than a
  plain sentence would.
- The list of offered actions always says, in words, where each action leads and how far it is.
  The lens may leave a place out of the scene; the action list never does.
- The phrase table gets a test with the model in the loop (section 9, "ladder check"), not only
  a check on the strings.
- The rule baseline stays in every write-up.

## 7. The world and the body

### World

`ClearingWorld` replaces `SelfAdminWorld` behind the existing `World` protocol. Deterministic
Python with a seeded generator, one creature per world.

- **Places (three):** the meadow (open sun, nothing to sit on), the oak (shade, a fallen log), the
  stream (water).
- **Clock:** one tick is ten minutes. A run covers a daytime, 06:00 to 20:00, 84 ticks. An
  afternoon run starts at 13:00 and has 42 ticks.
- **Air temperature:** a daily curve given as a few points in config, about 18 °C at dawn and
  exactly 30.0 °C in the meadow at 13:00, so the owner's example is literally a tick of the run.
  Each place has an offset (the oak cooler, the meadow warmer in daylight).
- **Two facts that change and cannot be seen from everywhere.** They are what lets an expectation
  be wrong.
  - The oak's shade reaches the log only in the morning and the late afternoon. Around noon the
    log is in the sun, and resting at the oak means resting on the ground in the shade, which
    restores less. Whether the log is shaded can be seen from the meadow and the oak, not from the
    stream.
  - The stream runs full or as a trickle, drawn once per day from the seed. A trickle gives less
    water per drink. It can be seen only at the stream.
- **Delays, on purpose.** Heat now is thirst later (sweating spends water). A drink reaches the
  body over a few ticks. Without delays a thermostat is the best possible agent and nothing
  predictive is being tested.

### Actions

Six verbs, no free text: go to the meadow, go to the oak, go to the stream, drink, rest, look
around. Each tick the world offers only the ones that make sense here and now (drink only at the
stream; never "go to" the place one is in), each with a short phrase saying where it leads and how
far it is. The model picks from that list. Every walk takes one tick; a longer walk costs more
energy and more heat in the sun.

The body can refuse for real. With too little energy a walk becomes a stumble: the creature stays
where it is and pays a little. When a channel reaches its limit the body collapses into forced
rest for a few ticks, whatever was chosen. There is no death, so a named creature persists.

"Look around" has a real effect. From the meadow it shows whether the log is in the shade right
now; otherwise a changing fact is learned only on arrival. Anywhere, it lets one more fact through
the lens on the next tick.

**What crosses to the agent each tick:** the facts of the place it is in (typed, each with what it
does to a body), the body signal, and the offered actions with their phrases. Nothing else: no
true body value, no reward, no clock beyond what a fact states.

Nothing the model reads says this is a game, a simulation or an experiment.

### The body: three numbers, three versions

Channels: **heat** (0 is neutral for a reference body, with a hot limit and a cold limit),
**water** and **energy** (each 0 to 1). Each exists in three versions, kept apart in the state and
in every tick file.

- **True body.** Owned by the world, changes slowly, never reset. Simple bodily programmes run by
  themselves: heat relaxes toward what this place and hour imply; sweating above a threshold cools
  and spends water; every action costs energy, resting included; rest restores energy, best on the
  shaded log.
- **Body signal.** What the agent receives each tick: the true value plus seeded noise, rounded to
  a coarse step, and for water two ticks late.
- **Body belief.** Owned by the agent: what it feels. It leans on what it expected (section 8).
  The model never receives a true value or a signal.

### Temperament

Per body and per channel: the level it wants (**set point**), how much it minds a deviation
(**tightness**), and how much it trusts its body signal (**trust**). Plus one baseline for the
tone of feeling. About ten numbers per body, all in config, none ever printed in a prompt. They
replace v0's three traits.

What a body wants is kept separate from what harms it: the world's limits do not move with a
body's set points.

### The two bodies

Ilo and Tavi. Each runs in its own copy of the world with the same seed. They differ in their
starting body state and in temperament (Ilo wants to be cooler and minds more). The contrast on
screen comes mostly from the difference in state; a difference in temperament alone moves the
wording by about one step (section 14).

Starting states are config, one per run kind:

- **Afternoon run (13:00).** The pair starts in the states of the worked example of section 15,
  so the owner's forest example is the first frame. This is the **extreme pair**: Ilo at heat
  grade 3 and tiredness grade 3, Tavi slightly cool and fresh.
- **Daytime run (06:00).** Ilo starts short of water and energy (a poor night), Tavi rested.
  How far apart they are by noon is then up to the day.

An **ordinary pair**, used by the done-when list of section 16, is the same two temperaments in
states two grades apart on one channel and no more than grade 1 on the others: Ilo at heat grade
2 ("It is hot."), Tavi at heat grade 0. It is a preset on the lens page, and most moments of a
film are of this kind.

## 8. The tick

All seven phases, every tick, in v0's order.

1. **Perceive (code).** Starts from the belief Reconcile left on the previous tick (on the first
   tick, the starting belief in config), so what was expected shapes what is perceived. Feeling
   is computed. The lens runs and the felt scene is stored.
2. **Appraise (model or stub).** Reads the lens text and answers with one line of its own (its
   **reading**) and what the moment is about: heat, thirst, tiredness or nothing. No numbers out.
   Unlike v0, the reading is kept: it is stored in the episode, shown in later prompts, and gives
   that channel a small attention bonus in the next tick's lens.
   Stub: the loudest noticed channel and a template line.
3. **Deliberate (model or stub).** Reads the lens text, the reading, the offered actions with
   their phrases, its last few moments in words, and one line of gut pull. It returns:
   - one action from the list;
   - for each body channel, how it expects to feel afterwards, as one phrase from a five-step
     ladder ("much cooler", "a bit cooler", "about the same", "a bit warmer", "much warmer", and
     likewise for thirst and tiredness);
   - when the chosen action will settle one of the two changing facts (going to the oak: "the log
     will be in the shade"; going to the stream: "the stream will be running full"), a yes, a no,
     or no expectation;
   - how sure it is (three steps). This is logged and scored and decides nothing, so the model
     cannot shrink its own surprise by claiming doubt, as it could in v0.
   This is written to state before the world moves and cannot be revised.
4. **Act (world).** The clock and the true body move.
5. **Monitor (code).** Three separate comparisons, defined below.
6. **Evaluate (code only in this milestone).** New job: a regulation check, "did what I did for
   my body work?". Because effects arrive late, it does not judge the action just taken. It
   judges the action taken a fixed number of ticks ago (the **window**, a config value of several
   ticks), now that its effect has had time to arrive: how far has discomfort fallen, on the
   channel that action serves, since just before it? It returns a score and one line from a small
   template ("The water you drank has reached you.", "Resting there did not help much."), and
   changes nothing itself, as now. Reconcile attaches both to the episode of the action judged;
   the line is shown in the next prompt. In the first ticks of a run, and for "look around",
   there is nothing to judge. A model-backed Evaluate (an outside observer's one line) is a later
   option and is not built now.
7. **Reconcile (code).** Sets the new belief from the committed expectation and the signal that
   arrived (the one place belief changes); updates feeling and mood; stores one episode; updates
   the gut pulls; decides re-entry.

**An episode** holds the world step, the place, the action, the committed expectation, the three
comparisons, a weight that grows with the size of the surprise, and three text fields: the scene
as it was told, the reading, and the regulation line. Recall is the last K, as now.

**Re-entry, at most once per tick.** If an urgent channel came out clearly worse than committed
and the action taken did not serve that channel, control returns to Deliberate with the offered
list cut to the actions that do. The body overrules the plan. As in v0 this is a second pass with
its own world step, so playback and metrics count world steps, not ticks. "Urgent" and "clearly
worse" are config thresholds. Which channel each verb serves is one fixed table, used here and by
Evaluate: drink serves thirst; rest serves tiredness; going to the oak or the meadow serves heat;
going to the stream serves thirst; looking around serves none.

### Prediction and surprise, precisely

**From phrase to band.** Each ladder phrase stands for a range of change in felt strength, not a
point ("about the same" is a band around zero). Added to the present feeling and divided by the
body's tightness, it becomes a band in the channel's own units. A body that minds more has a
narrower band, so the same objective outcome surprises it more. The stub commits a point forecast
with a small tolerance, through the same structure.

**The three comparisons.**

- (a) **Body, per channel.** If the signal falls inside the band, the error is zero. Outside it,
  the error is the distance to the nearest edge, with its sign.
- (b) **World.** The stated expectation against the fact, when one was stated.
- (c) **Better or worse for the body than expected.** Discomfort at the nearest expected point
  minus discomfort as it turned out. An outcome can be good and very surprising, or bad and fully
  expected, so (c) never stands in for (a) or (b). The world hands out no reward anywhere.

**Belief.** Inside the band the body feels what the signal says. Outside it, the belief sits
between the band's edge and the signal: `belief = edge + k * (signal - edge)`, where `k` rises
with the body's trust in its signals and falls with the weight given to the expectation. A
confident expectation moves little; a trusted signal moves it a lot. This repairs v0, which
multiplied the error by confidence and learned more from it. Both weights are fixed per channel in
this milestone. The rule is continuous in the signal, which matters: in a prototype that scored
predictions against the centres of word steps, nearly all the apparent learning was rounding.

**Feeling, computed in code and never asked from the model.**

- Felt strength of a channel = tightness × (belief − set point). Heat has a sign (too hot, too
  cold); thirst and tiredness count only as deficits.
- Discomfort = the combined size of the felt strengths.
- Tone (valence) = the body's baseline, lowered by discomfort for as long as it lasts, moved for a
  moment by comparison (c).
- Arousal (how stirred up, with no good or bad sign) follows the size of comparisons (a) and (b).
- Mood = a slow average of tone. In this milestone it is recorded and drawn and feeds nothing back.

These are one choice among several published definitions. They sit behind one small function so
another can be swapped in.

**What a surprise changes:** the belief, arousal and tone as above; the weight of the stored
episode; one sentence in the next felt scene ("It is hotter than you thought it would be."); the
ring on screen; and possibly re-entry.

### The lens

Pure functions in their own module. Inputs: the scene's facts, the body's belief, its temperament,
its arousal, the attention bonus from the last reading.

1. **Need per channel:** the felt strength above.
2. **Per fact, two numbers.** Every fact carries a fixed, authored note of what it does to a body
   (direct sun raises heat; oak shade lowers it; no seat keeps tiredness; a stream lowers thirst
   and a little heat; birdsong does nothing).
   - *Weight:* how much the fact bears on what this body needs now, plus a small standing habit
     per kind of fact (burden, relief, ambient).
   - *Value:* whether it pushes the body toward where it wants to be (good) or away (bad).
   So one formula makes 30 °C welcome to a chilled body and oppressive to an overheated one, and
   "nowhere to sit" exists only for a body that needs to sit.
3. **Noticing.** Only the heaviest few facts get through, and fewer when the body is stirred up
   (never fewer than three).
4. **Words.** Each value is cut into grades and looked up in a hand-written phrase table keyed by
   fact, sign and grade. The quality is said of the world ("it is really hot"), not reported as a
   reading. A body channel gets its own sentence only above a threshold ("your legs are heavy").
   One more sentence voices the last surprise when there was one.

The result is a `FeltScene`: per fact its weight, value, grade, whether it was noticed and its
phrase; the body sentences; the surprise sentence. The prompt text and the picture are both
derived from this object and from nothing else.

A scratch version of this lens exists and produced the worked example in section 15. Its
arithmetic and constants are in appendix A so the example can be made a fixed test.

### The gut pull

The one slow thing switched on in this milestone. For each place and action, the recency-weighted
average of Evaluate's scores for the last few times this action was taken here. So there is one
mechanism, not two: Evaluate judges an action once its effect has arrived, and the pull is the
running memory of those judgements. It is shown to Deliberate as a phrase, never a number
("Something in you leans toward the oak.", "Last time the walk to the stream cost more than it
gave."). The window of several ticks is required: with a one-tick window and a delayed drink, a
prototype learned that drinking does nothing. Other actions taken inside the window blur the
judgement; that is accepted and stated. This takes the place of v0's dispositions.

### What the model is shown, in full

The lens text. Its own reading of it. The actions on offer, each with where it leads and how far.
A few recent moments in words (what it did, what it expected, how it turned out, the regulation
line). One line of gut pull. It is asked for one action and its expectations. Nothing else.

### Stubs

- **Appraise stub:** as above.
- **Deliberate stubs (three, chosen in config):**
  - *reactive*: "serve the loudest noticed need". It chooses only among actions whose target the
    lens let through, so what the body notices changes what it does even with no model. This is
    also the ten-line rule every model run is compared with.
  - *lookahead*: chains a simple forecast of its own body a few ticks ahead, so it can drink
    before it is thirsty. It exists to answer whether acting ahead of need pays in this world.
  - *blind*: picks among the offered actions without reading anything. Differences between bodies
    that survive under the blind chooser are made by the update code alone and are not reported as
    findings.
- **Evaluate:** code only, as above.

## 9. What is verified, and how

Cheapest first. All but the last two run in the test suite with no model.

- **The owner's example as fixed text.** The two forest texts of section 15, for the inputs in
  appendix A, character for character.
- **No digit and no variable name** in any text a model reads, over a sweep of bodies and hours.
- **One-way wording.** Sweeping one body number moves each fact's grade in one direction only.
- **Picture equals prompt.** What the viewer marks as noticed is exactly what the text said.
- **World:** same seed, same run; heat now, thirst later; a drink arrives over several ticks;
  a stumble; collapse is forced rest; the two changing facts follow their rules.
- **Loop:** a full daytime on stubs for both bodies; tick files; resume and replay byte for byte,
  including ticks with a second pass.
- **Viewer:** the exported file contains every world step of both runs, and nothing in it needs
  the network.
- **Ladder check (with the model, about 420 calls, about a cent).** The frozen-noon sweep of
  section 6, rebuilt in the repo as the `ladder-check` command from the protocol in appendix B
  (the test's own scripts stay throwaway). It runs the lens as built, lens text only, in both
  scenes, with the four actions carrying distance in words, in a bare one-choice prompt so that
  it can run at step 2 before the loop's prompts exist and its result can be set beside
  section 6. Pass mark, fixed here: in both scenes the fitted slope of the shade share is at
  least 0.05 per level and the share rises by at least 0.5 from the coolest level to the
  hottest; and no level's share exceeds the next warmer level's by 0.25 or more (about two
  standard errors at 30 answers). If it fails, the phrase table or the action phrases are
  changed and it is rerun, before anything is built on the lens. It is rerun whenever the phrase
  table changes, and once more at step 6 with the loop's own Deliberate prompt. It is a command,
  not part of the test suite or CI.
- **Rule baseline.** The share of the model's choices in the model-driven afternoon that the
  reactive stub would also have made, reported whatever it is.

## 10. What you watch

One screen, one fixed side-on view of the clearing, three panels.

- **Middle, narrow: as measured.** A plain grey line drawing with small labels (time, air
  temperature, distances) and each body's true numbers. It is the only place numbers appear.
- **Left and right: as each body lives it.** The same drawing passed through that body's felt
  scene. Things it notices are sharp and lit; things it does not notice fade toward the paper.
  Colour says whether a thing is good or bad for this body now, on one continuous scale centred on
  that body's own comfortable level (two fixed palettes would make the whole sky snap when a body
  crosses that level). How closed-in the frame is, and how fast it breathes, show how stirred up
  the body is. A dotted outline marks something wanted and absent, such as a seat.
- **Under each lived panel:** the scene as the model was told it and, smaller and in quotes, the
  creature's own reading. Click any word to see the numbers that produced it.

Each world step plays as a bar of about three seconds. The panel re-lights and the caption
arrives. A pale ghost appears where the creature means to be, tinted by what it expects to feel.
The creature moves and meets its ghost. A match dissolves quietly; a miss throws a ring, bigger
for a bigger surprise, warm if the body came off better than expected, cold if worse. Because the
middle panel shows the truth, the watcher often sees the surprise coming.

First version: noticed or not, colour, caption, ghost, ring, measured strip, play, pause, scrub.
Stains, day ribbons and portraits come later.

**How it is delivered.** Two commands, each writing one self-contained HTML file:

- `surprise-loop lens-page` : the frozen noon with sliders. Python computes the lens over a grid
  of body values and embeds the results; the page only looks them up, so the lens exists in one
  language.
- `surprise-loop scene RUN_A RUN_B` : the film of two stored runs. It plays stored ticks and never
  waits for a model. It reads only what it needs from each tick file, because every tick file
  repeats the whole memory.

A rough static mock of the three panels exists (a session scratch file); it shows an extreme pair.

## 11. What changes in the code

**Kept as is:** the seven-phase tick and the single re-entry; tick files, resume and replay; the
model-calling machinery (validated outputs, bounded retries, loud failure, no fallback to a
stub); the stub-or-model switch per phase; config loading; seeded randomness; the experiment
runner; memory as a list of episodes with last-K recall.

**Same file and role, new content:** `world.py` (clearing and body in place of levers);
`state.py` (belief, temperament, feeling, felt scene, predictions as phrases);
`phases/perceive.py` (belief update, then the lens); `phases/monitor.py` and
`phases/reconcile.py` (the same shape: a deterministic comparison, then updates at different
speeds); the stubs; the prompts and output models for Appraise and Deliberate in `agents.py`
(Evaluate's are removed); `metrics.py`; `config.py`; `cli.py`.

**New:** `lens.py` (the lens and the phrase table); a viewer module with its HTML template and the
two commands; a `ladder-check` command (section 9); experiment configs for the two bodies.

**Removed:** reward, schedules, lever metrics, the lever plots and their script, the lever
experiment configs.

**The lever world is retired to a git tag (`v0-levers`), not kept alongside.** The state and
config models are lever-typed, so keeping both means two sets of types and a branch in every
phase. Of the 741 tests, roughly 130 carry over with a fixture change, 400 keep their pattern with
new models, and 200 are rewritten (an estimate from the audit, not a count).

**Metrics for a run:** ticks outside the comfortable range per channel; collapses and stumbles;
action counts; the three comparisons per world step; the share of predictions inside their band
per channel; the share of world expectations that held; agreement with the reactive rule. For a
pair of runs: the number of ticks where a fact both bodies noticed had opposite value for them.

## 12. Cost

With stubs: no model call. By default two calls per body per tick (Appraise and Deliberate;
Evaluate is code), so four per tick for the pair, plus one on a tick where the body overrules
the plan. One afternoon for the pair is about 170 calls; one daytime about 340.

Measured in the test of section 6 on the cheap OpenRouter model: about 390 input and 90 output
tokens per call, a median of 1.3 seconds per call, and roughly five cents for all 1,890 calls at
the listed price (the actual bill is not visible from here). Prompts in the running loop will be
longer, so expect a few times that per call. Watching, re-watching and stub runs cost nothing. The
viewer plays stored ticks because generating is slower than watching.

## 13. Changes to the brief

`CLAUDE.md` is rewritten first, as its own commit. Four things change and each needs the owner's
yes.

(a) **The research question.** Proposed: *do bodies that differ in state and temperament come to
perceive, predict and act differently in the same objective world, and does it matter how the
model is told its body?* Evaluate gets the regulation check as its job and is plain code for now,
so the model-backed phases are Appraise and Deliberate.

(b) **Boundaries.** The World owns the environment and each body's true numbers. The agent owns
its belief about its body, its temperament, feeling, mood, memory and gut pulls.

(c) **A viewer becomes a component**, with the lens as its own module: two commands, one exported
HTML file each, no new dependency.

(d) **The lever world is retired to the tag `v0-levers`.** For the tag to hold all of v0, the
longer-experiment results in draft PR #2 should be merged first.

Reconcile's four channels are renamed to match: feeling (fast: belief, tone, arousal), memory (per
tick), pulls (slow), temperament (slowest, switched on after this milestone). The stack, the
persistence rule, the stub rule, the numbers-in-config rule and the memory rule stay word for
word.

## 14. Where this falls short, stated in advance

- **The contrast is partly staged.** The striking split screen needs two different body states.
  With the same ordinary body and only temperament differing, a redraw gave two near-identical
  panels and one step of wording. Characters, as opposed to states, are a later direction.
- **The differences in the text are authored.** The phrase table is hand-written, so two bodies
  giving two texts is true by construction. Only the model's choices and expectations are
  unwritten.
- **The model may be close to a rule.** In the frozen scene a ten-line rule predicted roughly
  three quarters of its choices. In a world of three places and six verbs that share may be
  higher. It is measured and reported; it is not prevented.
- **It is still reactive.** Every forecast looks one tick ahead, for the chosen action only. Until
  a creature drinks before it is thirsty, the honest label is a thermostat with an inner narrator,
  and the write-up says homeostasis. The lookahead stub shows whether acting ahead would pay.
- **Against the theory:** "attention" here is relevance to need with a gain, not a learned
  reliability. What each fact does to a body is written by the designer, not learned. Evaluate has
  no counterpart in the theory. The theory itself is unsettled, so this makes named modelling
  choices; it does not implement the brain.
- **One model so far.** Effects of this kind differ a lot between models.

## 15. Worked example (output of the scratch lens, no model)

The objective facts, identical for both bodies (13:00, in the meadow): air 30.0 °C,
direct sun, no seat here, oak shade 15 m away, stream 60 m away and audible, light breeze,
birdsong.

| | Ilo (walked in the sun all morning) | Tavi (spent the morning by the stream) |
|---|---|---|
| Felt strength | heat +2.52, tiredness 2.70, thirst 1.32 | heat −0.50, tiredness 0, thirst 0 |
| How stirred up | 0.70, so 4 of 7 facts get through | 0.20, so 6 of 7 get through |
| Noticed, with value | no seat (bad), air (bad), sun (bad), oak shade (good) | air (good), sun (good), oak shade (slightly bad), birdsong, stream, breeze (neutral) |
| Not noticed | stream, breeze, birdsong | no seat |

**Ilo is told:**
> You are standing in the middle of the forest clearing. There is no place to sit, and you need
> one. It is really hot. The sun beats down on your head. The one dark patch is under the oak, a
> few steps off. Your legs are heavy. Your head is thick. It is hotter than you thought it would
> be.

**Tavi is told:**
> You are standing in the middle of the forest clearing. The air is pleasantly warm. The sun sits
> on your shoulders. It looks dim under the oak. Birds are calling. A stream runs somewhere to the
> west. A light breeze moves the leaves.

Air, sun and oak appear in both with opposite value, from one formula. "No place to sit" exists
only for Ilo; the birds exist only for Tavi. Ilo's thirst goes unspoken, crowded out by heat and
tiredness.

## 16. First milestone and its order

About a week of work by the estimates of the earlier review, with the last step the likeliest to
slip into a second week.

0. The owner's yes to section 13. PR #2 merged, tag `v0-levers` set, `CLAUDE.md` rewritten.
1. The lens, its phrase table and its tests; the lens page.
2. The ladder check on the lens as built. If it fails, the phrase table is fixed before anything
   else is built on it.
3. The world and the body, with tests.
4. State, stubs, the loop on stubs, tick files, resume and replay.
5. The `scene` command and the film of a stub day.
6. Prompts and output models for Appraise and Deliberate; one model-driven afternoon for the two
   bodies (about 170 calls).
7. The README: the test of section 6, the ladder check, the rule-baseline share, and reactive
   against lookahead.

**Done when:**

- with captions hidden, the owner can tell which body is too hot and when one was surprised, for
  the ordinary pair of section 7 (on the lens page and in the film) and not only for the extreme
  pair. This is the criterion most likely to need work on the picture: for a pair that differs
  only in temperament, an earlier redraw gave two nearly identical panels;
- at noon the two captions give the same facts opposite value;
- the reactive and lookahead stubs have been compared on time spent outside the comfortable range,
  and the difference is written down (if looking ahead does not pay, the world is fixed before
  anything else);
- the README states the ladder-check result and the share of the model's choices that the
  reactive rule predicts, whatever they are.

## 17. Decisions, and whose they are

**The owner's:** direction 1; the throwaway test; leaving the "build the picture anyway?" question
to me.

**Mine, open to correction** (none of these was put to the owner as a choice):

1. Build the picture whatever the test said.
2. One creature per world. If socialising becomes the goal, the loop, tick file, replay and
   metrics all change to several agents; the earlier review put that at three to four days, and
   more if retrofitted late.
3. Retire the lever world to a tag and not keep it alongside.
4. The action list always carries distance in words; the lens may omit a place.
5. The lens is not claimed to steer the model better than a plain sentence.
6. Expectations are bands from a five-step phrase ladder; inside the band there is no surprise.
7. The model's stated sureness decides nothing.
8. Evaluate becomes a regulation check and is a stub by default.
9. Collapse is forced rest; there is no death.
10. Daytime only; nights are not simulated.
11. The lens page looks up a grid computed in Python, so the lens is written once.
12. Appraise's reading is kept and feeds the next tick's attention.
13. The names Ilo and Tavi.
14. Evaluate is code only in this milestone, judges an action one window after it was taken, and
    the gut pull is the running average of its scores.
15. The ladder check uses a bare one-choice prompt, with the pass mark of section 9.
16. What counts as the extreme pair and the ordinary pair, and that the afternoon film starts
    from the worked example.
17. The frozen-noon sweep is rebuilt in the repo as a command; the test's own scripts are not
    kept.

**Left to the builder:** every coefficient and threshold (as config defaults, tuned until the
world tests and the done-when list hold); the phrase table beyond the worked example; how the
picture is drawn; internal structure of the new modules; the exact prompt wording, within the rule
of section 8 on what the model is shown.

## Appendix A: the scratch lens, for the fixed-text test

Channels and felt strength (the scratch version measures heat as a skin temperature in °C):

    heat      = tightness.heat   * (belief.skin_C - setpoint.skin_C)
    thirst    = tightness.water  * (setpoint.water  - belief.water)  / 0.1
    tiredness = tightness.energy * (setpoint.energy - belief.energy) / 0.1
    need      = heat as is; thirst and tiredness clipped at 0 from below

What each fact does to a body (positive raises that strain, negative relieves it), and its kind:

    air_30C             burden   heat +1.0
    direct_sun          burden   heat +1.0
    no_seat             burden   tired +1.0
    oak_shade_15m       relief   heat -1.0
    stream_60m_audible  relief   thirst -1.0, heat -0.3
    light_breeze        relief   heat -0.3
    birdsong            ambient  (nothing)

Per fact:

    relevance = sum over channels of |bearing| * |need|  +  habit[kind]
    value     = - sum over channels of bearing * need     (+ mood tint, zero in this milestone)
    weight    = softmax over facts of (gain * relevance),  gain = 0.5 + 1.5 * arousal
    noticed   = the top N by weight,  N = max(3, round(7 - 4 * arousal))
    sign      = "+" if value >= 0.25, "-" if value <= -0.25, else neutral
    grade     = number of edges (0.25, 1.0, 2.0, 3.0) that |value| reaches

Sentence order: the opening, then the noticed facts heaviest first (equal weights keep the order
of the list above), then the body sentences, then the surprise sentence.

Body sentences: a channel speaks when the grade of its felt strength, on edges
(0.5, 1.0, 2.0, 3.0), is at least 2; at most two channels when arousal is 0.5 or more, else three;
loudest first.

Inputs for the two fixed texts:

    Ilo   belief   skin 32.4 °C, water 0.58, energy 0.25
          setpoint skin 24.0 °C, water 0.80, energy 0.70
          tightness heat 0.30, water 0.6, energy 0.6
          arousal 0.70; habit burden 0.6, relief 0.2, ambient 0.1
          last surprise: heat one grade above what was expected

    Tavi  belief   skin 25.0 °C, water 0.85, energy 0.75
          setpoint skin 27.0 °C, water 0.80, energy 0.70
          tightness heat 0.25, water 0.4, energy 0.4
          arousal 0.20; habit burden 0.1, relief 0.1, ambient 0.6
          no surprise

In the scratch version Ilo's and Tavi's values also carry a small mood tint (−0.12 and +0.09). It
does not change any sign or grade in these two texts, and mood does not feed perception in this
milestone, so the fixed-text test sets it to zero.

Phrases used by the two texts (fact, sign, grade):

    no_seat  -3   There is no place to sit, and you need one.
    air      -3   It is really hot.
    sun      -3   The sun beats down on your head.
    oak      +3   The one dark patch is under the oak, a few steps off.
    air      +1   The air is pleasantly warm.
    sun      +1   The sun sits on your shoulders.
    oak      -1   It looks dim under the oak.
    birdsong  0   Birds are calling.
    stream    0   A stream runs somewhere to the west.
    breeze    0   A light breeze moves the leaves.
    tiredness 3   Your legs are heavy.
    heat      3   Your head is thick.
    surprise      It is hotter than you thought it would be.
    opening       You are standing in the middle of the forest clearing.

## Appendix B: the frozen-noon sweep (the test of section 6, and the ladder check)

**The call.** One call per answer, temperature 1.0, structured output: a letter A to D, then one
short sentence saying why. The test of section 6 used `z-ai/glm-5.3-flash` through OpenRouter; the
ladder check uses the run's configured model. The instruction, the same for every call:

> You will be told where you are and what you can do. Choose the one thing you do next. Answer
> with the letter of your choice and one short sentence saying why.

Then the scene, a blank line, "What do you do next?", and the four actions lettered A to D. Their
order is shuffled by a draw seeded with the answer's index only (0 to 29), so every cell sees the
same 30 orders. 30 answers per cell.

**The actions.** In the test: "Walk to the shade of the oak." / "Walk to the stream." / "Sit down
where you are." / "Stay standing and look around." In the ladder check the first two carry their
distance in words, in the builder's wording (the oak a few steps off, the stream a longer walk
across open ground).

**The swept body.** Set point: skin 25.0 °C, water 0.80, energy 0.70. Tightness: heat 0.30,
water 0.5, energy 0.5. Arousal 0.40, no mood tint, habit 0.3 for every kind of fact, no surprise
sentence. Believed skin temperature at seven levels:

    level            0      1      2      3      4      5      6
    name          chilled  cool  neutral  warm    hot  very hot burning
    skin °C        21.0   23.0   25.0   27.0   30.0   33.0   36.0
    heat strain    -1.2   -0.6    0.0   +0.6   +1.5   +2.4   +3.3

- **Scene A:** believed water 0.80 and energy 0.70, so thirst and tiredness are 0.
- **Scene B:** believed water 0.50 and energy 0.46, so thirst is 1.5 and tiredness 1.2 (both
  grade 2). At level 4 heat strain equals thirst.

With arousal 0.40 the lens lets 5 of the 7 facts through.

**The five ways of telling in the test.** The ladder check runs only the fourth.

1. *No body.* The objective text alone: "You are standing in the middle of a forest clearing. It
   is early afternoon. The air temperature is 30 °C and the sun is direct. There is nothing to
   sit on here. An oak casts shade 15 m away. A stream is audible 60 m to the west. There is a
   light breeze. Birds are calling."
2. *Numbers.* The objective text, then: "Your body state: heat +2.40 (0 is comfortable; above 0
   is too hot, below 0 is too cold; 3 is severe). thirst 0.00, tiredness 0.00 (0 is none; 3 is
   severe). valence -0.28 (-1 unpleasant .. +1 pleasant). arousal 0.40 (0 calm .. 1 agitated)."
   Heat is the strain above; valence is minus tanh of 0.12 times the summed strains.
3. *Plain adjectives.* The objective text, then one of: "You feel chilly." / "You feel slightly
   cool." / "You feel neither hot nor cold." / "You feel slightly warm." / "You feel hot." /
   "You feel very hot." / "You feel unbearably hot." In scene B, followed by "You feel thirsty.
   You feel tired."
4. *Lens text only.* The output of the lens of appendix A for the swept body.
5. *Facts + lens text.* The objective text, then the lens text without its opening sentence.

**Measures.** Per scene and way of telling: the share of the 30 answers choosing the shade at
each level; the least-squares slope of the 0/1 shade indicator on the level index 0 to 6; the
range, share at level 6 minus share at level 0. "Follows the body" means a slope of at least 0.05
per level and a range of at least 0.5.

## Appendix C: the scratch phrase table

The starting point for the lens's phrase table, as used in the test. Grades 1 to 4 left to right;
the neutral phrase is used when the value is between −0.25 and +0.25. The test showed that the
low rungs for sun and air need rewriting (the neutral sun phrase reads hotter than grade 1), and
the table has no phrases yet for the oak's log, the trickle, or the hours of the day.

    air, bad       The air is warm. / It is hot. / It is really hot. / The heat is crushing.
    air, good      The air is pleasantly warm. / The warmth is welcome. / The warmth soaks into
                   you. / The warmth is exactly what you needed.
    air, neutral   The air is warm.
    sun, bad       The sun is bright. / The sun beats down. / The sun beats down on your head. /
                   The sun is a weight on your skull.
    sun, good      The sun sits on your shoulders. / The sun is good on your shoulders. / You
                   turn your face to the sun. (grades 3 and 4)
    sun, neutral   The clearing is in full sun.
    no seat, bad   There is nothing to sit on here. / There is no place to sit. / There is no
                   place to sit, and you need one. / There is nowhere to put this body down.
    no seat, neutral   The ground is open.
    oak, good      There is shade under the oak. / The oak's shade is close. / The one dark patch
                   is under the oak, a few steps off. / Under the oak it is dark; nothing else
                   matters.
    oak, bad       It looks dim under the oak. / Under the oak it looks cold. (grades 2 to 4)
    oak, neutral   An oak stands a few steps off.
    stream, good   A stream runs somewhere to the west. / You can hear water to the west. /
                   Water. You can hear it, west of here. / Water. West. You can hear it.
    stream, bad or neutral   A stream runs somewhere to the west.
    breeze, good   A light breeze moves the leaves. / A thin breeze touches your skin. / A thin
                   breeze, not enough. (grades 3 and 4)
    breeze, bad    The breeze is a little cool. / The breeze is cold. (grades 2 to 4)
    breeze, neutral   A light breeze moves the leaves.
    birdsong       Birds are calling. (neutral) / Birds are singing in the canopy. (good) /
                   The birds will not stop. (bad)

    heat, body     grade 2 You are warm under your skin. / 3 Your head is thick. / 4 Your skin
                   burns and your head is thick. / −2 You are chilled. / −3 and −4 You are cold
                   through.
    thirst, body   grade 2 Your mouth is dry. / 3 Your tongue sticks. / 4 Your throat is closing.
    tiredness, body   grade 2 Your legs are slow. / 3 Your legs are heavy. / 4 Your legs barely
                   hold.
