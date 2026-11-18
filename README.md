# surprise-loop

An embodied LLM agent driven by prediction error. It has a simulated body it must keep in
balance. Each tick it reads its world as that body lives it, says how it expects its body to feel
after its next action, acts, and is corrected by what happens. The size of the miss is the
surprise.

This is faked embodiment, a working analogue. Nothing here supports the claim that an agent
feels.

- The brief: [`CLAUDE.md`](CLAUDE.md)
- The design, with the test that motivated it:
  [`docs/superpowers/specs/2026-11-17-lived-twice-design.md`](docs/superpowers/specs/2026-11-17-lived-twice-design.md)
- The build plan and its contracts:
  [`docs/superpowers/plans/2026-11-17-lived-twice.md`](docs/superpowers/plans/2026-11-17-lived-twice.md)

## Research question

Do bodies that differ in state and temperament come to perceive, predict and act differently in
the same objective world, and does it matter how the model is told its body?

## What is here

One forest clearing with three places (meadow, oak, stream), a clock from 06:00 to 20:00 in steps
of ten minutes, and a body with three numbers: heat, water and energy. The world is plain Python
under a seed and owns each body's true numbers. Two bodies, Ilo and Tavi, live the same seeded day,
each in its own copy of the world.

The model never sees a number. A piece of code called the lens turns the facts of the place and a
body's belief about itself into a short text: which facts this body notices, what each is worth to
it, and the words for that. The same 30 °C meadow at 13:00 reads like this for the two bodies of
the worked example:

> **Ilo** (overheated, tired): You are standing in the middle of the forest clearing. There is no
> place to sit, and you need one. It is really hot. The sun beats down on your head. The one dark
> patch is under the oak, a few steps off. Your legs are heavy. Your head is thick. It is hotter
> than you thought it would be.

> **Tavi** (comfortable, a little cool): You are standing in the middle of the forest clearing.
> The air is pleasantly warm. The sun sits on your shoulders. It looks chilly under the oak. Birds
> are calling. A stream runs somewhere to the west. A light breeze moves the leaves.

Each tick runs seven phases. Perceive builds that text. Appraise gives one line of the creature's
own reading. Deliberate picks one offered action and commits an expectation: for each of heat,
thirst and tiredness a phrase from a fixed ladder ("a bit cooler", "much less thirsty"), and, where
the action will settle an open fact, a yes or no ("will the log be in the shade?"). Act moves the
world. Monitor compares what came back with what was expected. Evaluate checks whether an action
taken a few steps ago eased the need it serves. Reconcile updates belief, feeling, memory and pulls,
and may send control back to Deliberate once when an urgent need came out worse than expected.
Appraise and Deliberate can run on a model or on a rule; everything else is code.

Every tick is written to `runs/<run_id>/tick_<n>.json`. A run can be resumed, and replayed byte for
byte without calling a model.

## Run it

    uv sync --all-groups
    uv run ruff check . && uv run ruff format --check . && uv run pytest

Without a model (rules in place of Appraise and Deliberate):

    uv run surprise-loop experiment experiments/clearing_afternoon.yaml
    uv run surprise-loop scene runs/clearing_afternoon-ilo-s1 runs/clearing_afternoon-tavi-s1 -o film.html
    uv run surprise-loop lens-page experiments/clearing_afternoon.yaml -o lens_page.html

`film.html` and `lens_page.html` are self-contained pages; open them in a browser. The film plays
the two runs side by side: each lived panel shows what that body notices and how it values it, a
pale ghost shows where the creature means to be, and a ring is thrown when the world answers
otherwise. The narrow middle panel is the world as measured.

With a model (the name comes from the environment, never from the repository):

    export SURPRISE_LOOP_MODEL=openrouter:<model id>      # and that provider's API key
    uv run surprise-loop ladder-check                     # 420 calls
    uv run surprise-loop ladder-check --deliberate        # 420 calls
    uv run surprise-loop experiment experiments/clearing_afternoon_llm.yaml --jobs 2   # about 170 calls
    uv run surprise-loop replay runs/clearing_afternoon_llm-ilo-s1

Other commands: `run` (one body and seed), `metrics` (the measures of a stored run).

## What was shown

Every model result below is from one model, `z-ai/glm-5.3-flash` through OpenRouter. Results of
this kind differ a lot between models.

### 1. Numbers do not carry a body; words do

Before anything was built, one frozen scene was put to the model 1,890 times: the 30 °C meadow,
four actions, and a body whose believed heat was swept over seven levels from chilled to burning.
The body was told in five ways. With the body given as numbers and heat the only need, 93% of
answers walked to the shade even for a chilled body. With a thirst and a tiredness competing, the
numbers did move the choice, but not in order (67% shade for a chilled body, 17% for a cool one).
With the body given in words, the share of shade answers followed the body. The full table and
the pre-set marks are in section 6 of the design. Two findings from it still stand as limits: the lens text steered no
better than one plain sentence about the body, and a ten-line rule predicted 72 to 80% of the
model's answers in that scene.

### 2. The ladder check

`ladder-check` repeats that sweep on the lens as built: a frozen noon in the meadow, seven levels
of believed heat, 30 answers per level, in two scenes. In scene A heat is the only need. In scene B
heat competes with a fixed thirst and tiredness. The check passes when the share of answers that
walk to the oak's shade rises with heat: a slope of at least 0.05 per level, a range of at least
0.5, and no fall of 0.25 or more between neighbouring levels.

Share of shade answers by level, coolest first:

| prompt | wording | scene A | scene B | result |
|---|---|---|---|---|
| plain | "dim" | 0.00 0.83 0.83 0.97 1.00 1.00 1.00 | 0.00 0.63 0.67 0.67 1.00 0.97 1.00 | passed |
| Deliberate | "dim" | 0.00 1.00 0.90 1.00 1.00 1.00 1.00 | 0.00 0.57 0.23 0.60 0.83 0.80 0.83 | **failed** (fall of 0.33) |
| plain | "chilly" | 0.00 0.40 0.83 0.90 1.00 1.00 1.00 | 0.00 0.27 0.67 0.83 0.97 0.97 1.00 | passed |
| Deliberate | "chilly" | 0.00 0.93 0.90 1.00 1.00 1.00 1.00 | 0.00 0.43 0.23 0.60 0.83 0.80 0.83 | passed (fall of 0.20) |

The plain prompt is the scene text, a question and four actions. The Deliberate prompt is the one
the agent really uses, with the reading, the lettered offers and the questions about what it
expects.

The first run through the Deliberate prompt failed. A cool body was told "It looks dim under the
oak", and the model took that as an invitation ("the oak already looks dim so its log should be
shaded"). The sentence was changed to "It looks chilly under the oak", and both checks were run
again (in the Deliberate check only the 60 answers whose prompt had changed were asked again). All
four curves then pass.

The second rung is still weak in the Deliberate prompt. A cool body with no other need walks to
the shade in 28 of 30 answers, and gives reasons such as "the oak looks chilly, so its shade should
cool me". The model reads "pleasantly warm" and "the sun sits on your shoulders" as something to
cool down from. The bottom rung, where the text has a sentence about the body itself ("You are
chilled"), works: no answer walks to the shade. So a body's mild liking for warmth, told only
through adjectives on the facts, does not reach this model's choice; a sentence about the body
does.

### 3. A day on rules: looking ahead pays

Three rules can stand in for Deliberate. `reactive` serves the loudest need now. `lookahead` tries
each offered action and a few steps after it on its own simple forecast and takes the least
discomfort. `blind` picks at random and reads nothing. Whole days, 84 steps, mean over seeds 1 to
20, counted as channel-steps outside the comfortable range (a step counts once for each of heat,
water and energy that is out of range, so a day has 252):

| rule | Ilo uncomfortable | Ilo collapsed steps | Ilo walks | Tavi uncomfortable | Tavi walks |
|---|---|---|---|---|---|
| reactive | 65.1 | 0 | 21.7 | 1.1 | 13.5 |
| lookahead | 21.2 | 0 | 33.3 | 0.2 | 29.0 |
| blind | 140.4 | 12.7 | 41.0 | 85.6 | 39.3 |

Looking ahead cuts Ilo's uncomfortable channel-steps to about a third of the reactive rule's. The
reactive rule falls into a trap around noon: it shuttles Ilo between the oak and the stream and
arrives hot at the water, where the body overrules it and sends it back (3.5 overrides a day; the
lookahead rule has almost none, because it drinks before noon).

The blind row matters for reading everything else. Two bodies come out different even under a
chooser that reads nothing, because the world and the update code treat different bodies
differently. A difference between Ilo and Tavi is therefore not, by itself, a finding about a
model.

### 4. One afternoon on a model

`experiments/clearing_afternoon_llm.yaml`, seed 1, 13:00 to 20:00, 42 steps for each body, 166
model calls (82 readings and 84 choices). Both runs replay. The rule rows are the same
afternoon and seed with a rule in place of the model.

| body | chooser | uncomfortable heat / water / energy | collapsed steps | expectation held heat / water / energy | mean size of miss | reactive rule predicts |
|---|---|---|---|---|---|---|
| Ilo | model | 27 / 9 / 31 | 3 | 0.69 / 0.52 / 0.57 | 0.46 | 43% |
| Ilo | reactive | 22 / 11 / 17 | 0 | 0.81 / 0.86 / 0.83 | 0.17 | 100% |
| Ilo | lookahead | 18 / 16 / 10 | 0 | 0.90 / 0.81 / 0.93 | 0.11 | 31% |
| Ilo | blind | 36 / 35 / 42 | 6 | 0.83 / 0.76 / 0.79 | 0.21 | 36% |
| Tavi | model | 1 / 0 / 0 | 0 | 0.64 / 0.57 / 0.50 | 0.42 | 24% |
| Tavi | reactive | 0 / 1 / 1 | 0 | 0.81 / 0.88 / 0.98 | 0.09 | 100% |
| Tavi | lookahead | 2 / 0 / 0 | 0 | 0.81 / 0.90 / 0.98 | 0.06 | 21% |
| Tavi | blind | 20 / 0 / 27 | 0 | 0.71 / 0.90 / 0.95 | 0.07 | 26% |

Model minus reactive rule, same seed: Ilo spent 17 more channel-steps outside its comfortable
range (5 more for heat, 2 fewer for water, 14 more for energy) and collapsed on 3 steps where the
rule never did. Tavi was comfortable under both.

What that says, and what it does not:

- **The model kept the hot body worse than a ten-line rule did,** and far better than a chooser
  that reads nothing. It is one afternoon and one seed; the size of the gap is not established.
- **The model is not the reactive rule.** The rule predicts 43% of Ilo's choices and 24% of
  Tavi's, about what it predicts of the blind chooser (36% and 26%). In the frozen scene of
  result 1 a rule predicted three quarters. A living afternoon with memory in the prompt is a
  different situation, and here the model's choices are its own. They are also not better.
- **Its expectations failed more often than the rules' forecasts,** and that is the point of the
  loop: the misses are real. The largest ones were the same for both bodies: drinking. The world
  delivers water two steps late, the model expected to be "much less thirsty" at once, and it
  said so afterwards ("Drinking three times has not eased my thirst, so resting in the pleasant
  spot is the sensible choice"). Ilo's first step was also a miss: it walked to the oak expecting
  the log to be in the shade, and at 13:00 it is in the sun.
- **It committed expectations about the world,** which the rules never do: Ilo 7 (4 held), Tavi 5
  (4 held).
- **The two bodies read their moments differently.** Ilo's 40 readings were about tiredness 17
  times, heat 7, thirst 7 and nothing 9; Tavi's 42 were about nothing 32 times. Ilo at 13:00: "The
  sun is hammering down on me, and all I can think about is that patch of shade under the oak".
  Tavi at 13:00: "The warmth settles on me like a friendly hand". These texts are downstream of
  the lens's authored phrases, so that they differ is true by construction; what the model adds
  is the reading and the choice.
- **The overrides did not help.** Ilo was overruled twice, on its first two ticks. At 13:00 it
  walked to the shade, which serves heat. Its thirst came out worse than it had expected (it had
  said "about the same", and the worked example starts Ilo believing it is less thirsty than it
  is), the override cut its choices to the stream, and it crossed the open ground at the hottest
  time of day. At the stream it drank, expecting to be "a bit less tired" as well; tiredness came
  out worse, and the override made it sit on the bank in the sun. Neither first choice was a bad
  one. The rule fires when an urgent need comes out worse than expected, and it cannot tell "the
  world answered worse" from "my belief, or my hope, was off". That is a weakness of the rule as
  designed, and it is left as it is for now.
- **No planning ahead was seen.** Later, exhausted at the stream, Ilo chose to drink five times
  in a row; on three of those steps its body gave way instead. It then rested on the bank until
  evening. Nothing in the afternoon looks like drinking before being thirsty.

## What was not shown

- **No stable individual differences.** One seed, one afternoon, one model. There is no claim
  that the two bodies differ beyond what their starting states and the update code give them.
- **The contrast is partly staged.** The split screen is striking for an overheated body next to
  a comfortable one. In the cool morning of a rule-driven day the two panels look much alike.
  Whether a watcher can tell the bodies apart with the captions hidden, for an ordinary pair, has
  not been checked by anyone but the builder.
- **The differences in the text are authored.** The phrase table is written by hand, so two
  bodies giving two texts is true by construction.
- **The cool rung of the ladder does not carry** in the prompt the agent really uses (result 2).
- **It is still reactive.** Every expectation looks one step ahead, for the chosen action only.
  The honest label is homeostasis with an inner narrator.
- **The rules' forecast is biased.** The rules count the lens's value for the oak's shade into
  their heat forecast, which the world's heat rule does not have. A slow forecast and a wide band
  keep it from showing as error. The rules never commit an expectation about the world.
- **The body signal is nearly exact.** Surprise comes from the world's delays and changing facts
  (water that arrives late, a log that moves into the sun), not from noisy senses.
- **Small leaks of realism in what a model reads.** Every scene opens "You are standing",
  whatever the body did last. After an invalid answer the retry message contains the library's
  error text. A model's own words are shown back to it unfiltered.
- **Against the theory.** "Attention" here is relevance to need with a gain, not learned
  reliability. What each fact does to a body is written by the designer, not learned. Evaluate
  has no counterpart in the theory. Temperament does not change (that channel is switched off in
  this milestone).

## The design's done-when list

- *With captions hidden, the owner can tell which body is too hot and when one was surprised, for
  the ordinary pair and not only the extreme pair.* Not established. It is the owner's call, and
  the builder's own look says the ordinary pair is hard to tell apart in the morning.
- *At noon the two captions give the same facts opposite value.* Yes: the two texts above. Over
  the afternoon a fact noticed by both bodies had opposite value on 7 steps of 42 under the
  reactive rule and on 6 under the model.
- *The reactive and lookahead rules have been compared on time outside the comfortable range.*
  Yes: result 3. Looking ahead pays.
- *The README states the ladder-check result and the share of the model's choices that the
  reactive rule predicts, whatever they are.* Results 2 and 4.

## The first version (v0)

v0 was a lever-pressing world: an agent pressed levers for reward under changing schedules, and
the question was whether agents with different trait numbers showed stable differences in
behaviour. All of it is at commit `2ec53fd`:

    git show 2ec53fd:README.md      # the full v0 write-up, with commands and results
    git checkout 2ec53fd            # the v0 code and its 741 tests

Its result with a real model was "not yet": trait numbers in the prompt barely reached behaviour,
and the measures that stayed stable were made by the update code, not by the model's choices.
Result 1 above suggests why.
