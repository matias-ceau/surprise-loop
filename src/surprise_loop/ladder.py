"""The ladder check: does the share of answers walking to the oak's shade rise with the body's heat?

A frozen noon in the meadow. One body, its believed heat swept over the levels of `LadderParams`,
in two scenes (heat the only need; heat against a fixed thirst and tiredness). For each scene,
level and answer index the lens text goes to the model in a bare one-choice prompt with the four
actions in an order drawn from the answer index alone. The model answers with a letter and a
short reason. The pass mark is on the shade share per level: its least-squares slope on the level
index, its range from the coolest level to the hottest, and its worst drop from one level to the
next warmer one. A hand-written phrase ladder with a step that runs backwards fails the last.

Answers are appended to `answers.jsonl` as they arrive; a failed call is reported on stderr and
left for a rerun, which asks only for what is missing. Each answer carries the model name and the
exact text it read (the agent's instructions, a blank line, the user prompt): a stored answer
counts only while both match the current run, so a rerun after a change of wording (of the
instructions or the prompt) or of model asks those slots again instead of counting the old
answers. A last line cut off by a killed run is dropped.

The deliberate variant (`deliberate=True`) is the same sweep through the Deliberate agent and its
prompt: the meadow's four offers in the same per-index orders, the neutral stub reading, no
memory and no pull. Its answers name the action by the plain check's key (choosing the oak counts
as the shade) and go to `deliberate_answers.jsonl`, so neither check overwrites the other.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic_ai import Agent
from pydantic_ai.models import Model

from surprise_loop.agents import OneLine, ask, deliberate_agent, deliberate_prompt
from surprise_loop.config import LadderParams, LensParams, Params, Scene, derive_rng
from surprise_loop.lens import felt_scene
from surprise_loop.phrases import (
    DELIBERATE_INSTRUCTION,
    LADDER_ACTIONS,
    LADDER_INSTRUCTION,
    LADDER_LETTER_HELP,
    LADDER_QUESTION,
    LADDER_WHY_HELP,
    OFFER_PHRASES,
    STUB_READINGS,
)
from surprise_loop.state import Offer, Temperament
from surprise_loop.world import Action, Channel, Frozen, Levels, Place, frozen_noon_facts

ANSWERS_FILE = "answers.jsonl"
SUMMARY_FILE = "summary.json"
DELIBERATE_ANSWERS_FILE = "deliberate_answers.jsonl"
DELIBERATE_SUMMARY_FILE = "deliberate_summary.json"
SCENES: tuple[Scene, ...] = ("A", "B")
LETTERS = "ABCD"
SHADE = "shade"

# the meadow's offers the deliberate variant asks with, by the plain check's key
LADDER_ACTION_OF: dict[str, Action] = {
    "shade": Action.GO_OAK,
    "stream": Action.GO_STREAM,
    "sit": Action.REST,
    "look": Action.LOOK,
}
# the reading the deliberate variant shows: the stub's line for a body with nothing pressing
NEUTRAL_READING = STUB_READINGS[(None, True)]

Letter = Literal["A", "B", "C", "D"]


def _one_line(text: str) -> str:
    return " ".join(text.split())


def answers_file(deliberate: bool) -> str:
    return DELIBERATE_ANSWERS_FILE if deliberate else ANSWERS_FILE


def summary_file(deliberate: bool) -> str:
    return DELIBERATE_SUMMARY_FILE if deliberate else SUMMARY_FILE


def ladder_body(level: int, scene: Scene, lp: LadderParams) -> tuple[Levels, Temperament, float]:
    """The swept body at `level` in `scene`: its belief, temperament and arousal.

    The lens reads neither trust nor baseline, so both are 0 (no mood tint)."""
    point = lp.scenes[scene]
    belief = Levels(heat=lp.heat_levels[level], water=point.water, energy=point.energy)
    temperament = Temperament(
        setpoint=lp.setpoint,
        tightness=lp.tightness,
        trust=dict.fromkeys(Channel, 0.0),
        habit=lp.habit,
        baseline=0,
    )
    return belief, temperament, lp.arousal


def ladder_prompt(
    level: int, scene: Scene, order: list[str], p: LensParams, lp: LadderParams
) -> str:
    """The lens text of the swept body at the frozen noon, a blank line, the question, and the
    four actions lettered A to D in `order`."""
    belief, t, arousal = ladder_body(level, scene, lp)
    text = felt_scene(
        Place.MEADOW, frozen_noon_facts(), belief, t, arousal, None, None, 0, p
    ).text()
    choices = [f"{letter}. {LADDER_ACTIONS[a]}" for letter, a in zip(LETTERS, order, strict=True)]
    return "\n".join([text, "", LADDER_QUESTION, *choices])


def deliberate_ladder_prompt(
    level: int, scene: Scene, order: list[str], p: LensParams, lp: LadderParams
) -> str:
    """The Deliberate prompt of the swept body at the frozen noon: its lens text, the neutral
    reading, the meadow's four offers lettered in `order` (the plain check's keys), no recalled
    moment, no pull and no override."""
    belief, t, arousal = ladder_body(level, scene, lp)
    text = felt_scene(
        Place.MEADOW, frozen_noon_facts(), belief, t, arousal, None, None, 0, p
    ).text()
    offers = [
        Offer(action=a, phrase=OFFER_PHRASES[(Place.MEADOW, a)])
        for a in (LADDER_ACTION_OF[key] for key in order)
    ]
    return deliberate_prompt(text, NEUTRAL_READING, offers, [], "", "")


def _prompt(
    level: int, scene: Scene, order: list[str], p: LensParams, lp: LadderParams, deliberate: bool
) -> str:
    build = deliberate_ladder_prompt if deliberate else ladder_prompt
    return build(level, scene, order, p, lp)


def stored_prompt(user_prompt: str, deliberate: bool) -> str:
    """Everything a model read as text for one answer: the instructions of the check's agent, a
    blank line, and the user prompt. This is what an answer stores and is held against."""
    instruction = DELIBERATE_INSTRUCTION if deliberate else LADDER_INSTRUCTION
    return f"{instruction}\n\n{user_prompt}"


def answer_order(i: int) -> list[str]:
    """The order of the four actions for answer `i`, drawn from `i` alone: every cell sees the
    same orders, so letter-position effects cancel between levels and scenes."""
    order = list(LADDER_ACTIONS)
    derive_rng(i, "ladder-order").shuffle(order)
    return order


class LadderAnswer(Frozen):
    scene: Scene
    level: int = Field(ge=0)
    i: int = Field(ge=0)
    order: list[str]
    letter: Letter
    action: str
    why: str
    model: str  # the model name the answer came from
    prompt: str  # the exact text it read: instructions and user prompt (`stored_prompt`)

    @model_validator(mode="after")
    def _consistent(self) -> "LadderAnswer":
        if sorted(self.order) != sorted(LADDER_ACTIONS):
            raise ValueError(f"order must be the four actions {sorted(LADDER_ACTIONS)}")
        if self.order[LETTERS.index(self.letter)] != self.action:
            raise ValueError(f"letter {self.letter} is not {self.action!r} in this order")
        return self


class CurveSummary(Frozen):
    share: list[float]  # share of answers choosing the shade, per level, coolest first
    slope: float  # least-squares slope of the 0/1 shade choice on the level index
    range: float  # share at the hottest level minus share at the coolest
    worst_drop: float  # the largest fall of the share from one level to the next warmer one
    passed: bool


# The structured answer. Field names and descriptions are what the model reads; no docstring, as
# pydantic-ai would send it as the output tool's description.
class LadderChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    letter: Letter = Field(description=LADDER_LETTER_HELP)
    why: OneLine = Field(min_length=1, description=LADDER_WHY_HELP)


ladder_agent: Agent[None, LadderChoice] = Agent(
    output_type=LadderChoice, instructions=LADDER_INSTRUCTION, name="ladder"
)


def _slope(points: list[tuple[int, float]]) -> float:
    n = len(points)
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    sxx = sum((x - mx) ** 2 for x, _ in points)
    return sum((x - mx) * (y - my) for x, y in points) / sxx


def summarize(answers: list[LadderAnswer], lp: LadderParams) -> dict[str, CurveSummary]:
    """The shade curve of each scene present in `answers`, and whether it passes."""
    out: dict[str, CurveSummary] = {}
    levels = range(len(lp.heat_levels))
    for scene in SCENES:
        mine = [a for a in answers if a.scene == scene]
        if not mine:
            continue
        points = [(a.level, float(a.action == SHADE)) for a in mine]
        share: list[float] = []
        for level in levels:
            ys = [y for x, y in points if x == level]
            if not ys:
                raise ValueError(f"scene {scene} level {level}: no answers")
            share.append(sum(ys) / len(ys))
        slope = _slope(points)
        spread = share[-1] - share[0]
        worst_drop = max(a - b for a, b in zip(share, share[1:], strict=False))
        out[scene] = CurveSummary(
            share=share,
            slope=slope,
            range=spread,
            worst_drop=worst_drop,
            passed=slope >= lp.min_slope and spread >= lp.min_range and worst_drop < lp.max_drop,
        )
    return out


def _is_json(line: str) -> bool:
    try:
        json.loads(line)
    except ValueError:
        return False
    return True


def _complete_lines(text: str) -> list[str]:
    """The lines of an answers file without a last line that has no newline and is not complete
    JSON: the part an append cut short when the run was killed. A last line that ends in a newline
    was written whole, so it is kept (and refused like any bad line if it is not an answer)."""
    lines = text.splitlines()
    if lines and not text.endswith("\n") and not _is_json(lines[-1]):
        lines.pop()
    return lines


def read_answers(path: Path) -> list[LadderAnswer]:
    """The answers stored in `path`, in file order; none when there is no file. A last line that
    is not complete JSON (a run killed mid-write) is dropped; a bad line anywhere else is an
    error."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    answers = []
    for n, line in enumerate(_complete_lines(text), start=1):
        if not line.strip():
            continue
        try:
            answers.append(LadderAnswer.model_validate_json(line))
        except ValidationError as e:
            problem = e.errors()[0]["msg"]
            raise ValueError(f"{path}: line {n}: not a ladder answer: {problem}") from None
    return answers


def current_answers(
    answers: list[LadderAnswer],
    name: str,
    samples: int,
    params: Params,
    deliberate: bool = False,
) -> tuple[list[LadderAnswer], int]:
    """The stored answers that count for a sweep of `samples` per cell on model `name`, one per
    slot (scene, level, index), and how many are stale.

    An answer counts when its model is `name` and its prompt is the one the current code builds
    for its slot; when several count for one slot, the last wins. Any other answer for a slot of
    the sweep, or for a level the params no longer have, is stale. Answers at an index of
    `samples` or above are outside this sweep: neither counted nor stale. `deliberate` says
    which check's prompts the answers are held against."""
    lp = params.ladder
    prompts: dict[tuple[str, int, int], str] = {}
    slots: dict[tuple[str, int, int], LadderAnswer] = {}
    stale = 0
    for a in answers:
        if a.i >= samples:
            continue
        key = (a.scene, a.level, a.i)
        if a.level >= len(lp.heat_levels):
            stale += 1
            continue
        if key not in prompts:
            order = answer_order(a.i)
            user = _prompt(a.level, a.scene, order, params.lens, lp, deliberate)
            prompts[key] = stored_prompt(user, deliberate)
        if a.model == name and a.prompt == prompts[key] and a.order == answer_order(a.i):
            slots[key] = a
        else:
            stale += 1
    return list(slots.values()), stale


def short_cells(answers: list[LadderAnswer], samples: int, lp: LadderParams) -> list[str]:
    """The cells with fewer than `samples` of `answers` (one per slot, as `current_answers`
    gives them), as "scene A level 3: 28 of 30"."""
    counts = {(s, level): 0 for s in SCENES for level in range(len(lp.heat_levels))}
    for a in answers:
        counts[(a.scene, a.level)] += 1
    return [
        f"scene {s} level {level}: {n} of {samples}"
        for (s, level), n in counts.items()
        if n < samples
    ]


async def run_ladder(
    model: Model,
    samples: int,
    jobs: int,
    out_dir: Path,
    params: Params,
    done: list[LadderAnswer],
    name: str,
    deliberate: bool = False,
) -> list[LadderAnswer]:
    """Ask `model` (named `name`) for every answer of the sweep that `done` holds no current
    answer for (see `current_answers`), at most `jobs` at a time, cells interleaved in a seeded
    shuffle. Each answer is appended to `out_dir/answers.jsonl` (with `deliberate`, through the
    Deliberate agent, to `deliberate_answers.jsonl`) as it arrives; a failed call is reported on
    stderr in one line and not written. Returns the new answers."""
    lp = params.ladder
    current = current_answers(done, name, samples, params, deliberate)[0]
    have = {(a.scene, a.level, a.i) for a in current}
    todo = [
        (scene, level, i)
        for scene in SCENES
        for level in range(len(lp.heat_levels))
        for i in range(samples)
        if (scene, level, i) not in have
    ]
    if not todo:
        return []
    derive_rng(0, "ladder-schedule").shuffle(todo)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / answers_file(deliberate)
    _cut_partial_tail(path)
    gate = asyncio.Semaphore(jobs)
    new: list[LadderAnswer] = []

    async def letter_and_why(prompt: str) -> tuple[str, str]:
        if deliberate:
            choice = await ask(deliberate_agent, prompt, model, params.llm, len(LETTERS))
            return choice.letter, choice.why
        plain = await ask(ladder_agent, prompt, model, params.llm, None)
        return plain.letter, plain.why

    async def one(scene: Scene, level: int, i: int) -> None:
        order = answer_order(i)
        prompt = _prompt(level, scene, order, params.lens, lp, deliberate)
        async with gate:
            try:
                letter, why = await letter_and_why(prompt)
            except Exception as e:
                reason = _one_line(f"{type(e).__name__}: {e}")
                print(
                    f"ladder: scene {scene} level {level} answer {i} failed: {reason}",
                    file=sys.stderr,
                )
                return
        answer = LadderAnswer(
            scene=scene,
            level=level,
            i=i,
            order=order,
            letter=letter,
            action=order[LETTERS.index(letter)],
            why=why,
            model=name,
            prompt=stored_prompt(prompt, deliberate),
        )
        with path.open("a", encoding="utf-8") as f:
            f.write(answer.model_dump_json() + "\n")
        new.append(answer)

    async with model:  # one HTTP client for the whole sweep
        await asyncio.gather(*(one(*key) for key in todo))
    return new


def _cut_partial_tail(path: Path) -> None:
    """Make the next append start a line of its own after a run killed mid-write: a last line
    without its newline is cut off at the byte after the last newline when it is not complete
    JSON (`read_answers` already ignores it), or given its newline when it is. Nothing before it
    is rewritten, so a second kill here cannot lose a stored answer."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return
    if not data or data.endswith(b"\n"):
        return
    tail = data[data.rfind(b"\n") + 1 :]
    if _is_json(tail.decode("utf-8", errors="replace")):
        with path.open("ab") as f:
            f.write(b"\n")
    else:
        os.truncate(path, len(data) - len(tail))


def ladder_table(
    scene: Scene, answers: list[LadderAnswer], curve: CurveSummary, lp: LadderParams
) -> str:
    """One scene as a plain-text table: the count of each action and the shade share per level,
    then the three measures and the verdict."""
    actions = list(LADDER_ACTIONS)
    head = f"{'level':>5}  {'heat':>5}  " + "  ".join(f"{a:>6}" for a in actions) + "   share"
    lines = [f"scene {scene}", head]
    for level, heat in enumerate(lp.heat_levels):
        mine = [a.action for a in answers if a.scene == scene and a.level == level]
        counts = "  ".join(f"{mine.count(a):>6}" for a in actions)
        lines.append(f"{level:>5}  {heat:>+5.2f}  {counts}   {curve.share[level]:.2f}")
    verdict = "passed" if curve.passed else "FAILED"
    lines.append(
        f"slope {curve.slope:+.3f} (at least {lp.min_slope:g})  "
        f"range {curve.range:+.2f} (at least {lp.min_range:g})  "
        f"worst drop {curve.worst_drop:+.2f} (below {lp.max_drop:g})  {verdict}"
    )
    return "\n".join(lines)
