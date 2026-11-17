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
exact prompt it answered: a stored answer counts only while both match the current run, so a
rerun after a change of wording or of model asks those slots again instead of counting the old
answers.
"""

import asyncio
import sys
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic_ai import Agent
from pydantic_ai.models import Model

from surprise_loop.agents import model_settings
from surprise_loop.config import LadderParams, LensParams, Params, Scene, derive_rng
from surprise_loop.lens import felt_scene
from surprise_loop.phrases import (
    LADDER_ACTIONS,
    LADDER_INSTRUCTION,
    LADDER_LETTER_HELP,
    LADDER_QUESTION,
    LADDER_WHY_HELP,
)
from surprise_loop.state import Temperament
from surprise_loop.world import Channel, Frozen, Levels, Place, frozen_noon_facts

ANSWERS_FILE = "answers.jsonl"
SUMMARY_FILE = "summary.json"
SCENES: tuple[Scene, ...] = ("A", "B")
LETTERS = "ABCD"
SHADE = "shade"

Letter = Literal["A", "B", "C", "D"]


def _one_line(text: str) -> str:
    return " ".join(text.split())


OneLine = Annotated[str, AfterValidator(_one_line)]


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
    prompt: str  # the exact user prompt it answered

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


def read_answers(path: Path) -> list[LadderAnswer]:
    """The answers stored in `path`, in file order; none when there is no file."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    answers = []
    for n, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            answers.append(LadderAnswer.model_validate_json(line))
        except ValidationError as e:
            problem = e.errors()[0]["msg"]
            raise ValueError(f"{path}: line {n}: not a ladder answer: {problem}") from None
    return answers


def current_answers(
    answers: list[LadderAnswer], name: str, samples: int, params: Params
) -> tuple[list[LadderAnswer], int]:
    """The stored answers that count for a sweep of `samples` per cell on model `name`, one per
    slot (scene, level, index), and how many are stale.

    An answer counts when its model is `name` and its prompt is the one the current code builds
    for its slot; when several count for one slot, the last wins. Any other answer for a slot of
    the sweep, or for a level the params no longer have, is stale. Answers at an index of
    `samples` or above are outside this sweep: neither counted nor stale."""
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
            prompts[key] = ladder_prompt(a.level, a.scene, answer_order(a.i), params.lens, lp)
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
) -> list[LadderAnswer]:
    """Ask `model` (named `name`) for every answer of the sweep that `done` holds no current
    answer for (see `current_answers`), at most `jobs` at a time, cells interleaved in a seeded
    shuffle. Each answer is appended to `out_dir/answers.jsonl` as it arrives; a failed call is
    reported on stderr in one line and not written. Returns the new answers."""
    lp = params.ladder
    have = {(a.scene, a.level, a.i) for a in current_answers(done, name, samples, params)[0]}
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
    path = out_dir / ANSWERS_FILE
    settings = model_settings(params.llm)
    gate = asyncio.Semaphore(jobs)
    new: list[LadderAnswer] = []

    async def one(scene: Scene, level: int, i: int) -> None:
        order = answer_order(i)
        prompt = ladder_prompt(level, scene, order, params.lens, lp)
        async with gate:
            try:
                result = await ladder_agent.run(
                    prompt,
                    model=model,
                    model_settings=settings,
                    retries={"output": params.llm.output_retries},
                )
            except Exception as e:
                reason = _one_line(f"{type(e).__name__}: {e}")
                print(
                    f"ladder: scene {scene} level {level} answer {i} failed: {reason}",
                    file=sys.stderr,
                )
                return
        choice = result.output
        answer = LadderAnswer(
            scene=scene,
            level=level,
            i=i,
            order=order,
            letter=choice.letter,
            action=order[LETTERS.index(choice.letter)],
            why=choice.why,
            model=name,
            prompt=prompt,
        )
        with path.open("a", encoding="utf-8") as f:
            f.write(answer.model_dump_json() + "\n")
        new.append(answer)

    async with model:  # one HTTP client for the whole sweep
        await asyncio.gather(*(one(*key) for key in todo))
    return new


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
