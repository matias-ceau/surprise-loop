"""The lens: the world's facts plus one body's belief, turned into a felt scene in words.

Plain code, no model. Each fact gets a weight (how much it bears on what this body needs now,
plus a standing habit per kind of fact) and a value (whether it pushes the body toward where it
wants to be, or away). Only the heaviest few are noticed, fewer when the body is stirred up. Each
noticed value is cut into grades and looked up in the phrase table; loud body channels and the
last surprise get their own sentences.
"""

import math

from surprise_loop.config import LensParams
from surprise_loop.feeling import felt, grade, needs
from surprise_loop.phrases import (
    BODY_PHRASES,
    FACT_PHRASES,
    OPENING,
    SURPRISE_PHRASES,
    WORLD_SURPRISE_PHRASES,
)
from surprise_loop.state import BodyLine, FeltFact, FeltScene, Sign, SurpriseNote, Temperament
from surprise_loop.world import Channel, Fact, FactName, Levels, Place


def _softmax(xs: list[float]) -> list[float]:
    if not xs:
        return []
    top = max(xs)
    exps = [math.exp(x - top) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]


def _sign(value: float, edge: float) -> Sign:
    if value >= edge:
        return "+"
    if value <= -edge:
        return "-"
    return "0"


def _phrase(name: FactName, sign: Sign, g: int) -> str:
    """The table's phrase for this sign and grade; the highest it has when the grade runs past
    the table; the neutral phrase when the table has nothing for this sign."""
    table = FACT_PHRASES[name]
    graded = table.get(sign) if sign != "0" else None
    if not graded or isinstance(graded, str):
        neutral = table["0"]
        assert isinstance(neutral, str)
        return neutral
    return graded[min(g, len(graded)) - 1]


def _surprise(voiced: SurpriseNote | None) -> str | None:
    if voiced is None:
        return None
    if voiced.channel is not None and voiced.worse is not None:
        return SURPRISE_PHRASES[(voiced.channel, voiced.worse)]
    if voiced.question is not None and voiced.actual is not None:
        return WORLD_SURPRISE_PHRASES[(voiced.question, voiced.actual)]
    return None


def _body_lines(strength: dict[Channel, float], arousal: float, p: LensParams) -> list[BodyLine]:
    loudest = sorted(Channel, key=lambda c: -abs(strength[c]))  # stable: ties in Channel order
    room = p.body_lines_narrow if arousal >= p.narrow_arousal else p.body_lines_wide
    lines: list[BodyLine] = []
    for c in loudest[:room]:
        g = grade(strength[c], p.body_edges)
        signed = g if strength[c] >= 0 else -g
        phrase = BODY_PHRASES[c].get(signed)
        if abs(signed) >= p.body_speak_grade and phrase is not None:
            lines.append(BodyLine(channel=c, grade=signed, felt=strength[c], phrase=phrase))
    return lines


def felt_scene(
    place: Place,
    facts: list[Fact],
    belief: Levels,
    t: Temperament,
    arousal: float,
    attention: Channel | None,
    voiced: SurpriseNote | None,
    extra: int,
    p: LensParams,
) -> FeltScene:
    """What this body, believing `belief`, makes of these facts at this place."""
    need = needs(belief, t)
    relevance: list[float] = []
    values: list[float] = []
    for fact in facts:
        bonus = p.attention_bonus if attention is not None and fact.bearing.get(attention) else 0
        bears = sum(abs(b) * abs(need[c]) for c, b in fact.bearing.items())
        relevance.append(bears + t.habit[fact.kind] + bonus)
        values.append(-sum(b * need[c] for c, b in fact.bearing.items()))
    gain = p.gain_base + p.gain_arousal * arousal
    weights = _softmax([gain * r for r in relevance])

    order = sorted(range(len(facts)), key=lambda i: -weights[i])  # stable: ties keep list order
    kept = max(p.min_noticed, round(len(facts) - p.drop_per_arousal * arousal)) + extra
    n_noticed = min(len(facts), kept)

    felt_facts: list[FeltFact] = []
    for rank, i in enumerate(order):
        fact, value = facts[i], values[i]
        sign = _sign(value, p.value_edges[0])
        g = grade(value, p.value_edges)
        noticed = rank < n_noticed
        felt_facts.append(
            FeltFact(
                name=fact.name,
                about=fact.about,
                weight=weights[i],
                value=value,
                sign=sign,
                grade=g,
                noticed=noticed,
                phrase=_phrase(fact.name, sign, g) if noticed else None,
            )
        )

    strength = felt(belief, t)
    return FeltScene(
        place=place,
        opening=OPENING[place],
        facts=felt_facts,
        body=_body_lines(strength, arousal, p),
        surprise=_surprise(voiced),
        felt=strength,
    )
