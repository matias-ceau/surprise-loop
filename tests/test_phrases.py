import re

import pytest

from surprise_loop.phrases import (
    DID_LINES,
    FACT_PHRASES,
    LADDER_WORDS,
    OFFER_PHRASES,
    RECALLED_SURPRISE_PHRASES,
    RECALLED_WORLD_SURPRISE_PHRASES,
    REGULATION_LINES,
    SURPRISE_PHRASES,
    WORLD_SURPRISE_PHRASES,
    all_phrases,
)
from surprise_loop.state import ChangeWord
from surprise_loop.world import GO_TARGET, SERVES, Action, Channel, FactName, OpenQuestion, Place

FORBIDDEN = re.compile(r"\b(game|simulation|experiment)", re.IGNORECASE)


def offered_at(place: Place) -> set[Action]:
    """The actions the world can offer at a place: the walks to the other places, drinking only at
    the stream, resting and looking everywhere."""
    walks = {a for a, target in GO_TARGET.items() if target is not place}
    drink = {Action.DRINK} if place is Place.STREAM else set()
    return walks | drink | {Action.REST, Action.LOOK}


def test_all_phrases_is_not_empty() -> None:
    assert len(all_phrases()) > 100


@pytest.mark.parametrize("phrase", all_phrases())
def test_a_phrase_is_plain_words(phrase: str) -> None:
    assert phrase.strip()
    assert not re.search(r"\d", phrase)
    assert "_" not in phrase
    assert not FORBIDDEN.search(phrase)


def test_every_fact_has_a_neutral_phrase() -> None:
    assert set(FACT_PHRASES) == set(FactName)
    for name, table in FACT_PHRASES.items():
        assert isinstance(table["0"], str), name
        for sign in ("+", "-"):
            if sign in table:
                assert 1 <= len(table[sign]) <= 4, (name, sign)


def test_every_offer_the_world_can_make_has_a_phrase() -> None:
    expected = {(place, action) for place in Place for action in offered_at(place)}
    assert set(OFFER_PHRASES) == expected


def test_ladder_words_are_five_distinct_phrases_per_channel() -> None:
    assert set(LADDER_WORDS) == set(Channel)
    for words in LADDER_WORDS.values():
        assert set(words) == set(ChangeWord)
        assert len(set(words.values())) == 5
        assert words[ChangeWord.SAME] == "about the same"


@pytest.mark.parametrize("name", [FactName.AIR, FactName.COOL_AIR, FactName.SUN])
def test_air_and_sun_neutral_phrases_differ_from_every_graded_phrase(name: FactName) -> None:
    table = FACT_PHRASES[name]
    graded = [*table["+"], *table["-"]]
    assert table["0"] not in graded


def test_the_cool_air_neutral_phrase_carries_no_temperature() -> None:
    neutral = FACT_PHRASES[FactName.COOL_AIR]["0"]
    assert isinstance(neutral, str)
    for word in ("cool", "cold", "chill", "warm", "hot"):
        assert word not in neutral.lower(), (word, neutral)


def test_surprise_tables_are_complete() -> None:
    assert set(SURPRISE_PHRASES) == {(c, w) for c in Channel for w in (True, False)}
    assert set(WORLD_SURPRISE_PHRASES) == {(q, a) for q in OpenQuestion for a in (True, False)}


def test_regulation_lines_cover_every_action_that_serves_a_channel() -> None:
    serving = [a for a, c in SERVES.items() if c is not None]
    assert set(REGULATION_LINES) == {(a, r) for a in serving for r in (1, 0, -1)}


def test_the_recalled_surprise_tables_mirror_the_present_ones() -> None:
    assert list(RECALLED_SURPRISE_PHRASES) == list(SURPRISE_PHRASES)
    assert list(RECALLED_WORLD_SURPRISE_PHRASES) == list(WORLD_SURPRISE_PHRASES)
    for table in (RECALLED_SURPRISE_PHRASES, RECALLED_WORLD_SURPRISE_PHRASES):
        for sentence in table.values():
            assert re.search(r"\b(was|were)\b", sentence), sentence


def test_a_recalled_rest_is_over() -> None:
    for (_, action), line in DID_LINES.items():
        if action is Action.REST:
            assert "for a while" in line and "sat down" not in line, line
