"""The fixed words and texts a model reads."""

import re

from surprise_loop import phrases
from surprise_loop.world import Belief, Channel

TEXTS = (phrases.CREATURE_TEXT, phrases.DELIBERATE_TEXT, phrases.APPRAISE_TEXT)


def test_word_tuples_have_their_lengths() -> None:
    assert len(phrases.FEEL_WORDS) == 5
    assert len(phrases.NEAR_WORDS) == 3
    assert len(phrases.TONE_WORDS) == 5
    assert len(phrases.STIR_WORDS) == 3
    assert len(phrases.SURPRISE_WORDS) == 3


def test_no_text_names_a_game_simulation_or_experiment() -> None:
    for text in TEXTS:
        assert not re.search(r"\b(game|simulation|experiment)", text, re.IGNORECASE)


def test_creature_text_names_every_belief_and_feeling() -> None:
    for name in [*Belief, *Channel]:
        assert re.search(rf"\b{name}\b", phrases.CREATURE_TEXT)
    assert phrases.CREATURE_TEXT.count("\n\n") == 5
