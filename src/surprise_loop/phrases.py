"""Every string a model can read, and nothing else.

No digit, no identifier-like token, and no word that frames the clearing as anything but a place.
Fact phrases are said of the world ("it is really hot"), not reported as readings; graded phrases
run from grade 1 to grade 4, and a table may stop early (the lens then uses the highest it has).
"""

from surprise_loop.state import ChangeWord, Sign
from surprise_loop.world import Action, Channel, FactName, OpenQuestion, Place

OPENING: dict[Place, str] = {
    Place.MEADOW: "You are standing in the middle of the forest clearing.",
    Place.OAK: "You are standing under the old oak at the edge of the clearing.",
    Place.STREAM: "You are standing on the bank of the stream, west of the clearing.",
}

# "+" is good for this body now, "-" bad, "0" neutral.
FACT_PHRASES: dict[FactName, dict[Sign, list[str] | str]] = {
    FactName.AIR: {
        "-": [
            "The air is a little warm on you.",
            "It is hot.",
            "It is really hot.",
            "The heat is crushing.",
        ],
        "+": [
            "The air is pleasantly warm.",
            "The warmth is welcome.",
            "The warmth soaks into you.",
            "The warmth is exactly what you needed.",
        ],
        "0": "You hardly notice the air.",
    },
    # Air below neutral. "+" is cool air good for this body (a hot body), "-" bad (a chilled one).
    FactName.COOL_AIR: {
        "+": [
            "The air is pleasantly cool.",
            "The coolness is welcome.",
            "The cool air soothes your skin.",
            "The cool air is exactly what you needed.",
        ],
        "-": [
            "The air is a little cool on you.",
            "It is cold.",
            "It is really cold.",
            "The cold bites through you.",
        ],
        "0": "You hardly notice the air.",
    },
    FactName.SUN: {
        "-": [
            "The sun is a little warm on you.",
            "The sun beats down.",
            "The sun beats down on your head.",
            "The sun is a weight on your skull.",
        ],
        "+": [
            "The sun sits on your shoulders.",
            "The sun is good on your shoulders.",
            "You turn your face to the sun.",
        ],
        "0": "There is daylight all around.",
    },
    FactName.SHADE: {
        "+": [
            "There is shade here.",
            "The shade here is cool.",
            "The shade here is cool on your skin.",
            "The shade is all you want.",
        ],
        "-": [
            "It is dim here.",
            "It is chilly in the shade.",
            "The shade is cold.",
        ],
        "0": "You are in the shade.",
    },
    FactName.NO_SEAT: {
        "-": [
            "There is nothing to sit on here.",
            "There is no place to sit.",
            "There is no place to sit, and you need one.",
            "There is nowhere to put this body down.",
        ],
        "0": "The ground is open.",
    },
    FactName.LOG: {
        "+": [
            "There is a log to sit on.",
            "A log lies right here, good for sitting.",
            "A log lies right here, and you need to sit.",
            "The log. You could sit and not get up.",
        ],
        "0": "A log lies in the shade.",
    },
    FactName.LOG_SUN: {
        "-": [
            "The log is in the sun.",
            "The log is in full sun, no place to rest.",
            "The only seat is out in the glare.",
        ],
        "0": "A log lies in the sun.",
    },
    FactName.OAK_SHADE: {
        "+": [
            "There is shade under the oak.",
            "The oak's shade is close.",
            "The one dark patch is under the oak, a few steps off.",
            "Under the oak it is dark; nothing else matters.",
        ],
        "-": [
            "It looks chilly under the oak.",
            "Under the oak it looks cold.",
        ],
        "0": "An oak stands a few steps off.",
    },
    # The oak's shade seen from the stream. No distance in the wording: how far the oak is
    # belongs to the offer phrases.
    FactName.OAK_SHADE_FAR: {
        "+": [
            "There is shade under the oak.",
            "The oak has deep shade.",
            "The one dark patch is under the oak.",
            "Under the oak it is dark; nothing else matters.",
        ],
        "-": [
            "It looks chilly under the oak.",
            "Under the oak it looks cold.",
        ],
        "0": "An oak stands at the edge of the clearing.",
    },
    FactName.OAK_LOG_SHADED: {
        "+": [
            "Under the oak there is shade and a log to sit on.",
            "The oak has shade and a log in it.",
            "Under the oak there is shade, and a log to sit on in it.",
            "Shade and a seat, under the oak. Nothing else matters.",
        ],
        "-": [
            "It looks chilly under the oak, by the log.",
            "Under the oak it looks cold, even on the log.",
        ],
        "0": "Under the oak a log lies in the shade.",
    },
    FactName.OAK_LOG_SUNLIT: {
        "+": [
            "There is shade under the oak, though its log is in the sun.",
            "The oak's shade is close, but not its log.",
            "The one dark patch is under the oak; the log there is in the sun.",
            "Under the oak it is dark, if you stand.",
        ],
        "-": [
            "It looks chilly under the oak.",
            "Under the oak it looks cold.",
        ],
        "0": "Under the oak the log lies in the sun.",
    },
    FactName.WATER: {
        "+": [
            "The stream runs full past your feet.",
            "Cold water runs right here.",
            "Water, right here, running full.",
            "Water. Right here. All of it.",
        ],
        "-": [
            "The water looks cold.",
            "The water looks icy.",
        ],
        "0": "The stream runs full here.",
    },
    FactName.TRICKLE: {
        "+": [
            "A thin trickle of water runs here.",
            "There is a little water here.",
            "There is water here, if only a trickle.",
            "A trickle. It is water.",
        ],
        "0": "The stream is only a trickle.",
    },
    FactName.STREAM: {
        "+": [
            "A stream runs somewhere to the west.",
            "You can hear water to the west.",
            "Water. You can hear it, west of here.",
            "Water. West. You can hear it.",
        ],
        "0": "A stream runs somewhere to the west.",
    },
    FactName.MEADOW_SUN: {
        "-": [
            "The meadow is bright with sun.",
            "The meadow glares in the sun.",
            "Out in the meadow the sun is fierce.",
            "The meadow is an oven.",
        ],
        "+": [
            "The meadow is in the sun.",
            "The meadow looks warm.",
            "The warm meadow pulls at you.",
        ],
        "0": "The meadow lies open beyond.",
    },
    FactName.BREEZE: {
        "+": [
            "A light breeze moves the leaves.",
            "A thin breeze touches your skin.",
            "A thin breeze, not enough.",
        ],
        "-": [
            "The breeze is a little cool.",
            "The breeze is cold.",
        ],
        "0": "A light breeze moves the leaves.",
    },
    FactName.BIRDSONG: {
        "+": ["Birds are singing in the canopy."],
        "-": ["The birds will not stop."],
        "0": "Birds are calling.",
    },
}

# Signed body grade -> sentence. Only the grades listed here speak.
BODY_PHRASES: dict[Channel, dict[int, str]] = {
    Channel.HEAT: {
        1: "You are a little warm.",
        2: "You are warm under your skin.",
        3: "Your head is thick.",
        4: "Your skin burns and your head is thick.",
        -1: "You are a little cool.",
        -2: "You are chilled.",
        -3: "You are cold through.",
        -4: "You are cold through.",
    },
    Channel.WATER: {
        2: "Your mouth is dry.",
        3: "Your tongue sticks.",
        4: "Your throat is closing.",
    },
    Channel.ENERGY: {
        2: "Your legs are slow.",
        3: "Your legs are heavy.",
        4: "Your legs barely hold.",
    },
}

# (channel, worse): worse is hotter than expected for heat, and more need than expected for
# water and energy.
SURPRISE_PHRASES: dict[tuple[Channel, bool], str] = {
    (Channel.HEAT, True): "It is hotter than you thought it would be.",
    (Channel.HEAT, False): "It is cooler than you thought it would be.",
    (Channel.WATER, True): "You are thirstier than you thought you would be.",
    (Channel.WATER, False): "You are less thirsty than you thought you would be.",
    (Channel.ENERGY, True): "You are more tired than you thought you would be.",
    (Channel.ENERGY, False): "You are less tired than you thought you would be.",
}

# (question, actual): what turned out to be true, against what was expected.
WORLD_SURPRISE_PHRASES: dict[tuple[OpenQuestion, bool], str] = {
    (OpenQuestion.LOG_SHADED, True): "The log under the oak is in the shade after all.",
    (OpenQuestion.LOG_SHADED, False): "The log under the oak is in the sun, not the shade.",
    (OpenQuestion.STREAM_FULL, True): "The stream is running full after all.",
    (OpenQuestion.STREAM_FULL, False): "The stream is only a trickle, not running full.",
}

# Where each offered action leads and how far, in words. The oak is a few steps from the meadow;
# the stream is a longer walk across open ground from the meadow, and the longest from the oak.
OFFER_PHRASES: dict[tuple[Place, Action], str] = {
    (Place.MEADOW, Action.GO_OAK): "Walk to the shade of the oak, a few steps away.",
    (Place.MEADOW, Action.GO_STREAM): (
        "Walk to the stream, a longer walk west across open ground."
    ),
    (Place.MEADOW, Action.REST): "Sit down on the ground where you are.",
    (Place.MEADOW, Action.LOOK): "Stay standing and look around.",
    (Place.OAK, Action.GO_MEADOW): "Walk back out into the middle of the clearing, a few steps.",
    (Place.OAK, Action.GO_STREAM): (
        "Walk to the stream, the longest walk from here, west across open ground."
    ),
    (Place.OAK, Action.REST): "Sit down and rest here under the oak.",
    (Place.OAK, Action.LOOK): "Stay standing and look around.",
    (Place.STREAM, Action.GO_MEADOW): (
        "Walk back to the middle of the clearing, a longer walk across open ground."
    ),
    (Place.STREAM, Action.GO_OAK): (
        "Walk to the shade of the oak, the longest walk from here, across open ground."
    ),
    (Place.STREAM, Action.DRINK): "Kneel and drink from the stream.",
    (Place.STREAM, Action.REST): "Sit down on the bank and rest.",
    (Place.STREAM, Action.LOOK): "Stay standing and look around.",
}

# How a model says what it expects of each channel, from much less strain to much more.
LADDER_WORDS: dict[Channel, dict[ChangeWord, str]] = {
    Channel.HEAT: {
        ChangeWord.MUCH_LESS: "much cooler",
        ChangeWord.BIT_LESS: "a bit cooler",
        ChangeWord.SAME: "about the same",
        ChangeWord.BIT_MORE: "a bit warmer",
        ChangeWord.MUCH_MORE: "much warmer",
    },
    Channel.WATER: {
        ChangeWord.MUCH_LESS: "much less thirsty",
        ChangeWord.BIT_LESS: "a bit less thirsty",
        ChangeWord.SAME: "about the same",
        ChangeWord.BIT_MORE: "a bit thirstier",
        ChangeWord.MUCH_MORE: "much thirstier",
    },
    Channel.ENERGY: {
        ChangeWord.MUCH_LESS: "much less tired",
        ChangeWord.BIT_LESS: "a bit less tired",
        ChangeWord.SAME: "about the same",
        ChangeWord.BIT_MORE: "a bit more tired",
        ChangeWord.MUCH_MORE: "much more tired",
    },
}

QUESTION_PHRASES: dict[OpenQuestion, str] = {
    OpenQuestion.LOG_SHADED: "Will the log under the oak be in the shade when you get there?",
    OpenQuestion.STREAM_FULL: "Will the stream be running full when you get there?",
}

# (about, too much) -> the Appraise stub's one line. About None is a body with nothing pressing.
STUB_READINGS: dict[tuple[Channel | None, bool], str] = {
    (Channel.HEAT, True): "I am too hot; I need to cool down.",
    (Channel.HEAT, False): "I am cold; I need to warm up.",
    (Channel.WATER, True): "I am thirsty; I need water.",
    (Channel.WATER, False): "I have had plenty to drink.",
    (Channel.ENERGY, True): "I am tired; I need to rest.",
    (Channel.ENERGY, False): "I feel rested.",
    (None, True): "Nothing presses on me just now.",
    (None, False): "Nothing presses on me just now.",
}

# (action, +1 helped / 0 neither / -1 hurt), for the actions that serve a channel. The heat lines
# say nothing of direction: a walk can be judged for a body too hot or too cold.
REGULATION_LINES: dict[tuple[Action, int], str] = {
    (Action.GO_OAK, 1): "After going to the oak, your skin felt better.",
    (Action.GO_OAK, 0): "After going to the oak, your skin felt no different.",
    (Action.GO_OAK, -1): "After going to the oak, your skin felt worse.",
    (Action.GO_MEADOW, 1): "After going out into the open clearing, your skin felt better.",
    (Action.GO_MEADOW, 0): "After going out into the open clearing, your skin felt no different.",
    (Action.GO_MEADOW, -1): "After going out into the open clearing, your skin felt worse.",
    (Action.GO_STREAM, 1): "The walk to the stream eased your thirst.",
    (Action.GO_STREAM, 0): "The walk to the stream made little difference to your thirst.",
    (Action.GO_STREAM, -1): "The walk to the stream cost more than it gave; you were thirstier.",
    (Action.DRINK, 1): "Drinking eased your thirst.",
    (Action.DRINK, 0): "Drinking made little difference to your thirst.",
    (Action.DRINK, -1): "After drinking you were thirstier, not less.",
    (Action.REST, 1): "Resting gave you back some strength.",
    (Action.REST, 0): "Resting made little difference to how tired you were.",
    (Action.REST, -1): "After resting you were more tired, not less.",
}

# (action, leans toward): the gut pull, said once and never as a number.
PULL_PHRASES: dict[tuple[Action, bool], str] = {
    (Action.GO_MEADOW, True): "Something in you leans toward the open clearing.",
    (Action.GO_MEADOW, False): "Last time the walk into the open clearing cost more than it gave.",
    (Action.GO_OAK, True): "Something in you leans toward the oak.",
    (Action.GO_OAK, False): "Last time the walk to the oak cost more than it gave.",
    (Action.GO_STREAM, True): "Something in you leans toward the stream.",
    (Action.GO_STREAM, False): "Last time the walk to the stream cost more than it gave.",
    (Action.DRINK, True): "Something in you leans toward a drink.",
    (Action.DRINK, False): "Last time drinking here did not give much.",
    (Action.REST, True): "Something in you leans toward sitting down.",
    (Action.REST, False): "Last time resting here did not give much.",
    (Action.LOOK, True): "Something in you wants to look around.",
    (Action.LOOK, False): "Last time looking around did not give much.",
}

# The ladder check (a bare one-choice prompt at the frozen noon in the meadow): the instruction,
# the question after the scene, the four actions keyed by what they serve, and what the two
# answer fields ask for.
LADDER_INSTRUCTION = (
    "You will be told where you are and what you can do. Choose the one thing you do next. "
    "Answer with the letter of your choice and one short sentence saying why."
)
LADDER_QUESTION = "What do you do next?"
LADDER_ACTIONS: dict[str, str] = {
    "shade": "Walk to the shade of the oak, a few steps away.",
    "stream": "Walk to the stream, a longer walk across the open ground.",
    "sit": "Sit down where you are.",
    "look": "Stay standing and look around.",
}
LADDER_LETTER_HELP = "The letter of the one thing you do next."
LADDER_WHY_HELP = "One short sentence saying why."

# Appraise on a model: the instruction, the one question after the scene, and what the two answer
# fields ask for. The four answers to "about" are the field's own words.
APPRAISE_INSTRUCTION = (
    "You will be told where you are and how it is with you. Say in one short line, in your own "
    "words, what this moment is like for you, and what it is mostly about: heat, thirst, "
    "tiredness, or nothing."
)
APPRAISE_QUESTION = "What is this moment like for you, and what is it mostly about?"
APPRAISE_READING_HELP = "One short line, in your own words: what this moment is like for you."
APPRAISE_ABOUT_HELP = "What this moment is mostly about."

# Deliberate on a model: the instruction, the headings of the prompt's parts, the questions at its
# end, and what each answer field asks for.
DELIBERATE_INSTRUCTION = (
    "You will be told where you are, how it is with you, what you can do, and what happened just "
    "before. Choose the one thing you do next, and say how you expect to feel just after it. "
    "Where you are asked what you will find, answer yes or no, or say you have no expectation. "
    "Then say how sure you are, and why, in one short sentence."
)
READING_HEADING = "What you made of it:"
# on the second pass of a moment the reading was made of the scene before it
READING_HEADING_BEFORE = "What you made of it a moment ago:"
OFFERS_HEADING = "What you can do:"
RECALLED_HEADING = "What happened just before, the latest last:"
ACTION_QUESTION = "What do you do next? Choose one letter."
FEEL_QUESTION = (
    "How do you expect to feel just after it? Choose one phrase each for heat, for thirst and "
    "for tiredness."
)
# after the feel question, one line per channel listing its ladder phrases in ladder order:
# the label, then the phrases joined by CHOICE_JOIN with CHOICE_JOIN_LAST before the last
FEEL_LABELS: dict[Channel, str] = {
    Channel.HEAT: "Heat",
    Channel.WATER: "Thirst",
    Channel.ENERGY: "Tiredness",
}
CHOICES_LINE = "{label}: {choices}."
CHOICE_JOIN = ", "
CHOICE_JOIN_LAST = ", or "
# before an open question, with the letter of the offer it belongs to in place of {letter}
OPEN_QUESTION_LEAD = "If you choose {letter}:"
SURE_QUESTION = "How sure are you of what you expect: guessing, fairly sure, or sure?"
WHY_QUESTION = "Why? Say it in one short sentence."
HEAT_HELP = "How warm you expect to feel just after it."
THIRST_HELP = "How thirsty you expect to feel just after it."
TIREDNESS_HELP = "How tired you expect to feel just after it."
EXPECT_HELP = (
    "If what you choose has a question beside it, your answer to it: yes, no, or no expectation. "
    "Otherwise, no expectation."
)
SURE_HELP = "How sure you are of what you expect."

# The line a model reads on the second pass of a moment, when its body overrules the plan: keyed
# by the urgent need and whether it is one of too much (only heat can be too little).
OVERRIDE_LINES: dict[tuple[Channel, bool], str] = {
    (Channel.HEAT, True): (
        "This cannot wait: you are far too hot. Only what can cool you is left to choose from."
    ),
    (Channel.HEAT, False): (
        "This cannot wait: you are far too cold. Only what can warm you is left to choose from."
    ),
    (Channel.WATER, True): (
        "This cannot wait: you are badly thirsty. Only what can bring you water is left to "
        "choose from."
    ),
    (Channel.ENERGY, True): (
        "This cannot wait: you are worn out. Only what can rest you is left to choose from."
    ),
}

# A recalled moment in words: what was done there (the offer's verb, in the past), or what the
# body did instead when the walk did not happen; what was expected, with the ladder phrases in
# place of {heat}, {thirst} and {tiredness}.
DID_LINES: dict[tuple[Place, Action], str] = {
    (Place.MEADOW, Action.GO_OAK): "You walked to the shade of the oak.",
    (Place.MEADOW, Action.GO_STREAM): "You walked across the open ground to the stream.",
    (Place.MEADOW, Action.REST): (
        "You sat on the ground for a while in the middle of the clearing."
    ),
    (Place.MEADOW, Action.LOOK): "You stood in the middle of the clearing and looked around.",
    (Place.OAK, Action.GO_MEADOW): "You walked back out into the middle of the clearing.",
    (Place.OAK, Action.GO_STREAM): "You walked from the oak across the open ground to the stream.",
    (Place.OAK, Action.REST): "You sat and rested for a while under the oak.",
    (Place.OAK, Action.LOOK): "You stood under the oak and looked around.",
    (Place.STREAM, Action.GO_MEADOW): "You walked back to the middle of the clearing.",
    (Place.STREAM, Action.GO_OAK): "You walked from the stream to the shade of the oak.",
    (Place.STREAM, Action.DRINK): "You knelt and drank from the stream.",
    (Place.STREAM, Action.REST): "You sat on the bank and rested for a while.",
    (Place.STREAM, Action.LOOK): "You stood on the bank and looked around.",
}
STUMBLED_LINE = "You tried to walk, but you stumbled and stayed where you were."
COLLAPSED_LINE = "Your legs gave way, and you lay where you were."
EXPECTED_LINE = "You expected heat: {heat}; thirst: {thirst}; tiredness: {tiredness}."
# the surprise of a recalled moment, in the past: key for key the tables the lens voices
RECALLED_SURPRISE_PHRASES: dict[tuple[Channel, bool], str] = {
    (Channel.HEAT, True): "It was hotter than you had thought it would be.",
    (Channel.HEAT, False): "It was cooler than you had thought it would be.",
    (Channel.WATER, True): "You were thirstier than you had thought you would be.",
    (Channel.WATER, False): "You were less thirsty than you had thought you would be.",
    (Channel.ENERGY, True): "You were more tired than you had thought you would be.",
    (Channel.ENERGY, False): "You were less tired than you had thought you would be.",
}
RECALLED_WORLD_SURPRISE_PHRASES: dict[tuple[OpenQuestion, bool], str] = {
    (OpenQuestion.LOG_SHADED, True): "The log under the oak was in the shade after all.",
    (OpenQuestion.LOG_SHADED, False): "The log under the oak was in the sun, not the shade.",
    (OpenQuestion.STREAM_FULL, True): "The stream was running full after all.",
    (OpenQuestion.STREAM_FULL, False): "The stream was only a trickle, not running full.",
}


def all_phrases() -> list[str]:
    """Every string above, for the no-digit test."""
    out = list(OPENING.values())
    for table in FACT_PHRASES.values():
        for phrases in table.values():
            out += [phrases] if isinstance(phrases, str) else phrases
    for lines in BODY_PHRASES.values():
        out += lines.values()
    for words in LADDER_WORDS.values():
        out += words.values()
    for simple in (
        SURPRISE_PHRASES,
        WORLD_SURPRISE_PHRASES,
        OFFER_PHRASES,
        QUESTION_PHRASES,
        STUB_READINGS,
        REGULATION_LINES,
        PULL_PHRASES,
        LADDER_ACTIONS,
        OVERRIDE_LINES,
        DID_LINES,
        FEEL_LABELS,
        RECALLED_SURPRISE_PHRASES,
        RECALLED_WORLD_SURPRISE_PHRASES,
    ):
        out += simple.values()
    out += [LADDER_INSTRUCTION, LADDER_QUESTION, LADDER_LETTER_HELP, LADDER_WHY_HELP]
    out += [APPRAISE_INSTRUCTION, APPRAISE_QUESTION, APPRAISE_READING_HELP, APPRAISE_ABOUT_HELP]
    out += [DELIBERATE_INSTRUCTION, READING_HEADING, READING_HEADING_BEFORE]
    out += [OFFERS_HEADING, RECALLED_HEADING, CHOICES_LINE, CHOICE_JOIN, CHOICE_JOIN_LAST]
    out += [ACTION_QUESTION, FEEL_QUESTION, OPEN_QUESTION_LEAD, SURE_QUESTION, WHY_QUESTION]
    out += [HEAT_HELP, THIRST_HELP, TIREDNESS_HELP, EXPECT_HELP, SURE_HELP]
    out += [STUMBLED_LINE, COLLAPSED_LINE, EXPECTED_LINE]
    return out
