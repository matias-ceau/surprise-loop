"""The two bodies of the contracts, shared by the tests: temperaments, beliefs, fixed texts.

The values are those of the plan's "The two bodies"; the experiment files must agree with them
(tests/test_config.py checks that).
"""

from surprise_loop.state import SurpriseNote, Temperament
from surprise_loop.world import Channel, FactKind, Levels

ILO = Temperament(
    setpoint=Levels(heat=-0.1, water=0.80, energy=0.70),
    tightness={Channel.HEAT: 3.0, Channel.WATER: 6.0, Channel.ENERGY: 6.0},
    trust={Channel.HEAT: 0.5, Channel.WATER: 0.3, Channel.ENERGY: 0.5},
    habit={FactKind.BURDEN: 0.6, FactKind.RELIEF: 0.2, FactKind.AMBIENT: 0.1},
    baseline=-0.1,
)
TAVI = Temperament(
    setpoint=Levels(heat=0.2, water=0.80, energy=0.70),
    tightness={Channel.HEAT: 2.5, Channel.WATER: 4.0, Channel.ENERGY: 4.0},
    trust={Channel.HEAT: 0.5, Channel.WATER: 0.5, Channel.ENERGY: 0.5},
    habit={FactKind.BURDEN: 0.1, FactKind.RELIEF: 0.1, FactKind.AMBIENT: 0.6},
    baseline=0.1,
)

# the afternoon start (the worked example): beliefs, arousal and the voiced surprise
ILO_AFTERNOON = Levels(heat=0.74, water=0.58, energy=0.25)
TAVI_AFTERNOON = Levels(heat=0.0, water=0.85, energy=0.75)
ILO_AFTERNOON_AROUSAL = 0.70
TAVI_AFTERNOON_AROUSAL = 0.20
HOTTER = SurpriseNote(channel=Channel.HEAT, worse=True)

# the fixed texts for the 13:00 meadow facts and the afternoon start
ILO_TEXT = (
    "You are standing in the middle of the forest clearing. There is no place to sit, and you "
    "need one. It is really hot. The sun beats down on your head. The one dark patch is under "
    "the oak, a few steps off. Your legs are heavy. Your head is thick. It is hotter than you "
    "thought it would be."
)
TAVI_TEXT = (
    "You are standing in the middle of the forest clearing. The air is pleasantly warm. The sun "
    "sits on your shoulders. It looks chilly under the oak. Birds are calling. A stream runs "
    "somewhere to the west. A light breeze moves the leaves."
)
