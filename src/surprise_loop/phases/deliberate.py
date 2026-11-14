"""Deliberate: choose an action and predict its outcome."""

from collections import Counter

from surprise_loop.config import Runtime, derive_rng
from surprise_loop.phases.heuristics import (
    PRESSES,
    clamp,
    episodes_of,
    expected_reward,
    goal_utilities,
    require,
    softmax,
)
from surprise_loop.state import Decision, Prediction, Workspace
from surprise_loop.world import Action, Event

STUB_RATIONALE = "stub"

_DEFAULT_EVENT = {
    Action.PRESS_A: Event.NO_REWARD,
    Action.PRESS_B: Event.NO_REWARD,
    Action.EXPLORE: Event.EXPLORED,
    Action.REST: Event.RESTED,
}


async def deliberate_stub(ws: Workspace, rt: Runtime) -> Workspace:
    percept = require(ws.percept, "percept", "deliberate")
    appraisal = require(ws.appraisal, "appraisal", "deliberate")
    params = rt.config.params.deliberate
    agent = ws.agent
    recalled = percept.recalled

    utilities = goal_utilities(agent, recalled, params)
    mood_pull = params.approach_weight * appraisal.valence + params.vigor_weight * appraisal.arousal
    for action in PRESSES:
        utilities[action] += mood_pull
    if ws.conflict is None:
        for action in Action:
            utilities[action] += agent.dispositions.biases[action]

    actions = list(utilities)
    probs = softmax([utilities[a] for a in actions], params.temperature)
    draw = derive_rng(rt.config.seed, agent.tick + 1, ws.pass_index).random()
    chosen = actions[-1]
    cumulative = 0.0
    for action, p in zip(actions, probs, strict=True):
        cumulative += p
        if draw < cumulative:
            chosen = action
            break

    estimate = expected_reward(chosen, recalled, agent.traits.surprise_learning_rate, params)
    # Ties go to the most recent event: Counter keeps first-seen order, newest is seen first.
    events = Counter(e.event for e in episodes_of(chosen, recalled))
    expected_event = events.most_common(1)[0][0] if events else _DEFAULT_EVENT[chosen]
    prediction = Prediction(
        expected_event=expected_event,
        expected_reward=estimate.value if chosen in PRESSES else 0,
        confidence=clamp(estimate.n / (estimate.n + params.confidence_halfsat), 0, 1),
    )
    decision = Decision(action=chosen, prediction=prediction, rationale=STUB_RATIONALE)
    return ws.model_copy(update={"decision": decision})


async def deliberate(ws: Workspace, rt: Runtime) -> Workspace:
    backend = rt.config.backends.deliberate
    if backend == "stub":
        return await deliberate_stub(ws, rt)
    raise NotImplementedError(f'deliberate backend "{backend}" is not implemented yet')
