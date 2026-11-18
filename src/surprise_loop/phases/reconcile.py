"""Reconcile: apply one pass's outcome to mental state. The only phase that changes the agent.

In order: the belief moves from the expectation toward the signal; the feeling follows the new
belief and the comparison; attention goes where the reading was; the loudest surprise is voiced
for the next scene; the episode is stored; the judged episode gets its score and its pull moves;
and on pass 0 an urgent need that came out worse than expected, and that the action did not
serve, sends control back to Deliberate once (the override). Reconcile never changes
`agent.tick`; `tick()` does.
"""

from surprise_loop.config import Params, Runtime
from surprise_loop.feeling import feel, needs
from surprise_loop.phases.evaluate import judged_index
from surprise_loop.phases.heuristics import (
    require,
    serving_actions,
    too_much,
    updated_belief,
    voiced_note,
)
from surprise_loop.state import (
    Comparison,
    Episode,
    Feedback,
    Memory,
    Override,
    PassRecord,
    Pulls,
    Temperament,
    Workspace,
)
from surprise_loop.world import Action, Channel, Levels, Outcome, Place


def judged(
    episodes: list[Episode], pulls: Pulls, feedback: Feedback, params: Params
) -> tuple[list[Episode], Pulls]:
    """The episodes with the judged one scored, and the pulls with its pull moved."""
    if feedback.judged_step is None or feedback.score is None:
        return episodes, pulls
    i = judged_index(len(episodes), params.evaluate.window)
    if i is None or episodes[i].step != feedback.judged_step:
        raise ValueError(
            f"feedback judged step {feedback.judged_step}, which is not the episode "
            f"{params.evaluate.window} back"
        )
    episode = episodes[i]
    scored = episode.model_copy(update={"score": feedback.score, "regulation": feedback.line})
    v = pulls.values[episode.place][episode.action]
    moved = v + params.pulls.rate * (feedback.score - v)
    values = {place: dict(per_action) for place, per_action in pulls.values.items()}
    values[episode.place][episode.action] = moved
    return [*episodes[:i], scored, *episodes[i + 1 :]], Pulls(values=values)


def override_for(
    action: Action,
    place: Place,
    belief: Levels,
    t: Temperament,
    comparison: Comparison,
    outcome: Outcome,
    params: Params,
) -> Override | None:
    """The first channel, in `Channel` order, whose need is urgent after the update, came out
    worse than expected by `worse_margin`, and is not served by the action taken at `place` (the
    place before the step); an override only if an action serving it is offered now.

    Worse is signed by the need's direction: for a need of too much (too hot, thirsty, tired)
    `body_felt` above zero is worse; for too little (too cold) below zero is."""
    p = params.override
    if not p.enabled:
        return None
    need = needs(belief, t)

    def worse(c: Channel) -> float:
        return comparison.body_felt[c] * (1 if too_much(c, need[c]) else -1)

    channel = next(
        (
            c
            for c in Channel
            if abs(need[c]) >= p.urgent_felt
            and worse(c) >= p.worse_margin
            and action not in serving_actions(c, too_much(c, need[c]), place)
        ),
        None,
    )
    if channel is None:
        return None
    observation = outcome.observation
    serving = serving_actions(channel, too_much(channel, need[channel]), observation.place)
    actions = [a for a in observation.offers if a in serving]
    if not actions:
        return None
    return Override(
        channel=channel,
        actions=actions,
        detail=(
            f"{channel.value} need {need[channel]:.2f} is at least {p.urgent_felt:.2f} and came "
            f"out {worse(channel):.2f} worse than expected; "
            f"{action.value} does not serve it"
        ),
    )


async def reconcile(ws: Workspace, rt: Runtime) -> Workspace:
    percept = require(ws.percept, "percept", "reconcile")
    appraisal = require(ws.appraisal, "appraisal", "reconcile")
    decision = require(ws.decision, "decision", "reconcile")
    outcome = require(ws.outcome, "outcome", "reconcile")
    comparison = require(ws.comparison, "comparison", "reconcile")
    feedback = require(ws.feedback, "feedback", "reconcile")
    params = rt.config.params
    agent = ws.agent
    t = agent.temperament
    action = decision.action

    belief = updated_belief(
        outcome.observation.signal, decision.expectation.bands, t, params.belief
    )
    feeling = feel(belief, t, agent.feeling, comparison, params.feeling)
    voiced = voiced_note(
        action, comparison, decision.expectation.world, params.reconcile.voice_threshold
    )

    salience = min(1, params.memory.base_salience + params.memory.salience_gain * comparison.size)
    episode = Episode(
        tick=agent.tick + 1,
        step=outcome.observation.step,
        place=ws.observation.place,
        action=action,
        event=outcome.observation.event,
        expectation=decision.expectation,
        comparison=comparison,
        need_before=needs(agent.belief, t),
        salience=salience,
        scene=percept.scene.text(),
        reading=appraisal.reading,
    )
    stored, pulls = judged(agent.memory.episodes, agent.pulls, feedback, params)

    override = None
    if ws.pass_index == 0:
        override = override_for(
            action, ws.observation.place, belief, t, comparison, outcome, params
        )

    record = PassRecord(
        tick=agent.tick + 1,
        pass_index=ws.pass_index,
        scene=percept.scene,
        offers=[o.action for o in percept.offers],
        decision=decision,
        outcome=outcome,
        comparison=comparison,
        feedback=feedback,
        belief_after=belief,
        feeling_after=feeling,
        override=override,
    )
    new_agent = agent.model_copy(
        update={
            "belief": belief,
            "feeling": feeling,
            "attention": appraisal.about,
            "voiced": voiced,
            "memory": Memory(episodes=[*stored, episode]),
            "pulls": pulls,
        }
    )
    update: dict[str, object] = {"agent": new_agent, "passes": (*ws.passes, record)}
    if ws.pass_index == 0:
        update["override"] = override
    return ws.model_copy(update=update)
