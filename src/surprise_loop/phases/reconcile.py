"""Reconcile: apply one pass's outcome to mental state through four channels.

Each channel has its own timescale (see "Deterministic maths" in docs/plan-v0.md):

- mood (fast): valence and arousal move with signed and unsigned prediction error;
- memory (per pass): one episode, salience weighted by unsigned prediction error;
- goals (slow): a per-goal error EMA; a priority is revised only after `patience` consecutive
  over-threshold passes of one sign;
- dispositions (slowest): only the taken action's bias drifts towards the outcome value.

On pass 0, a downward goal revision that coincides with a strong habit for the action taken is a
`habit_vs_goal` conflict, which sends control back to Deliberate once. Reconcile never changes
`agent.tick`; `tick()` does.
"""

import math

from surprise_loop.config import Params, Runtime
from surprise_loop.phases.heuristics import clamp, require
from surprise_loop.state import (
    GOAL_OF_ACTION,
    AgentState,
    Conflict,
    Dispositions,
    Episode,
    Feedback,
    GoalName,
    Goals,
    Memory,
    Mood,
    PassRecord,
    PredictionError,
    Workspace,
)
from surprise_loop.world import Action, Outcome


def _sign(value: float) -> int:
    return (value > 0) - (value < 0)


def update_mood(mood: Mood, error: PredictionError, agent: AgentState, params: Params) -> Mood:
    """`v ← m·v + (1 − m)·tanh(RS·signed)`; `a ← m·a + (1 − m)·min(1, gain·unsigned)`."""
    m = agent.traits.mood_inertia
    rs = agent.traits.reward_sensitivity
    valence = m * mood.valence + (1 - m) * math.tanh(rs * error.signed)
    drive = min(1, params.mood.arousal_gain * error.unsigned)
    arousal = m * mood.arousal + (1 - m) * drive
    return Mood(valence=clamp(valence, -1, 1), arousal=clamp(arousal, 0, 1))


def update_goals(
    goals: Goals, goal: GoalName, error: PredictionError, alpha: float, params: Params
) -> tuple[Goals, int]:
    """The goals after one pass serving `goal`, and the revision's sign (0 when none)."""
    p = params.goals
    beta = p.ema_rate * alpha
    ema = (1 - beta) * goals.error_ema[goal] + beta * error.signed
    s = _sign(ema) if abs(ema) > p.error_threshold else 0
    streak = goals.error_streak[goal]
    if s == 0:
        streak = 0
    elif _sign(streak) == s:
        streak += s
    else:
        streak = s

    priorities = dict(goals.priorities)
    revisions = goals.revisions
    revised = 0
    if s != 0 and abs(streak) >= p.patience:
        priorities[goal] = clamp(
            priorities[goal] + p.revision_step * s, p.min_priority, p.max_priority
        )
        streak = 0
        ema = 0.0
        revisions += 1
        revised = s

    updated = Goals(
        priorities=priorities,
        error_ema={**goals.error_ema, goal: ema},
        error_streak={**goals.error_streak, goal: streak},
        revisions=revisions,
    )
    return updated, revised


def update_dispositions(
    dispositions: Dispositions,
    action: Action,
    reward: float,
    feedback: Feedback,
    agent: AgentState,
    params: Params,
) -> Dispositions:
    """`bias[a] ← clamp(bias[a] + rate·α·(v − bias[a]), ±bound)`, `v = RS·reward + w·alignment`."""
    p = params.dispositions
    alpha = agent.traits.surprise_learning_rate
    value = agent.traits.reward_sensitivity * reward + p.alignment_weight * feedback.alignment
    bias = dispositions.biases[action]
    bias = clamp(bias + p.rate * alpha * (value - bias), -p.bound, p.bound)
    return Dispositions(biases={**dispositions.biases, action: bias})


def _episode(
    tick: int,
    ws: Workspace,
    outcome: Outcome,
    error: PredictionError,
    feedback: Feedback,
    params: Params,
) -> Episode:
    decision = require(ws.decision, "decision", "reconcile")
    return Episode(
        tick=tick,
        step=outcome.info.step,
        action=decision.action,
        prediction=decision.prediction,
        event=outcome.observation.event,
        reward=outcome.observation.reward,
        prediction_error=error,
        alignment=feedback.alignment,
        critique=feedback.critique,
        salience=clamp(params.memory.base_salience + error.unsigned, 0, 1),
    )


async def reconcile(ws: Workspace, rt: Runtime) -> Workspace:
    decision = require(ws.decision, "decision", "reconcile")
    outcome = require(ws.outcome, "outcome", "reconcile")
    error = require(ws.prediction_error, "prediction_error", "reconcile")
    feedback = require(ws.feedback, "feedback", "reconcile")
    params = rt.config.params
    agent = ws.agent
    tick = agent.tick + 1
    action = decision.action
    goal = GOAL_OF_ACTION[action]

    mood = update_mood(agent.mood, error, agent, params)
    memory = Memory(
        episodes=[*agent.memory.episodes, _episode(tick, ws, outcome, error, feedback, params)]
    )
    goals, revised = update_goals(
        agent.goals, goal, error, agent.traits.surprise_learning_rate, params
    )
    dispositions = update_dispositions(
        agent.dispositions, action, outcome.observation.reward, feedback, agent, params
    )

    conflict: Conflict | None = None
    bias = dispositions.biases[action]
    if (
        ws.pass_index == 0
        and params.conflict.enabled
        and revised < 0
        and bias >= params.conflict.habit_threshold
    ):
        conflict = Conflict(
            kind="habit_vs_goal",
            action=action,
            detail=(
                f"goal {goal.value} revised down to {goals.priorities[goal]} while "
                f"bias[{action.value}] = {bias} >= {params.conflict.habit_threshold}"
            ),
        )

    record = PassRecord(
        tick=tick,
        pass_index=ws.pass_index,
        decision=decision,
        outcome=outcome,
        prediction_error=error,
        feedback=feedback,
        goal_revised=goal if revised else None,
        conflict=conflict,
    )
    new_agent = agent.model_copy(
        update={"mood": mood, "memory": memory, "goals": goals, "dispositions": dispositions}
    )
    update: dict[str, object] = {"agent": new_agent, "passes": (*ws.passes, record)}
    if ws.pass_index == 0:
        update["conflict"] = conflict
    return ws.model_copy(update=update)
