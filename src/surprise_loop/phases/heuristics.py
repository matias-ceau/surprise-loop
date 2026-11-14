"""Pure helpers shared by the stub phases, so Deliberate and Evaluate cannot drift apart."""

import math
from collections.abc import Sequence
from typing import NamedTuple

from surprise_loop.config import DeliberateParams
from surprise_loop.state import AgentState, Episode, GoalName
from surprise_loop.world import Action

PRESSES = (Action.PRESS_A, Action.PRESS_B)


class Estimate(NamedTuple):
    value: float  # E[x]
    n: int  # recalled episodes of the action


def episodes_of(action: Action, recalled: Sequence[Episode]) -> list[Episode]:
    """The recalled episodes of `action`, most recent first (`recalled` is oldest first)."""
    return [e for e in reversed(recalled) if e.action == action]


def expected_reward(
    action: Action, recalled: Sequence[Episode], alpha: float, params: DeliberateParams
) -> Estimate:
    """`E[x] = (p·P + Σ wᵢ·rᵢ) / (p + Σ wᵢ)`, `wᵢ = (1 − α)^i`, `p = prior_weight·(1 − α)^n`."""
    rewards = [e.reward for e in episodes_of(action, recalled)]
    n = len(rewards)
    decay = 1 - alpha
    weights = [decay**i for i in range(n)]
    p = params.prior_weight * decay**n
    numerator = p * params.prior_expected_reward + sum(
        w * r for w, r in zip(weights, rewards, strict=True)
    )
    return Estimate(value=numerator / (p + sum(weights)), n=n)


def goal_utilities(
    agent: AgentState, recalled: Sequence[Episode], params: DeliberateParams
) -> dict[Action, float]:
    """Goal-directed utility of every action, in `Action` order, without bias or mood terms."""
    priorities = agent.goals.priorities
    rs = agent.traits.reward_sensitivity
    alpha = agent.traits.surprise_learning_rate
    costs = params.action_costs
    utilities: dict[Action, float] = {}
    for action in Action:
        if action in PRESSES:
            e = expected_reward(action, recalled, alpha, params).value
            value = priorities[GoalName.SEEK_REWARD] * rs * e
        elif action == Action.EXPLORE:
            value = priorities[GoalName.EXPLORE] * params.explore_value
        else:
            value = priorities[GoalName.CONSERVE] * params.rest_value
        utilities[action] = value - costs[action]
    return utilities


def softmax(values: Sequence[float], temperature: float) -> list[float]:
    """Numerically stable softmax of `values / temperature`."""
    top = max(values)
    exps = [math.exp((v - top) / temperature) for v in values]
    total = sum(exps)
    return [x / total for x in exps]


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def require[T](value: T | None, name: str, phase: str) -> T:
    """`value`, or a ValueError naming the workspace field `phase` needs but lacks."""
    if value is None:
        raise ValueError(f"{phase} needs ws.{name}, which is missing")
    return value
