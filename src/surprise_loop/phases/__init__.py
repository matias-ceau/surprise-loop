"""The seven phases of one tick; each is `async (Workspace, Runtime) -> Workspace`."""

from surprise_loop.phases.act import act
from surprise_loop.phases.appraise import appraise
from surprise_loop.phases.deliberate import deliberate
from surprise_loop.phases.evaluate import evaluate
from surprise_loop.phases.monitor import monitor
from surprise_loop.phases.perceive import perceive
from surprise_loop.phases.reconcile import reconcile

__all__ = ["act", "appraise", "deliberate", "evaluate", "monitor", "perceive", "reconcile"]
