"""Training loops, evaluation, and experiment management."""

from .train_surrogate import train_surrogate
from .train_inverter import train_inverter
from .evaluate import evaluate_surrogate, canary_test

__all__ = [
    "train_surrogate",
    "train_inverter",
    "evaluate_surrogate",
    "canary_test",
]
