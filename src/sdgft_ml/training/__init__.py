"""Training loops, evaluation, and experiment management."""

from .train_surrogate import train_surrogate
from .train_inverter import train_inverter
from .train_ensemble import train_ensemble
from .evaluate import evaluate_surrogate, canary_test
from .sensitivity import compute_jacobian, sensitivity_report
from .round_trip import round_trip_test, round_trip_with_exact

__all__ = [
    "train_surrogate",
    "train_inverter",
    "train_ensemble",
    "evaluate_surrogate",
    "canary_test",
    "compute_jacobian",
    "sensitivity_report",
    "round_trip_test",
    "round_trip_with_exact",
]
