"""CLI prediction and visualization API."""

from .predict import main as predict_main
from .visualize import (
    plot_training_history,
    plot_parameter_landscape,
    plot_canary_report,
)

__all__ = [
    "predict_main",
    "plot_training_history",
    "plot_parameter_landscape",
    "plot_canary_report",
]
