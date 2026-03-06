"""Experimental reference data for SDGFT validation.

Contains 22 precision measurements from PDG 2024, Planck 2018,
NuFIT 5.3, BICEP/Keck 2021 and derived values.

Usage::

    from sdgft_ml.validation import EXPERIMENTAL_DATA, ExperimentalValue
    from sdgft_ml.validation import validate_at_axiom, scorecard
"""

from .experimental_data import (
    EXPERIMENTAL_DATA,
    ExperimentalValue,
    validate_at_point,
    validate_at_axiom,
    chi_squared,
    scorecard,
    validate_surrogate_vs_real,
)

__all__ = [
    "EXPERIMENTAL_DATA",
    "ExperimentalValue",
    "validate_at_point",
    "validate_at_axiom",
    "chi_squared",
    "scorecard",
    "validate_surrogate_vs_real",
]
