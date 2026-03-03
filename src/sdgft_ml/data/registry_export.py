"""Export SDGFT observables from the registry into ML-ready formats.

Usage::

    from sdgft_ml.data.registry_export import export_observables
    df = export_observables()          # pandas DataFrame
    records = export_observables(as_dict=True)  # list[dict]
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _import_registry():
    """Lazy import to avoid heavy SDGFT init at module level."""
    import sdgft  # noqa: F401 – triggers registration
    from sdgft import REGISTRY
    return REGISTRY


def observable_to_dict(obs) -> dict[str, Any]:
    """Convert a single Observable dataclass to a flat dict."""
    return {
        "name": obs.name,
        "symbol": obs.symbol,
        "formula": obs.formula,
        "predicted": float(obs.predicted),
        "observed": float(obs.observed),
        "observed_uncertainty": float(obs.observed_uncertainty),
        "unit": obs.unit,
        "level": obs.level,
        "d_star_variant": obs.d_star_variant,
        "dependencies": list(obs.dependencies),
        "is_upper_limit": obs.is_upper_limit,
        "is_diagnostic": obs.is_diagnostic,
        "deviation_abs": float(obs.deviation_abs),
        "deviation_percent": float(obs.deviation_percent),
        "sigma_tension": obs.sigma_tension,
        "status": obs.status,
    }


def export_observables(
    *,
    as_dict: bool = False,
    levels: list[int] | None = None,
    exclude_diagnostics: bool = False,
) -> pd.DataFrame | list[dict[str, Any]]:
    """Export all registered SDGFT observables.

    Parameters
    ----------
    as_dict : bool
        If True return ``list[dict]`` instead of DataFrame.
    levels : list[int] | None
        Filter to specific DAG levels (0-7). ``None`` keeps all.
    exclude_diagnostics : bool
        If True, drop observables marked ``is_diagnostic=True``.

    Returns
    -------
    DataFrame or list[dict] with one row per observable.
    """
    registry = _import_registry()
    records: list[dict[str, Any]] = []
    for obs in registry:
        if levels is not None and obs.level not in levels:
            continue
        if exclude_diagnostics and obs.is_diagnostic:
            continue
        records.append(observable_to_dict(obs))

    if as_dict:
        return records

    df = pd.DataFrame(records)
    # Ensure stable column order
    cols = [
        "name", "symbol", "predicted", "observed",
        "observed_uncertainty", "unit", "level", "d_star_variant",
        "dependencies", "deviation_abs", "deviation_percent",
        "sigma_tension", "status", "is_upper_limit", "is_diagnostic",
        "formula",
    ]
    return df[[c for c in cols if c in df.columns]]


def export_predicted_vector(
    names: list[str] | None = None,
) -> tuple[list[str], list[float]]:
    """Return ordered (names, predicted_values) for the default SDGFT point.

    Useful as the ground-truth "canary" vector at (Δ=5/24, δ=1/24).
    """
    registry = _import_registry()
    if names is None:
        names = [obs.name for obs in registry]
    values = [float(registry.get(n).predicted) for n in names]
    return names, values
