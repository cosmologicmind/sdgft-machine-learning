"""Symbolic distillation: extract human-readable formulas from trained models.

Uses PySR (symbolic regression) to discover closed-form expressions
that approximate the ML model's learned relationships.

This is the "knowledge extraction" step: from neural network weights
back to mathematical formulas that physicists can interpret.

Usage::

    from sdgft_ml.loop.distillation import distill_observable
    result = distill_observable(model, "n_s", edge_index)
    print(result.equation)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class DistillationResult:
    """Result of symbolic regression for one observable."""
    observable: str
    equation: str
    complexity: int
    mse: float
    r2: float
    n_points: int


def distill_observable(
    model: Any,
    observable_name: str,
    edge_index: Any,
    n_points: int = 5000,
    delta_range: tuple[float, float] = (0.05, 0.40),
    delta_g_range: tuple[float, float] = (0.01, 0.08),
    max_complexity: int = 20,
    n_iterations: int = 50,
    device: str = "cpu",
    seed: int = 42,
) -> DistillationResult:
    """Extract a symbolic formula for one observable from the surrogate.

    Generates a dataset of (Δ, δ_g) → observable predictions from
    the trained model, then runs PySR to find a closed-form expression.

    Parameters
    ----------
    model : SurrogateGNN
        Trained surrogate model.
    observable_name : str
        Which observable to distill.
    edge_index : array-like
        DAG edge index.
    n_points : int
        Number of evaluation points.
    max_complexity : int
        PySR maximum expression complexity.
    n_iterations : int
        PySR iterations.

    Returns
    -------
    DistillationResult
    """
    try:
        from pysr import PySRRegressor
    except ImportError:
        raise ImportError(
            "PySR is required for symbolic distillation. "
            "Install with: pip install 'sdgft-ml[symbolic]'"
        )

    import torch
    from ..data.dag_builder import observable_names

    names = observable_names()
    obs_idx = names.index(observable_name)

    # Generate predictions
    rng = np.random.default_rng(seed)
    phi = (1.0 + 5.0 ** 0.5) / 2.0

    deltas = rng.uniform(*delta_range, n_points)
    delta_gs = rng.uniform(*delta_g_range, n_points)
    X = np.column_stack([deltas, delta_gs])

    model.eval()
    if isinstance(edge_index, np.ndarray):
        edge_index = torch.from_numpy(edge_index)
    ei = edge_index.to(device)

    preds = []
    with torch.no_grad():
        for i in range(n_points):
            params = torch.tensor(
                [deltas[i], delta_gs[i], phi], dtype=torch.float32,
            ).unsqueeze(0).to(device)
            pred = model(params, ei).cpu().numpy()
            preds.append(pred[obs_idx])
    y = np.array(preds)

    # Run PySR
    sr_model = PySRRegressor(
        niterations=n_iterations,
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["log", "exp", "sqrt", "sin", "cos"],
        maxsize=max_complexity,
        populations=30,
        variable_names=["delta", "delta_g"],
        deterministic=True,
        random_state=seed,
        verbosity=0,
    )
    sr_model.fit(X, y)

    best = sr_model.get_best()
    equation = str(best["equation"])
    complexity = int(best["complexity"])
    mse = float(best["loss"])

    # R² on full data
    y_pred = sr_model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return DistillationResult(
        observable=observable_name,
        equation=equation,
        complexity=complexity,
        mse=mse,
        r2=r2,
        n_points=n_points,
    )


def distill_all(
    model: Any,
    edge_index: Any,
    observables: list[str] | None = None,
    **kwargs,
) -> list[DistillationResult]:
    """Distill symbolic formulas for multiple observables."""
    from ..data.dag_builder import observable_names

    if observables is None:
        # Focus on the most physics-relevant observables
        observables = [
            "n_s", "r_tensor", "omega_b", "omega_de",
            "w_de_fp", "alpha_em_inv_tree", "higgs_mass",
            "theta_12", "theta_23", "theta_13",
        ]

    results = []
    for name in observables:
        print(f"Distilling {name}...")
        try:
            result = distill_observable(model, name, edge_index, **kwargs)
            results.append(result)
            print(f"  {name} = {result.equation} (R² = {result.r2:.4f})")
        except Exception as e:
            print(f"  Failed: {e}")

    return results
