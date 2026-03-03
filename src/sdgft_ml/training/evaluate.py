"""Evaluation utilities and the canary test.

The "canary test" verifies the surrogate at the true SDGFT axiom point
(Δ=5/24, δ_g=1/24, φ=golden ratio) and compares against registry values.

Usage::

    from sdgft_ml.training.evaluate import canary_test, evaluate_surrogate
    results = canary_test(model, edge_index)
    metrics = evaluate_surrogate(model, val_loader, edge_index)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from ..data.parameter_sweep import ParametricForward
from ..data.dag_builder import build_dag, build_edge_index, observable_names


def evaluate_surrogate(
    model: Any,
    val_data: np.ndarray,
    val_targets: np.ndarray,
    edge_index: np.ndarray,
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate the surrogate on validation data.

    Parameters
    ----------
    model : SurrogateGNN
    val_data : (N, 3) parameter vectors
    val_targets : (N, n_nodes) target observable values
    edge_index : (2, E)
    device : str

    Returns
    -------
    dict with per-observable MSE, MAE, R², and overall metrics
    """
    model.eval()
    model = model.to(device)
    names = observable_names()
    n_nodes = len(names)

    all_preds = []
    with torch.no_grad():
        for i in range(len(val_data)):
            params = torch.tensor(val_data[i], dtype=torch.float32).unsqueeze(0).to(device)
            ei = torch.from_numpy(edge_index).to(device)
            pred = model(params, ei).cpu().numpy()
            all_preds.append(pred)

    preds = np.array(all_preds)  # (N, n_nodes)

    # Per-observable metrics
    per_obs = {}
    for j, name in enumerate(names):
        y_true = val_targets[:, j]
        y_pred = preds[:, j]
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        per_obs[name] = {"mse": mse, "mae": mae, "r2": r2}

    # Overall
    overall_mse = np.mean([(m["mse"]) for m in per_obs.values()])
    overall_r2 = np.mean([m["r2"] for m in per_obs.values() if not np.isnan(m["r2"])])

    return {
        "per_observable": per_obs,
        "overall_mse": overall_mse,
        "overall_r2": overall_r2,
        "n_samples": len(val_data),
    }


def canary_test(
    model: Any,
    edge_index: np.ndarray | torch.Tensor,
    device: str = "cpu",
    tolerance: float = 0.05,
) -> dict[str, Any]:
    """Canary test: evaluate the surrogate at the true SDGFT point.

    Compares ML predictions against exact SDGFT computations at
    (Δ=5/24, δ_g=1/24, φ=golden ratio).

    Parameters
    ----------
    model : SurrogateGNN
    edge_index : DAG edge index
    device : str
    tolerance : float
        Relative tolerance for "pass" status.

    Returns
    -------
    dict with:
        - predictions: model outputs
        - targets: exact SDGFT values
        - relative_errors: per-observable relative error
        - pass_rate: fraction of observables within tolerance
        - status: "PASS" or "FAIL"
    """
    delta = 5.0 / 24.0
    delta_g = 1.0 / 24.0
    phi = (1.0 + math.sqrt(5.0)) / 2.0

    # Exact SDGFT computation
    fwd = ParametricForward(delta=delta, delta_g=delta_g, phi=phi)
    exact = fwd.compute_all()
    names = observable_names()
    targets = np.array([exact[n] for n in names], dtype=np.float32)

    # Model prediction
    model.eval()
    model = model.to(device)
    if isinstance(edge_index, np.ndarray):
        edge_index = torch.from_numpy(edge_index)
    ei = edge_index.to(device)

    with torch.no_grad():
        params = torch.tensor([delta, delta_g, phi], dtype=torch.float32).unsqueeze(0).to(device)
        pred = model(params, ei).cpu().numpy()

    # Compute relative errors
    rel_errors = {}
    passes = 0
    for i, name in enumerate(names):
        target = targets[i]
        prediction = pred[i]
        if abs(target) > 1e-15:
            rel_err = abs(prediction - target) / abs(target)
        else:
            rel_err = abs(prediction - target)
        rel_errors[name] = rel_err
        if rel_err < tolerance:
            passes += 1

    pass_rate = passes / len(names)
    status = "PASS" if pass_rate > 0.90 else "FAIL"

    return {
        "predictions": {n: pred[i] for i, n in enumerate(names)},
        "targets": {n: targets[i] for i, n in enumerate(names)},
        "relative_errors": rel_errors,
        "pass_rate": pass_rate,
        "status": status,
        "n_passed": passes,
        "n_total": len(names),
    }


def print_canary_report(result: dict[str, Any]) -> None:
    """Pretty-print the canary test results."""
    print(f"\n{'='*60}")
    print(f"CANARY TEST at (Δ=5/24, δ_g=1/24, φ=golden)")
    print(f"{'='*60}")
    print(f"Status: {result['status']} ({result['n_passed']}/{result['n_total']} "
          f"within 5% tolerance, pass rate: {result['pass_rate']:.1%})")
    print(f"\n{'Observable':<25s} {'Target':>12s} {'Predicted':>12s} {'RelErr':>10s} {'OK':>4s}")
    print("-" * 65)

    for name in sorted(result["relative_errors"], key=lambda n: result["relative_errors"][n], reverse=True):
        target = result["targets"][name]
        pred = result["predictions"][name]
        err = result["relative_errors"][name]
        ok = "✓" if err < 0.05 else "✗"
        print(f"{name:<25s} {target:>12.6g} {pred:>12.6g} {err:>10.4%} {ok:>4s}")


def evaluate_inverter(
    model: Any,
    n_test: int = 100,
    device: str = "cpu",
    seed: int = 123,
) -> dict[str, Any]:
    """Evaluate the CVAE inverter on random samples.

    For each test sample, compute observables → invert → compare
    recovered parameters to true parameters.
    """
    rng = np.random.default_rng(seed)
    delta_range = (0.05, 0.40)
    delta_g_range = (0.01, 0.08)
    phi_default = (1.0 + math.sqrt(5.0)) / 2.0

    errors = {"delta": [], "delta_g": [], "phi": []}

    for _ in range(n_test):
        delta = rng.uniform(*delta_range)
        delta_g = rng.uniform(*delta_g_range)
        fwd = ParametricForward(delta=delta, delta_g=delta_g, phi=phi_default)
        obs = fwd.feature_vector()

        obs_tensor = torch.from_numpy(obs.astype(np.float32)).to(device)
        mean_params, _ = model.invert(obs_tensor, n_samples=50)
        pred = mean_params.cpu().numpy()

        errors["delta"].append(abs(pred[0] - delta))
        errors["delta_g"].append(abs(pred[1] - delta_g))
        errors["phi"].append(abs(pred[2] - phi_default))

    return {
        "mean_abs_error": {k: np.mean(v) for k, v in errors.items()},
        "std_abs_error": {k: np.std(v) for k, v in errors.items()},
        "n_test": n_test,
    }
