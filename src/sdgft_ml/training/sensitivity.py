"""Sensitivity & Jacobian analysis for SDGFT observables.

Computes how each observable responds to changes in the axiom parameters
(Δ, δ_g, φ).  This reveals:
  • Which observables are most sensitive to each parameter
  • The Fisher information content for parameter recovery
  • Optimal loss weights for the CVAE inverter

Usage::

    from sdgft_ml.training.sensitivity import (
        compute_jacobian, fisher_information, sensitivity_weights,
    )
    J = compute_jacobian()  # (37, 3) matrix
    F = fisher_information(J)  # (3, 3) matrix
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..data.parameter_sweep import ParametricForward


# ── Numerical Jacobian ────────────────────────────────────────────

def compute_jacobian(
    delta: float = 5.0 / 24.0,
    delta_g: float = 1.0 / 24.0,
    phi: float = (1.0 + math.sqrt(5.0)) / 2.0,
    h: float = 1e-6,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Compute the numerical Jacobian ∂obs_i / ∂param_j at a point.

    Uses central finite differences for O(h²) accuracy.

    Parameters
    ----------
    delta, delta_g, phi : evaluation point
    h : step size for finite differences

    Returns
    -------
    J : (n_obs, 3) Jacobian matrix
    obs_names : list of observable names (rows)
    param_names : list of parameter names (columns)
    """
    params = [delta, delta_g, phi]
    param_names = ["Δ", "δ_g", "φ"]

    # Central point
    fwd_0 = ParametricForward(delta=delta, delta_g=delta_g, phi=phi)
    obs_0 = fwd_0.feature_vector()
    obs_names = list(ParametricForward.OBSERVABLE_KEYS)
    n_obs = len(obs_names)

    J = np.zeros((n_obs, 3))

    for j in range(3):
        # Forward perturbation
        p_fwd = list(params)
        p_fwd[j] += h
        fwd_plus = ParametricForward(
            delta=p_fwd[0], delta_g=p_fwd[1], phi=p_fwd[2]
        )
        obs_plus = fwd_plus.feature_vector()

        # Backward perturbation
        p_bwd = list(params)
        p_bwd[j] -= h
        fwd_minus = ParametricForward(
            delta=p_bwd[0], delta_g=p_bwd[1], phi=p_bwd[2]
        )
        obs_minus = fwd_minus.feature_vector()

        # Central difference
        J[:, j] = (obs_plus - obs_minus) / (2 * h)

    return J, obs_names, param_names


def normalized_jacobian(
    delta: float = 5.0 / 24.0,
    delta_g: float = 1.0 / 24.0,
    phi: float = (1.0 + math.sqrt(5.0)) / 2.0,
    h: float = 1e-6,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Compute the dimensionless Jacobian: (param_j / obs_i) · ∂obs_i/∂param_j.

    This is the "elasticity" — the fractional change in observable per
    fractional change in parameter.

    Returns
    -------
    J_norm : (n_obs, 3) normalized Jacobian
    obs_names, param_names
    """
    J, obs_names, param_names = compute_jacobian(delta, delta_g, phi, h)
    params = np.array([delta, delta_g, phi])

    fwd = ParametricForward(delta=delta, delta_g=delta_g, phi=phi)
    obs = fwd.feature_vector()

    # Normalize: J_norm[i,j] = (param_j / obs_i) * J[i,j]
    J_norm = np.zeros_like(J)
    for i in range(len(obs)):
        for j in range(3):
            if abs(obs[i]) > 1e-15:
                J_norm[i, j] = (params[j] / obs[i]) * J[i, j]
            else:
                J_norm[i, j] = J[i, j] * params[j]  # absolute sensitivity

    return J_norm, obs_names, param_names


# ── Fisher Information ────────────────────────────────────────────

def fisher_information(
    J: np.ndarray,
    obs_sigmas: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the Fisher information matrix from the Jacobian.

    F_ij = Σ_k (∂obs_k/∂param_i)(∂obs_k/∂param_j) / σ_k²

    Parameters
    ----------
    J : (n_obs, n_params) Jacobian
    obs_sigmas : (n_obs,) measurement uncertainties.
                 If None, assumes unit weights.

    Returns
    -------
    F : (n_params, n_params) Fisher information matrix
    """
    n_obs, n_params = J.shape
    if obs_sigmas is None:
        # Unit weights
        W = np.eye(n_obs)
    else:
        W = np.diag(1.0 / obs_sigmas**2)

    # F = J^T W J
    F = J.T @ W @ J
    return F


def cramer_rao_bounds(F: np.ndarray) -> np.ndarray:
    """Compute Cramér-Rao lower bounds on parameter uncertainties.

    Returns
    -------
    sigma_min : (n_params,) minimum achievable std deviation
    """
    F_inv = np.linalg.inv(F)
    return np.sqrt(np.diag(F_inv))


# ── Sensitivity Weights for Inverter ──────────────────────────────

def sensitivity_weights(
    J: np.ndarray,
    target_param_idx: int = 1,  # δ_g by default
    alpha: float = 2.0,
) -> np.ndarray:
    """Compute per-observable weights that upweight δ_g-sensitive observables.

    Weights are proportional to |∂obs_i/∂param_target|^alpha, normalized.

    Parameters
    ----------
    J : (n_obs, n_params) Jacobian
    target_param_idx : which parameter to focus on (0=Δ, 1=δ_g, 2=φ)
    alpha : exponent (higher = more aggressive weighting)

    Returns
    -------
    weights : (n_obs,) normalized weights summing to n_obs
    """
    sensitivity = np.abs(J[:, target_param_idx])
    # Avoid zero weights
    sensitivity = sensitivity + 1e-10
    # Power law weighting
    w = sensitivity ** alpha
    # Normalize to sum = n_obs (so mean weight = 1)
    w = w * len(w) / w.sum()
    return w


def combined_sensitivity_weights(
    J: np.ndarray,
    param_importance: np.ndarray | None = None,
    alpha: float = 1.5,
) -> np.ndarray:
    """Compute per-observable weights that balance information across all parameters.

    For each observable, compute max sensitivity across parameters (weighted
    by param_importance) and use that to weight the reconstruction loss.

    Parameters
    ----------
    J : (n_obs, n_params) Jacobian
    param_importance : (n_params,) relative importance weights.
                       Default: [1, 3, 0.5] — upweight δ_g recovery.
    alpha : exponent for power-law weighting

    Returns
    -------
    weights : (n_obs,) normalized weights
    """
    if param_importance is None:
        # Default: upweight δ_g (idx=1) since it's hardest to recover
        param_importance = np.array([1.0, 3.0, 0.5])

    n_obs = J.shape[0]
    scores = np.zeros(n_obs)
    for i in range(n_obs):
        scores[i] = np.sum(np.abs(J[i, :]) * param_importance)

    scores = scores + 1e-10
    w = scores ** alpha
    w = w * n_obs / w.sum()
    return w


# ── Analysis Report ───────────────────────────────────────────────

def sensitivity_report(
    delta: float = 5.0 / 24.0,
    delta_g: float = 1.0 / 24.0,
    phi: float = (1.0 + math.sqrt(5.0)) / 2.0,
) -> dict[str, Any]:
    """Full sensitivity analysis report at a given parameter point.

    Returns
    -------
    dict with Jacobian, normalized Jacobian, Fisher matrix,
    Cramér-Rao bounds, and per-observable sensitivity breakdown.
    """
    J, obs_names, param_names = compute_jacobian(delta, delta_g, phi)
    J_norm, _, _ = normalized_jacobian(delta, delta_g, phi)
    F = fisher_information(J)
    cr_bounds = cramer_rao_bounds(F)

    # Per-observable breakdown
    obs_breakdown = []
    for i, name in enumerate(obs_names):
        obs_breakdown.append({
            "name": name,
            "dobs_dDelta": J[i, 0],
            "dobs_ddelta_g": J[i, 1],
            "dobs_dphi": J[i, 2],
            "elasticity_Delta": J_norm[i, 0],
            "elasticity_delta_g": J_norm[i, 1],
            "elasticity_phi": J_norm[i, 2],
            "total_sensitivity": np.sqrt(np.sum(J[i, :] ** 2)),
            "delta_g_fraction": abs(J[i, 1]) / (np.sum(np.abs(J[i, :])) + 1e-15),
        })

    # Sort by δ_g sensitivity
    obs_by_delta_g = sorted(obs_breakdown, key=lambda x: abs(x["dobs_ddelta_g"]), reverse=True)

    return {
        "jacobian": J,
        "jacobian_normalized": J_norm,
        "fisher_information": F,
        "cramer_rao_bounds": cr_bounds,
        "obs_names": obs_names,
        "param_names": param_names,
        "obs_breakdown": obs_breakdown,
        "obs_sorted_by_delta_g": obs_by_delta_g,
        "point": {"delta": delta, "delta_g": delta_g, "phi": phi},
    }


def print_sensitivity_report(report: dict[str, Any]) -> None:
    """Pretty-print the sensitivity analysis."""
    print(f"\n{'='*80}")
    print(f"SENSITIVITY ANALYSIS at (Δ={report['point']['delta']:.4f}, "
          f"δ_g={report['point']['delta_g']:.5f}, φ={report['point']['phi']:.4f})")
    print(f"{'='*80}")

    # Fisher information and Cramér-Rao bounds
    F = report["fisher_information"]
    cr = report["cramer_rao_bounds"]
    param_names = report["param_names"]
    print(f"\n── Fisher Information Matrix ──")
    for i, pi in enumerate(param_names):
        row = " ".join(f"{F[i,j]:12.4g}" for j in range(3))
        print(f"  {pi:>4s}: {row}")

    print(f"\n── Cramér-Rao Lower Bounds ──")
    for i, pi in enumerate(param_names):
        print(f"  σ_min({pi}) = {cr[i]:.6g}")

    # Observable sensitivity table
    print(f"\n── Per-Observable Sensitivity (sorted by |∂obs/∂δ_g|) ──")
    print(f"{'Observable':<25s} {'∂/∂Δ':>12s} {'∂/∂δ_g':>12s} {'∂/∂φ':>12s} "
          f"{'δ_g frac':>10s}")
    print("-" * 75)
    for obs in report["obs_sorted_by_delta_g"]:
        print(f"{obs['name']:<25s} "
              f"{obs['dobs_dDelta']:>12.4g} "
              f"{obs['dobs_ddelta_g']:>12.4g} "
              f"{obs['dobs_dphi']:>12.4g} "
              f"{obs['delta_g_fraction']:>10.1%}")

    # Elasticity table
    print(f"\n── Elasticities (dimensionless, sorted by |ε_δ_g|) ──")
    sorted_by_elast = sorted(
        report["obs_breakdown"],
        key=lambda x: abs(x["elasticity_delta_g"]),
        reverse=True,
    )
    print(f"{'Observable':<25s} {'ε_Δ':>10s} {'ε_δ_g':>10s} {'ε_φ':>10s}")
    print("-" * 60)
    for obs in sorted_by_elast:
        print(f"{obs['name']:<25s} "
              f"{obs['elasticity_Delta']:>10.4f} "
              f"{obs['elasticity_delta_g']:>10.4f} "
              f"{obs['elasticity_phi']:>10.4f}")
