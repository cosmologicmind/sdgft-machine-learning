"""Round-trip pipeline test: params → Surrogate → Inverter → compare.

Validates the entire ML pipeline end-to-end by:
1. Generating random parameter sets
2. Running them through the surrogate (forward prediction)
3. Running surrogate output through the inverter (parameter recovery)
4. Comparing recovered parameters to originals

Usage::

    from sdgft_ml.training.round_trip import round_trip_test
    results = round_trip_test(surrogate, inverter, edge_index)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from ..data.parameter_sweep import ParametricForward
from ..data.dag_builder import observable_names


def round_trip_test(
    surrogate: Any,
    inverter: Any,
    edge_index: torch.Tensor | np.ndarray,
    n_test: int = 500,
    device: str = "cpu",
    seed: int = 777,
    norm_mean: np.ndarray | None = None,
    norm_std: np.ndarray | None = None,
    obs_mean: np.ndarray | None = None,
    obs_std: np.ndarray | None = None,
    param_min: np.ndarray | None = None,
    param_max: np.ndarray | None = None,
    use_log_features: bool = False,
    clip_z: float = 10.0,
) -> dict[str, Any]:
    """End-to-end round-trip pipeline test.

    Parameters
    ----------
    surrogate : SurrogateGNN  (trained)
    inverter : InverterCVAE  (trained)
    edge_index : DAG edge index
    n_test : number of test points
    device : torch device
    seed : RNG seed
    norm_mean, norm_std : surrogate normalization stats
    obs_mean, obs_std : inverter observable normalization stats
    param_min, param_max : inverter parameter denormalization bounds (if normalize_params)
    use_log_features : whether the inverter uses log-augmented features
    clip_z : float
        Clip normalized inputs to ±clip_z standard deviations.
        Prevents near-constant observables from producing extreme
        z-scores due to surrogate approximation error. Set to 0 to disable.

    Returns
    -------
    dict with per-parameter MAE, R², and full arrays for plotting
    """
    rng = np.random.default_rng(seed)
    phi_default = (1.0 + math.sqrt(5.0)) / 2.0

    if isinstance(edge_index, np.ndarray):
        edge_index = torch.from_numpy(edge_index)
    ei = edge_index.to(device)

    surrogate.eval()
    inverter.eval()
    surrogate = surrogate.to(device)
    inverter = inverter.to(device)

    # Random parameter sets
    deltas = rng.uniform(0.05, 0.40, n_test).astype(np.float32)
    delta_gs = rng.uniform(0.01, 0.08, n_test).astype(np.float32)
    phis = np.full(n_test, phi_default, dtype=np.float32)

    true_params = np.stack([deltas, delta_gs, phis], axis=1)  # (N, 3)
    recovered_params = np.zeros_like(true_params)

    with torch.no_grad():
        for i in range(n_test):
            # Step 1: Surrogate forward pass
            params_t = torch.tensor(
                [deltas[i], delta_gs[i], phis[i]], dtype=torch.float32
            ).unsqueeze(0).to(device)
            surr_pred = surrogate(params_t, ei).cpu().numpy()  # (n_nodes,)

            # Denormalize surrogate output
            if norm_mean is not None and norm_std is not None:
                surr_obs = surr_pred * norm_std + norm_mean
            else:
                surr_obs = surr_pred

            # Step 2: Prepare inverter input
            if obs_mean is not None and obs_std is not None:
                if use_log_features:
                    # Log-augment then normalize
                    obs_log = np.log1p(np.abs(surr_obs)) * np.sign(surr_obs)
                    obs_aug = np.concatenate([surr_obs, obs_log])
                    obs_input = (obs_aug - obs_mean) / obs_std
                else:
                    obs_input = (surr_obs - obs_mean) / obs_std
                # Clip extreme z-scores from surrogate approximation error
                # (near-constant observables can produce huge z-scores)
                if clip_z > 0:
                    obs_input = np.clip(obs_input, -clip_z, clip_z)
            else:
                obs_input = surr_obs

            # Step 3: Inverter recovery
            obs_t = torch.tensor(obs_input, dtype=torch.float32).to(device)
            mean_p, _ = inverter.invert(obs_t, n_samples=50)
            rec = mean_p.cpu().numpy()

            # Denormalize parameters if needed
            if param_min is not None and param_max is not None:
                param_range = param_max - param_min
                param_range[param_range < 1e-12] = 1.0
                rec = rec * param_range + param_min

            recovered_params[i] = rec

    # Compute metrics
    param_names = ["Δ", "δ_g", "φ"]
    results = {"n_test": n_test, "per_param": {}}

    for j, name in enumerate(param_names):
        true_col = true_params[:, j]
        rec_col = recovered_params[:, j]
        err = np.abs(true_col - rec_col)
        mae = np.mean(err)
        rmse = np.sqrt(np.mean(err**2))
        ss_res = np.sum((true_col - rec_col) ** 2)
        ss_tot = np.sum((true_col - true_col.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        results["per_param"][name] = {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "max_err": float(np.max(err)),
            "median_err": float(np.median(err)),
        }

    # Overall
    all_mae = np.mean([r["mae"] for r in results["per_param"].values()])
    all_r2 = np.mean([r["r2"] for r in results["per_param"].values()])
    results["overall_mae"] = float(all_mae)
    results["overall_r2"] = float(all_r2)

    # Store arrays for plotting
    results["true_params"] = true_params
    results["recovered_params"] = recovered_params

    return results


def round_trip_with_exact(
    inverter: Any,
    n_test: int = 500,
    device: str = "cpu",
    seed: int = 777,
    obs_mean: np.ndarray | None = None,
    obs_std: np.ndarray | None = None,
    param_min: np.ndarray | None = None,
    param_max: np.ndarray | None = None,
    use_log_features: bool = False,
) -> dict[str, Any]:
    """Round-trip using exact SDGFT computation (bypasses surrogate).

    This isolates inverter quality from surrogate error.
    """
    rng = np.random.default_rng(seed)
    phi_default = (1.0 + math.sqrt(5.0)) / 2.0

    inverter.eval()
    inverter = inverter.to(device)

    deltas = rng.uniform(0.05, 0.40, n_test).astype(np.float32)
    delta_gs = rng.uniform(0.01, 0.08, n_test).astype(np.float32)
    phis = np.full(n_test, phi_default, dtype=np.float32)
    true_params = np.stack([deltas, delta_gs, phis], axis=1)
    recovered_params = np.zeros_like(true_params)

    with torch.no_grad():
        for i in range(n_test):
            fwd = ParametricForward(
                delta=float(deltas[i]),
                delta_g=float(delta_gs[i]),
                phi=float(phis[i]),
            )
            obs_raw = fwd.feature_vector().astype(np.float32)

            if obs_mean is not None and obs_std is not None:
                if use_log_features:
                    obs_log = np.log1p(np.abs(obs_raw)) * np.sign(obs_raw)
                    obs_aug = np.concatenate([obs_raw, obs_log])
                    obs_input = (obs_aug - obs_mean) / obs_std
                else:
                    obs_input = (obs_raw - obs_mean) / obs_std
            else:
                obs_input = obs_raw

            obs_t = torch.tensor(obs_input, dtype=torch.float32).to(device)
            mean_p, _ = inverter.invert(obs_t, n_samples=50)
            rec = mean_p.cpu().numpy()

            if param_min is not None and param_max is not None:
                param_range = param_max - param_min
                param_range[param_range < 1e-12] = 1.0
                rec = rec * param_range + param_min

            recovered_params[i] = rec

    param_names = ["Δ", "δ_g", "φ"]
    results = {"n_test": n_test, "per_param": {}}

    for j, name in enumerate(param_names):
        true_col = true_params[:, j]
        rec_col = recovered_params[:, j]
        err = np.abs(true_col - rec_col)
        mae = float(np.mean(err))
        rmse = float(np.sqrt(np.mean(err**2)))
        ss_res = np.sum((true_col - rec_col) ** 2)
        ss_tot = np.sum((true_col - true_col.mean()) ** 2)
        r2 = float(1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))
        results["per_param"][name] = {
            "mae": mae, "rmse": rmse, "r2": r2,
            "max_err": float(np.max(err)),
            "median_err": float(np.median(err)),
        }

    all_mae = np.mean([r["mae"] for r in results["per_param"].values()])
    all_r2 = np.mean([r["r2"] for r in results["per_param"].values()])
    results["overall_mae"] = float(all_mae)
    results["overall_r2"] = float(all_r2)
    results["true_params"] = true_params
    results["recovered_params"] = recovered_params

    return results


def print_round_trip_report(results: dict[str, Any], title: str = "Round-Trip") -> None:
    """Pretty-print round-trip test results."""
    print(f"\n{'='*60}")
    print(f"{title} Pipeline Test ({results['n_test']} samples)")
    print(f"{'='*60}")
    print(f"\n{'Parameter':<12s} {'MAE':>10s} {'RMSE':>10s} {'R²':>10s} "
          f"{'MaxErr':>10s} {'MedErr':>10s}")
    print("-" * 65)
    for name, m in results["per_param"].items():
        print(f"{name:<12s} {m['mae']:>10.6f} {m['rmse']:>10.6f} "
              f"{m['r2']:>10.4f} {m['max_err']:>10.6f} {m['median_err']:>10.6f}")
    print(f"\n  Overall MAE: {results['overall_mae']:.6f}, R²: {results['overall_r2']:.4f}")
