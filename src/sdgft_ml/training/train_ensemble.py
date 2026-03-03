"""Ensemble surrogate training — train N GNN surrogates with different seeds.

An ensemble of independently-trained surrogates provides:
  • Better predictive accuracy (ensemble averaging)
  • Principled epistemic uncertainty (inter-model spread)
  • Robustness to seed-dependent local optima

Usage::

    from sdgft_ml.training.train_ensemble import train_ensemble, EnsembleConfig
    models, histories = train_ensemble(EnsembleConfig(n_members=5))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .train_surrogate import TrainConfig, TrainHistory, train_surrogate
from ..data.dag_builder import observable_names


# ── Config ────────────────────────────────────────────────────────

@dataclass
class EnsembleConfig:
    """Configuration for ensemble training."""
    n_members: int = 5
    base_seed: int = 42
    # Base training config (seed will be overridden per member)
    n_epochs: int = 500
    batch_size: int = 128
    lr: float = 3e-3
    hidden_dim: int = 64
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.10
    n_samples: int = 10_000
    loss_alpha: float = 0.7
    cosine_annealing: bool = True
    cosine_T_max: int = 0

    def member_config(self, member_idx: int) -> TrainConfig:
        """Create a TrainConfig for ensemble member `member_idx`."""
        return TrainConfig(
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            hidden_dim=self.hidden_dim,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
            n_samples=self.n_samples,
            loss_alpha=self.loss_alpha,
            cosine_annealing=self.cosine_annealing,
            cosine_T_max=self.cosine_T_max,
            seed=self.base_seed + member_idx,
            save_dir=f"runs/ensemble/member_{member_idx}",
        )


@dataclass
class EnsembleResult:
    """Results from ensemble training."""
    models: list[Any] = field(default_factory=list)
    histories: list[TrainHistory] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)

    @property
    def mean_val_loss(self) -> float:
        return float(np.mean(self.val_losses))

    @property
    def std_val_loss(self) -> float:
        return float(np.std(self.val_losses))


# ── Ensemble Training ─────────────────────────────────────────────

def train_ensemble(
    config: EnsembleConfig | None = None,
    device: str | None = None,
) -> EnsembleResult:
    """Train an ensemble of GNN surrogates with different seeds.

    Parameters
    ----------
    config : EnsembleConfig
        Ensemble configuration.
    device : str
        Torch device.

    Returns
    -------
    EnsembleResult with models, histories, and val_losses.
    """
    if config is None:
        config = EnsembleConfig()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    result = EnsembleResult()
    print(f"{'='*60}")
    print(f"Training Ensemble: {config.n_members} members")
    print(f"{'='*60}")

    for i in range(config.n_members):
        print(f"\n{'─'*40}")
        print(f"Member {i+1}/{config.n_members} (seed={config.base_seed + i})")
        print(f"{'─'*40}")
        member_cfg = config.member_config(i)
        model, history = train_surrogate(member_cfg, device=device)
        result.models.append(model)
        result.histories.append(history)
        result.val_losses.append(history.best_val_loss)

    print(f"\n{'='*60}")
    print(f"Ensemble Training Complete")
    print(f"  Val losses: {[f'{v:.6f}' for v in result.val_losses]}")
    print(f"  Mean: {result.mean_val_loss:.6f} ± {result.std_val_loss:.6f}")
    print(f"{'='*60}")

    return result


# ── Ensemble Prediction ───────────────────────────────────────────

def ensemble_predict(
    ensemble: EnsembleResult,
    delta: float,
    delta_g: float,
    phi: float,
    edge_index: torch.Tensor,
    norm_mean: np.ndarray | None = None,
    norm_std: np.ndarray | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Predict at a single point using the full ensemble.

    Returns mean predictions and epistemic standard deviations.

    Parameters
    ----------
    ensemble : EnsembleResult
    delta, delta_g, phi : input parameters
    edge_index : DAG edge index (on correct device)
    norm_mean, norm_std : normalization stats (from first member's history)

    Returns
    -------
    (means, stds) : dicts mapping observable name → value
    """
    names = observable_names()
    all_preds = []

    for model in ensemble.models:
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            params = torch.tensor(
                [delta, delta_g, phi], dtype=torch.float32
            ).unsqueeze(0).to(device)
            pred = model(params, edge_index).cpu().numpy()

        # Denormalize using that member's history norm stats
        if norm_mean is not None and norm_std is not None:
            pred = pred * norm_std + norm_mean

        all_preds.append(pred)

    preds = np.stack(all_preds, axis=0)  # (n_members, n_nodes)
    means = preds.mean(axis=0)
    stds = preds.std(axis=0)

    return (
        {n: float(means[i]) for i, n in enumerate(names)},
        {n: float(stds[i]) for i, n in enumerate(names)},
    )


def ensemble_predict_batch(
    ensemble: EnsembleResult,
    params_array: np.ndarray,
    edge_index: torch.Tensor,
    norm_mean: np.ndarray | None = None,
    norm_std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict for a batch of parameter sets.

    Parameters
    ----------
    params_array : (N, 3) parameter vectors
    edge_index : DAG edge index

    Returns
    -------
    (means, stds) : (N, n_nodes), (N, n_nodes)
    """
    names = observable_names()
    n_nodes = len(names)
    N = len(params_array)

    all_member_preds = []

    for model in ensemble.models:
        model.eval()
        device = next(model.parameters()).device
        member_preds = []

        with torch.no_grad():
            for i in range(N):
                params = torch.tensor(
                    params_array[i], dtype=torch.float32
                ).unsqueeze(0).to(device)
                pred = model(params, edge_index).cpu().numpy()

                if norm_mean is not None and norm_std is not None:
                    pred = pred * norm_std + norm_mean

                member_preds.append(pred)

        all_member_preds.append(np.array(member_preds))  # (N, n_nodes)

    stacked = np.stack(all_member_preds, axis=0)  # (n_members, N, n_nodes)
    means = stacked.mean(axis=0)  # (N, n_nodes)
    stds = stacked.std(axis=0)  # (N, n_nodes)
    return means, stds


def ensemble_canary_test(
    ensemble: EnsembleResult,
    edge_index: torch.Tensor | np.ndarray,
    device: str = "cpu",
    tolerance: float = 0.05,
) -> dict[str, Any]:
    """Run canary test on each ensemble member and report aggregate.

    Returns
    -------
    dict with per-member results, ensemble mean predictions,
    ensemble uncertainties, and aggregate pass rate.
    """
    import math
    from ..data.parameter_sweep import ParametricForward

    delta = 5.0 / 24.0
    delta_g = 1.0 / 24.0
    phi_val = (1.0 + math.sqrt(5.0)) / 2.0

    fwd = ParametricForward(delta=delta, delta_g=delta_g, phi=phi_val)
    exact = fwd.compute_all()
    names = observable_names()
    targets = np.array([exact[n] for n in names], dtype=np.float32)

    if isinstance(edge_index, np.ndarray):
        edge_index = torch.from_numpy(edge_index)

    # Use first member's normalization
    norm_mean = ensemble.histories[0].norm_mean
    norm_std = ensemble.histories[0].norm_std

    # All members predict — each has own norm stats
    all_preds = []
    for idx, model in enumerate(ensemble.models):
        model.eval()
        model = model.to(device)
        nm = ensemble.histories[idx].norm_mean
        ns = ensemble.histories[idx].norm_std
        ei = edge_index.to(device)
        with torch.no_grad():
            params = torch.tensor(
                [delta, delta_g, phi_val], dtype=torch.float32
            ).unsqueeze(0).to(device)
            pred = model(params, ei).cpu().numpy()
        # Denormalize with member's own stats
        pred = pred * ns + nm
        all_preds.append(pred)

    preds = np.stack(all_preds, axis=0)  # (n_members, n_nodes)
    ens_mean = preds.mean(axis=0)
    ens_std = preds.std(axis=0)

    # Evaluate ensemble mean
    rel_errors = {}
    passes = 0
    for i, name in enumerate(names):
        t = targets[i]
        p = ens_mean[i]
        if abs(t) > 1e-15:
            rel_err = abs(p - t) / abs(t)
        else:
            rel_err = abs(p - t)
        rel_errors[name] = rel_err
        if rel_err < tolerance:
            passes += 1

    pass_rate = passes / len(names)

    # Per-member pass rates
    member_pass_rates = []
    for m_idx in range(len(ensemble.models)):
        m_passes = 0
        for i, name in enumerate(names):
            t = targets[i]
            p = preds[m_idx, i]
            if abs(t) > 1e-15:
                re = abs(p - t) / abs(t)
            else:
                re = abs(p - t)
            if re < tolerance:
                m_passes += 1
        member_pass_rates.append(m_passes / len(names))

    return {
        "ensemble_predictions": {n: float(ens_mean[i]) for i, n in enumerate(names)},
        "ensemble_std": {n: float(ens_std[i]) for i, n in enumerate(names)},
        "targets": {n: float(targets[i]) for i, n in enumerate(names)},
        "relative_errors": rel_errors,
        "pass_rate": pass_rate,
        "n_passed": passes,
        "n_total": len(names),
        "member_pass_rates": member_pass_rates,
        "member_val_losses": ensemble.val_losses,
        "status": "PASS" if pass_rate > 0.90 else "FAIL",
    }


def print_ensemble_report(result: dict[str, Any]) -> None:
    """Pretty-print ensemble canary test results."""
    print(f"\n{'='*70}")
    print(f"ENSEMBLE CANARY TEST ({len(result['member_pass_rates'])} members)")
    print(f"{'='*70}")
    print(f"Ensemble status: {result['status']} "
          f"({result['n_passed']}/{result['n_total']} within 5%)")
    print(f"\nPer-member: {['%.1f%%' % (r*100) for r in result['member_pass_rates']]}")
    print(f"Val losses: {['%.6f' % v for v in result['member_val_losses']]}")

    print(f"\n{'Observable':<25s} {'Target':>12s} {'Ensemble':>12s} "
          f"{'±Std':>10s} {'RelErr':>10s} {'OK':>4s}")
    print("-" * 75)

    sorted_names = sorted(
        result["relative_errors"],
        key=lambda n: result["relative_errors"][n],
        reverse=True,
    )
    for name in sorted_names:
        t = result["targets"][name]
        p = result["ensemble_predictions"][name]
        s = result["ensemble_std"][name]
        err = result["relative_errors"][name]
        ok = "✓" if err < 0.05 else "✗"
        print(f"{name:<25s} {t:>12.6g} {p:>12.6g} {s:>10.4g} {err:>10.4%} {ok:>4s}")
