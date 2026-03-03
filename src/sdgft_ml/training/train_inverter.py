"""Training loop for the Conditional VAE inverter.

Trains the CVAE to invert: observables → (Δ, δ_g, φ).

Usage::

    from sdgft_ml.training.train_inverter import train_inverter
    model, history = train_inverter(n_epochs=300, n_samples=5000)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ..data.parameter_sweep import ParametricForward, sweep_latin_hypercube


# ── Dataset ───────────────────────────────────────────────────────

class InverterDataset(Dataset):
    """Dataset of (observables, params) pairs for CVAE training."""

    def __init__(
        self,
        observables: np.ndarray,
        params: np.ndarray,
        obs_mean: np.ndarray | None = None,
        obs_std: np.ndarray | None = None,
    ):
        if obs_mean is not None and obs_std is not None:
            observables = (observables - obs_mean) / obs_std
        self.observables = torch.from_numpy(observables.astype(np.float32))
        self.params = torch.from_numpy(params.astype(np.float32))

    def __len__(self) -> int:
        return self.observables.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.observables[idx], self.params[idx]


# ── Config ────────────────────────────────────────────────────────

@dataclass
class InverterConfig:
    """Training hyperparameters for the CVAE inverter."""
    n_epochs: int = 300
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    scheduler_patience: int = 20
    scheduler_factor: float = 0.5
    beta_start: float = 0.0
    beta_end: float = 1.0
    beta_warmup_epochs: int = 50
    beta_cyclical: bool = False
    beta_n_cycles: int = 4
    free_bits: float = 0.0
    normalize_params: bool = False
    n_samples: int = 5000
    val_frac: float = 0.15
    hidden_dim: int = 128
    latent_dim: int = 16
    n_hidden: int = 3
    seed: int = 42
    save_dir: str = "runs/inverter"
    # v3: per-parameter loss weights  [Δ, δ_g, φ]
    param_weights: list[float] | None = None
    # v3: sensitivity-weighted observable features
    use_sensitivity_weights: bool = False
    # v3: log-scale augmented features (doubles input dim)
    use_log_features: bool = False
    # v3: cosine annealing for LR
    cosine_annealing: bool = False
    # v4: minimum floor for obs_std normalization (match surrogate)
    min_obs_std: float = 1e-12


@dataclass
class InverterHistory:
    """Training metrics."""
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    recon_loss: list[float] = field(default_factory=list)
    kl_loss: list[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = 0


def train_inverter(
    config: InverterConfig | None = None,
    device: str | None = None,
) -> tuple[Any, InverterHistory]:
    """Train the CVAE inverter model.

    Parameters
    ----------
    config : InverterConfig
        Training configuration.
    device : str
        Torch device.

    Returns
    -------
    model, history
    """
    from ..models.inverter import InverterCVAE

    if config is None:
        config = InverterConfig()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Training CVAE Inverter on {device}")

    # Data
    df = sweep_latin_hypercube(n_samples=config.n_samples, seed=config.seed)
    obs_keys = ParametricForward.OBSERVABLE_KEYS
    param_keys = ParametricForward.PARAM_KEYS

    observables = df[obs_keys].values.astype(np.float32)
    params = df[param_keys].values.astype(np.float32)

    # v3: log-scale augmented features
    if config.use_log_features:
        obs_log = np.log1p(np.abs(observables)) * np.sign(observables)
        observables = np.concatenate([observables, obs_log], axis=1)
        print(f"  Log features enabled: input dim {observables.shape[1]}")

    # Normalize observables
    obs_mean = observables.mean(axis=0)
    obs_std = observables.std(axis=0)
    obs_std[obs_std < config.min_obs_std] = config.min_obs_std

    # v3: sensitivity-weighted features (rescale columns)
    sens_weights_np = None
    if config.use_sensitivity_weights:
        from .sensitivity import compute_jacobian, combined_sensitivity_weights
        J, _, _ = compute_jacobian()
        if config.use_log_features:
            # Duplicate weights for log features
            w = combined_sensitivity_weights(J, alpha=1.5)
            sens_weights_np = np.concatenate([w, w])
        else:
            sens_weights_np = combined_sensitivity_weights(J, alpha=1.5)
        print(f"  Sensitivity weights: min={sens_weights_np.min():.3f}, "
              f"max={sens_weights_np.max():.3f}")

    # Optionally normalize parameters to [0,1] for balanced MSE
    param_min = params.min(axis=0) if config.normalize_params else None
    param_max = params.max(axis=0) if config.normalize_params else None
    if config.normalize_params:
        param_range = param_max - param_min
        param_range[param_range < 1e-12] = 1.0
        params_norm = (params - param_min) / param_range
    else:
        params_norm = params

    # Split
    n_val = int(len(params) * config.val_frac)
    rng = np.random.default_rng(config.seed)
    indices = rng.permutation(len(params))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    train_ds = InverterDataset(
        observables[train_idx], params_norm[train_idx], obs_mean, obs_std,
    )
    val_ds = InverterDataset(
        observables[val_idx], params_norm[val_idx], obs_mean, obs_std,
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Model
    n_obs = observables.shape[1]  # may be 2x if log features
    model = InverterCVAE(
        n_observables=n_obs,
        n_params=len(param_keys),
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        n_hidden=config.n_hidden,
    ).to(device)

    # CRITICAL: when normalize_params=True, targets are in [0,1],
    # so the model's sigmoid output range must also be [0,1].
    if config.normalize_params:
        with torch.no_grad():
            model.param_min.fill_(0.0)
            model.param_max.fill_(1.0)
        print("  Output range adjusted to [0,1] for normalized params")

    n_params_total = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params_total:,}")

    # v3: per-parameter loss weights
    pw = None
    if config.param_weights is not None:
        pw = torch.tensor(config.param_weights, dtype=torch.float32, device=device)
        print(f"  Param weights: {config.param_weights}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )
    if config.cosine_annealing:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config.n_epochs // 3, T_mult=2, eta_min=1e-6,
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=config.scheduler_patience, factor=config.scheduler_factor,
        )

    # Training
    history = InverterHistory()
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.n_epochs):
        # β-schedule: cyclical or monotonic warmup
        if config.beta_cyclical:
            # Cyclical annealing: N full warmup cycles
            cycle_len = config.n_epochs / config.beta_n_cycles
            t = (epoch % cycle_len) / max(config.beta_warmup_epochs, 1)
            beta = config.beta_start + (config.beta_end - config.beta_start) * min(t, 1.0)
        elif epoch < config.beta_warmup_epochs:
            beta = config.beta_start + (
                (config.beta_end - config.beta_start)
                * epoch / config.beta_warmup_epochs
            )
        else:
            beta = config.beta_end

        # ── Train ──
        model.train()
        epoch_loss, epoch_recon, epoch_kl = [], [], []
        for obs_batch, params_batch in train_loader:
            obs_batch = obs_batch.to(device)
            params_batch = params_batch.to(device)

            params_pred, mu, logvar = model(obs_batch)
            total, recon, kl = model.loss(
                params_pred, params_batch, mu, logvar,
                beta=beta, free_bits=config.free_bits,
                param_weights=pw,
            )

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss.append(total.item())
            epoch_recon.append(recon.item())
            epoch_kl.append(kl.item())

        # ── Validate ──
        model.eval()
        val_losses = []
        with torch.no_grad():
            for obs_batch, params_batch in val_loader:
                obs_batch = obs_batch.to(device)
                params_batch = params_batch.to(device)
                params_pred, mu, logvar = model(obs_batch)
                total, _, _ = model.loss(
                    params_pred, params_batch, mu, logvar, beta=beta,
                )
                val_losses.append(total.item())

        t_loss = np.mean(epoch_loss)
        v_loss = np.mean(val_losses)
        if config.cosine_annealing:
            scheduler.step(epoch)
        else:
            scheduler.step(v_loss)

        history.train_loss.append(t_loss)
        history.val_loss.append(v_loss)
        history.recon_loss.append(np.mean(epoch_recon))
        history.kl_loss.append(np.mean(epoch_kl))

        if v_loss < history.best_val_loss:
            history.best_val_loss = v_loss
            history.best_epoch = epoch
            torch.save({
                "model_state": model.state_dict(),
                "obs_mean": obs_mean,
                "obs_std": obs_std,
                "param_min": param_min,
                "param_max": param_max,
                "normalize_params": config.normalize_params,
            }, save_dir / "best_inverter.pt")

        if (epoch + 1) % 30 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:3d}/{config.n_epochs} | "
                f"Loss: {t_loss:.6f} | Val: {v_loss:.6f} | "
                f"Recon: {np.mean(epoch_recon):.6f} | KL: {np.mean(epoch_kl):.4f} | "
                f"β: {beta:.3f}"
            )

    # Load best
    ckpt = torch.load(save_dir / "best_inverter.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"\nTraining complete. Best val loss: {history.best_val_loss:.6f} "
          f"at epoch {history.best_epoch + 1}")

    return model, history
