"""Training loop for the GNN surrogate model.

Generates parameter-sweep training data, constructs PyG graphs,
and trains the SurrogateGNN to reproduce all SDGFT observables.

Usage::

    from sdgft_ml.training.train_surrogate import train_surrogate
    model, history = train_surrogate(n_epochs=200, n_samples=2000)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ..data.parameter_sweep import ParametricForward, sweep_latin_hypercube
from ..data.dag_builder import build_dag, build_edge_index, observable_names


# ── Dataset ───────────────────────────────────────────────────────

class SDGFTGraphDataset(Dataset):
    """Dataset of (params, observables) pairs for GNN training."""

    def __init__(
        self,
        params: np.ndarray,
        targets: np.ndarray,
        edge_index: np.ndarray,
    ):
        """
        Parameters
        ----------
        params : (N, 3)  — input parameters per sample
        targets : (N, n_nodes) — target observable values per sample
        edge_index : (2, E) — shared DAG edge index
        """
        self.params = torch.from_numpy(params.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))
        self.edge_index = torch.from_numpy(edge_index)

    def __len__(self) -> int:
        return self.params.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.params[idx], self.targets[idx]


def _prepare_data(
    n_samples: int = 2000,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[SDGFTGraphDataset, SDGFTGraphDataset, np.ndarray]:
    """Generate sweep data and split into train/val datasets.

    Returns
    -------
    train_ds, val_ds, edge_index
    """
    print(f"Generating {n_samples} parameter sweep samples (LHS)...")
    df = sweep_latin_hypercube(n_samples=n_samples, seed=seed)

    obs_keys = ParametricForward.OBSERVABLE_KEYS
    param_keys = ParametricForward.PARAM_KEYS

    params = df[param_keys].values.astype(np.float32)
    targets = df[obs_keys].values.astype(np.float32)

    # Normalize targets: log-transform large-scale values
    # Keep a normalizer for each column
    target_means = targets.mean(axis=0)
    target_stds = targets.std(axis=0)
    target_stds[target_stds < 1e-12] = 1.0  # avoid div-by-zero
    targets_norm = (targets - target_means) / target_stds

    # Build DAG edge index
    adj, names = build_dag()
    edge_index = build_edge_index(adj, names)

    # Split
    n_val = int(len(params) * val_frac)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(params))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    train_ds = SDGFTGraphDataset(params[train_idx], targets_norm[train_idx], edge_index)
    val_ds = SDGFTGraphDataset(params[val_idx], targets_norm[val_idx], edge_index)

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, "
          f"Nodes: {len(names)}, Edges: {edge_index.shape[1]}")

    return train_ds, val_ds, edge_index


# ── Training loop ─────────────────────────────────────────────────

@dataclass
class TrainConfig:
    """Training hyperparameters."""
    n_epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    scheduler_patience: int = 15
    scheduler_factor: float = 0.5
    grad_clip: float = 1.0
    n_samples: int = 2000
    val_frac: float = 0.15
    hidden_dim: int = 64
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    seed: int = 42
    save_dir: str = "runs/surrogate"


@dataclass
class TrainHistory:
    """Training metrics over epochs."""
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    lr_history: list[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = 0


def train_surrogate(
    config: TrainConfig | None = None,
    device: str | None = None,
) -> tuple[Any, TrainHistory]:
    """Train the GNN surrogate model.

    Parameters
    ----------
    config : TrainConfig
        Training configuration. Defaults to sensible values.
    device : str
        Torch device. Auto-detects GPU if available.

    Returns
    -------
    model, history
    """
    from ..models.surrogate_gnn import SurrogateGNN

    if config is None:
        config = TrainConfig()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Training GNN surrogate on {device}")
    print(f"Config: {config}")

    # Data
    train_ds, val_ds, edge_index = _prepare_data(
        n_samples=config.n_samples,
        val_frac=config.val_frac,
        seed=config.seed,
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
    )

    n_nodes = len(observable_names())
    edge_index_t = torch.from_numpy(edge_index).to(device)

    # Model
    model = SurrogateGNN(
        n_params=3,
        n_nodes=n_nodes,
        hidden_dim=config.hidden_dim,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=config.dropout,
    ).to(device)

    n_params_total = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params_total:,}")

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=config.scheduler_patience,
        factor=config.scheduler_factor,
    )
    loss_fn = nn.MSELoss()

    # Training
    history = TrainHistory()
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    def _expand_edge_index(batch_size: int) -> torch.Tensor:
        """Expand edge index for a batch of graphs."""
        offsets = torch.arange(batch_size, device=device).unsqueeze(1) * n_nodes
        ei = edge_index_t.unsqueeze(0).expand(batch_size, -1, -1)
        return (ei + offsets.unsqueeze(1)).reshape(2, -1)

    for epoch in range(config.n_epochs):
        # ── Train ──
        model.train()
        train_losses = []
        for params, targets in train_loader:
            params, targets = params.to(device), targets.to(device)
            B = params.size(0)
            ei = _expand_edge_index(B)

            pred = model(params, ei)  # (B * n_nodes,)
            target_flat = targets.reshape(-1)
            loss = loss_fn(pred, target_flat)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            train_losses.append(loss.item())

        # ── Validate ──
        model.eval()
        val_losses = []
        with torch.no_grad():
            for params, targets in val_loader:
                params, targets = params.to(device), targets.to(device)
                B = params.size(0)
                ei = _expand_edge_index(B)
                pred = model(params, ei)
                loss = loss_fn(pred, targets.reshape(-1))
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.lr_history.append(lr)

        if val_loss < history.best_val_loss:
            history.best_val_loss = val_loss
            history.best_epoch = epoch
            torch.save(model.state_dict(), save_dir / "best_model.pt")

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:3d}/{config.n_epochs} | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"LR: {lr:.2e} | Best: {history.best_val_loss:.6f} (ep {history.best_epoch+1})"
            )

    # Load best model
    model.load_state_dict(torch.load(save_dir / "best_model.pt", weights_only=True))
    print(f"\nTraining complete. Best val loss: {history.best_val_loss:.6f} "
          f"at epoch {history.best_epoch + 1}")

    return model, history
