"""Anomaly detector for SDGFT parameter space.

Trains an autoencoder on surrogate predictions to find anomalous
parameter regions — places where the model's predictions deviate
from the learned "normal" manifold.

This can reveal:
- Parameter regions where new physics phenomena emerge
- Boundary effects of the axiom constraints
- Unexplored regions requiring more data

Architecture
-----------
Autoencoder on the residual vector (predicted - exact) across
all observables. High reconstruction error → anomalous region.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class ResidualAutoencoder(nn.Module):
    """Autoencoder for anomaly detection on prediction residuals.

    Parameters
    ----------
    n_features : int
        Dimension of the residual vector (= number of observables).
    hidden_dim : int
        Width of hidden layers.
    bottleneck_dim : int
        Dimension of the bottleneck (compressed representation).
    n_hidden : int
        Number of hidden layers in encoder/decoder.
    """

    def __init__(
        self,
        n_features: int = 36,
        hidden_dim: int = 64,
        bottleneck_dim: int = 8,
        n_hidden: int = 2,
    ):
        super().__init__()

        # Encoder
        enc_layers: list[nn.Module] = [nn.Linear(n_features, hidden_dim), nn.SiLU()]
        for _ in range(n_hidden - 1):
            enc_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        enc_layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers: list[nn.Module] = [nn.Linear(bottleneck_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_hidden - 1):
            dec_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        dec_layers.append(nn.Linear(hidden_dim, n_features))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        reconstruction, bottleneck_embedding
        """
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute anomaly score = reconstruction MSE per sample."""
        recon, _ = self.forward(x)
        return ((x - recon) ** 2).mean(dim=-1)


@dataclass
class AnomalyResult:
    """Result of anomaly detection scan."""
    params: np.ndarray  # (N, 3) — parameter vectors
    scores: np.ndarray  # (N,) — anomaly scores
    threshold: float    # score threshold for "anomalous"
    anomaly_mask: np.ndarray  # (N,) bool


class AnomalyDetector:
    """Detect anomalous parameter regions in SDGFT space.

    Workflow:
    1. Generate or load residuals (surrogate_pred - exact) over parameter space
    2. Train autoencoder on residuals from "normal" regions
    3. Score all regions: high reconstruction error = anomalous

    Parameters
    ----------
    n_features : int
        Number of observables.
    hidden_dim : int
        Autoencoder hidden width.
    bottleneck_dim : int
        Bottleneck dimension.
    """

    def __init__(
        self,
        n_features: int = 36,
        hidden_dim: int = 64,
        bottleneck_dim: int = 8,
        device: str = "cpu",
    ):
        self.device = device
        self.model = ResidualAutoencoder(
            n_features=n_features,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
        ).to(device)

    def fit(
        self,
        residuals: np.ndarray,
        n_epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
    ) -> list[float]:
        """Train the autoencoder on residual data.

        Parameters
        ----------
        residuals : (N, n_features) — residual vectors

        Returns
        -------
        loss_history : list of epoch-mean losses
        """
        data = torch.from_numpy(residuals.astype(np.float32)).to(self.device)
        dataset = torch.utils.data.TensorDataset(data)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        loss_history = []

        for epoch in range(n_epochs):
            self.model.train()
            epoch_loss = []
            for (batch,) in loader:
                recon, _ = self.model(batch)
                loss = loss_fn(recon, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
            loss_history.append(np.mean(epoch_loss))

        return loss_history

    def detect(
        self,
        params: np.ndarray,
        residuals: np.ndarray,
        quantile: float = 0.95,
    ) -> AnomalyResult:
        """Score parameter regions and detect anomalies.

        Parameters
        ----------
        params : (N, 3) — parameter vectors
        residuals : (N, n_features) — residual vectors
        quantile : float
            Fraction of data considered "normal" (threshold).

        Returns
        -------
        AnomalyResult with scores and anomaly flags.
        """
        self.model.eval()
        data = torch.from_numpy(residuals.astype(np.float32)).to(self.device)

        with torch.no_grad():
            scores = self.model.anomaly_score(data).cpu().numpy()

        threshold = float(np.quantile(scores, quantile))
        anomaly_mask = scores > threshold

        return AnomalyResult(
            params=params,
            scores=scores,
            threshold=threshold,
            anomaly_mask=anomaly_mask,
        )

    def compute_residuals(
        self,
        surrogate_model: Any,
        params: np.ndarray,
        edge_index: np.ndarray,
    ) -> np.ndarray:
        """Compute residuals: surrogate prediction - exact SDGFT.

        Parameters
        ----------
        surrogate_model : SurrogateGNN
        params : (N, 3) — parameter vectors
        edge_index : (2, E)

        Returns
        -------
        residuals : (N, n_features)
        """
        from ..data.parameter_sweep import ParametricForward
        from ..data.dag_builder import observable_names
        import math

        names = observable_names()
        ei = torch.from_numpy(edge_index).to(self.device)
        surrogate_model = surrogate_model.to(self.device)
        surrogate_model.eval()

        residuals = []
        for row in params:
            # Exact
            fwd = ParametricForward(
                delta=float(row[0]),
                delta_g=float(row[1]),
                phi=float(row[2]),
            )
            try:
                exact = fwd.compute_all()
                exact_vec = np.array([exact.get(n, 0.0) for n in names], dtype=np.float32)
                if any(math.isnan(v) for v in exact_vec):
                    continue
            except (ValueError, ZeroDivisionError):
                continue

            # Predicted
            with torch.no_grad():
                p = torch.tensor(row, dtype=torch.float32).unsqueeze(0).to(self.device)
                pred_vec = surrogate_model(p, ei).cpu().numpy()

            residuals.append(pred_vec - exact_vec)

        return np.array(residuals, dtype=np.float32)
