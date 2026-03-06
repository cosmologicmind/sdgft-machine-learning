"""High-level predictor wrapping the GNN ensemble for easy inference.

Usage::

    from sdgft_ml.inference import SDGFTPredictor

    predictor = SDGFTPredictor()                    # auto-loads ensemble
    result = predictor.predict(0.2083, 0.04167)     # single point
    df = predictor.predict_batch(param_array)        # vectorized
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sdgft_ml.data.dag_builder import (
    build_dag,
    build_edge_index,
    observable_names,
)
from sdgft_ml.models.surrogate_gnn import SurrogateGNN


# Default paths (relative to package root)
_PKG_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_CHECKPOINT_DIR = _PKG_ROOT / "checkpoints" / "ensemble"

PHI = (1.0 + math.sqrt(5.0)) / 2.0


class SDGFTPredictor:
    """High-level interface for SDGFT observable prediction.

    Loads a trained GNN ensemble from checkpoints and provides
    simple `predict()` / `predict_batch()` methods.

    Parameters
    ----------
    checkpoint_dir : str or Path, optional
        Directory containing ``member_0/`` … ``member_4/`` subdirectories,
        each with ``best_model.pt`` and ``norms.npz``.
        Defaults to ``<project>/checkpoints/ensemble/``.
    device : str
        ``"auto"`` (default), ``"cpu"``, or ``"cuda"``.
    n_members : int
        Number of ensemble members to load (default 5).
    hidden_dim : int
        GATv2 hidden dimension (must match checkpoint).
    n_heads : int
        GATv2 attention heads.
    n_layers : int
        GATv2 message-passing layers.

    Example
    -------
    >>> p = SDGFTPredictor()
    >>> result = p.predict()  # axiom point
    >>> print(f"Higgs mass: {result['higgs_mass']:.2f} GeV")
    Higgs mass: 125.63 GeV
    """

    def __init__(
        self,
        checkpoint_dir: str | Path | None = None,
        device: str = "auto",
        n_members: int = 5,
        hidden_dim: int = 128,
        n_heads: int = 8,
        n_layers: int = 6,
    ):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.checkpoint_dir = Path(checkpoint_dir or _DEFAULT_CHECKPOINT_DIR)
        self.n_members = n_members

        # Build DAG topology
        self.obs_names = observable_names()
        self.n_obs = len(self.obs_names)
        adj, dag_names = build_dag()
        edge_index_np = build_edge_index(adj, dag_names)
        self.edge_index = torch.from_numpy(edge_index_np).to(self.device)

        # Load ensemble
        self.models: list[SurrogateGNN] = []
        self.norms: list[dict[str, np.ndarray]] = []

        for i in range(n_members):
            member_dir = self.checkpoint_dir / f"member_{i}"
            model_path = member_dir / "best_model.pt"
            norms_path = member_dir / "norms.npz"

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint not found: {model_path}\n"
                    f"Ensure the ensemble checkpoints are in {self.checkpoint_dir}"
                )

            model = SurrogateGNN(
                n_params=3,
                n_nodes=self.n_obs,
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=0.1,
            ).to(self.device)
            model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )
            model.eval()
            self.models.append(model)

            norms = np.load(norms_path)
            self.norms.append({"mean": norms["mean"], "std": norms["std"]})

    def predict(
        self,
        delta: float = 5.0 / 24.0,
        delta_g: float = 1.0 / 24.0,
        phi: float = PHI,
    ) -> dict[str, float]:
        """Predict all 37 observables at a single parameter point.

        Parameters
        ----------
        delta : float
            Fibonacci-lattice conflict (default: axiom 5/24).
        delta_g : float
            Lattice tension (default: axiom 1/24).
        phi : float
            Golden ratio (default: (1+√5)/2).

        Returns
        -------
        dict mapping observable name → predicted value (ensemble mean).
        """
        result = self.predict_with_uncertainty(delta, delta_g, phi)
        return {k: v["mean"] for k, v in result.items()}

    def predict_with_uncertainty(
        self,
        delta: float = 5.0 / 24.0,
        delta_g: float = 1.0 / 24.0,
        phi: float = PHI,
    ) -> dict[str, dict[str, float]]:
        """Predict with ensemble uncertainty (mean ± std).

        Returns
        -------
        dict mapping observable name → {"mean": float, "std": float}
        """
        params = torch.tensor(
            [[delta, delta_g, phi]], dtype=torch.float32
        ).to(self.device)

        all_preds = []
        for model, norms in zip(self.models, self.norms):
            with torch.no_grad():
                raw = model(params, self.edge_index)
                raw_np = raw.cpu().numpy()
                denorm = raw_np * norms["std"] + norms["mean"]
                all_preds.append(denorm)

        stacked = np.stack(all_preds, axis=0)  # (n_members, n_obs)
        means = stacked.mean(axis=0)
        stds = stacked.std(axis=0)

        return {
            name: {"mean": float(means[i]), "std": float(stds[i])}
            for i, name in enumerate(self.obs_names)
        }

    def predict_batch(
        self,
        params: np.ndarray,
        batch_size: int = 5000,
    ) -> "pd.DataFrame":
        """Predict observables for a batch of parameter sets.

        Parameters
        ----------
        params : (N, 2) or (N, 3) array
            Columns: [delta, delta_g] or [delta, delta_g, phi].
            If 2 columns, phi defaults to the golden ratio.
        batch_size : int
            GPU batch size for inference.

        Returns
        -------
        pandas DataFrame with columns: delta, delta_g, phi, + 37 observables
        """
        import pandas as pd

        params = np.asarray(params, dtype=np.float32)
        if params.ndim == 1:
            params = params.reshape(1, -1)
        if params.shape[1] == 2:
            phi_col = np.full((len(params), 1), PHI, dtype=np.float32)
            params = np.hstack([params, phi_col])

        n_total = len(params)

        # Use all ensemble members and average (consistent with predict_with_uncertainty)
        all_member_preds = []

        for model, norms in zip(self.models, self.norms):
            member_preds = np.empty((n_total, self.n_obs), dtype=np.float32)

            for start in range(0, n_total, batch_size):
                end = min(start + batch_size, n_total)
                batch = torch.from_numpy(params[start:end]).to(self.device)

                B = batch.shape[0]
                ei = self.edge_index.clone()
                offsets = torch.arange(B, device=self.device).repeat_interleave(
                    self.edge_index.shape[1]
                ) * self.n_obs
                ei_batched = ei.repeat(1, B) + offsets.unsqueeze(0)

                with torch.no_grad():
                    raw = model(batch, ei_batched)
                    raw_np = raw.cpu().numpy().reshape(B, self.n_obs)
                    member_preds[start:end] = raw_np * norms["std"] + norms["mean"]

            all_member_preds.append(member_preds)

        # Ensemble mean across members
        all_preds = np.stack(all_member_preds, axis=0).mean(axis=0)

        df = pd.DataFrame(params, columns=["delta", "delta_g", "phi"])
        for i, name in enumerate(self.obs_names):
            df[name] = all_preds[:, i]
        return df

    @property
    def info(self) -> dict[str, Any]:
        """Return model metadata."""
        n_params = sum(p.numel() for p in self.models[0].parameters())
        return {
            "n_ensemble_members": self.n_members,
            "n_observables": self.n_obs,
            "n_model_parameters": n_params,
            "architecture": "GATv2Conv",
            "hidden_dim": 128,
            "n_heads": 8,
            "n_layers": 6,
            "device": self.device,
            "checkpoint_dir": str(self.checkpoint_dir),
        }

    def __repr__(self) -> str:
        return (
            f"SDGFTPredictor(members={self.n_members}, "
            f"obs={self.n_obs}, device='{self.device}')"
        )
