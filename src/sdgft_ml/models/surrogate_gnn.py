"""GATv2-based Graph Neural Network surrogate for SDGFT observables.

The GNN mirrors the SDGFT computation DAG:
- Each node = one observable
- Edges = dependency relationships (parent → child)
- Input: axiom parameters (Δ, δ_g, φ) injected into root nodes
- Output: predicted value for every observable

Architecture
-----------
1. Parameter encoder: MLP maps (Δ, δ_g, φ) → per-node initial embedding
2. GATv2Conv layers: message-passing along the DAG edges
3. Node decoder: MLP maps final node embedding → scalar prediction

This is a *surrogate model* — once trained, it replaces the SDGFT
computation chain with a differentiable, GPU-accelerated approximation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import GATv2Conv
except ImportError:
    GATv2Conv = None


class ParameterEncoder(nn.Module):
    """Encode input parameters into per-node initial features.

    Maps the 3-D parameter vector (Δ, δ_g, φ) to a hidden dim
    embedding for each node in the DAG.
    """

    def __init__(
        self,
        n_params: int = 3,
        n_nodes: int = 36,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        # Shared MLP: params → per-node embedding
        self.mlp = nn.Sequential(
            nn.Linear(n_params, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_nodes * hidden_dim),
        )
        self.hidden_dim = hidden_dim

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        params : (B, n_params) or (n_params,)

        Returns
        -------
        (B * n_nodes, hidden_dim)  — ready for batched graph input
        """
        if params.dim() == 1:
            params = params.unsqueeze(0)
        B = params.size(0)
        h = self.mlp(params)  # (B, n_nodes * hidden_dim)
        h = h.view(B * self.n_nodes, self.hidden_dim)
        return h


class NodeDecoder(nn.Module):
    """Decode node embeddings to scalar observable predictions."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """(N, hidden_dim) → (N, 1)"""
        return self.mlp(h)


class SurrogateGNN(nn.Module):
    """GATv2-based surrogate for the full SDGFT computation DAG.

    Parameters
    ----------
    n_params : int
        Number of input parameters (default 3: Δ, δ_g, φ).
    n_nodes : int
        Number of observable nodes in the DAG (default 36).
    hidden_dim : int
        Hidden dimension for node embeddings.
    n_heads : int
        Number of attention heads in GATv2Conv.
    n_layers : int
        Number of message-passing layers.
    dropout : float
        Dropout rate for attention coefficients.
    """

    def __init__(
        self,
        n_params: int = 3,
        n_nodes: int = 36,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if GATv2Conv is None:
            raise ImportError(
                "torch_geometric is required for SurrogateGNN. "
                "Install with: pip install torch-geometric"
            )

        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim

        # 1. Parameter encoder
        self.encoder = ParameterEncoder(n_params, n_nodes, hidden_dim)

        # 2. Graph attention layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(
                GATv2Conv(
                    hidden_dim,
                    hidden_dim // n_heads,
                    heads=n_heads,
                    dropout=dropout,
                    add_self_loops=True,
                    concat=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        # 3. Node decoder
        self.decoder = NodeDecoder(hidden_dim)

        # Activation
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        params: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the surrogate.

        Parameters
        ----------
        params : (B, n_params) or (n_params,)
            Axiom parameters.
        edge_index : (2, E)
            DAG edge index (shared across batch).
        batch : (B * n_nodes,) | None
            Batch assignment vector for batched graphs.

        Returns
        -------
        predictions : (B * n_nodes,)
            Predicted value for each observable node.
        """
        # Encode parameters to node embeddings
        h = self.encoder(params)  # (B * n_nodes, hidden_dim)

        # Message passing along DAG
        for conv, norm in zip(self.convs, self.norms):
            h_res = h
            h = conv(h, edge_index)
            h = norm(h)
            h = self.act(h)
            h = self.dropout(h)
            h = h + h_res  # residual connection

        # Decode to scalar predictions
        pred = self.decoder(h).squeeze(-1)  # (B * n_nodes,)
        return pred

    def predict(
        self,
        delta: float,
        delta_g: float,
        phi: float,
        edge_index: torch.Tensor,
    ) -> dict[str, float]:
        """Convenience: predict all observables for a single parameter set.

        Returns
        -------
        dict mapping observable names → predicted values
        """
        from ..data.dag_builder import observable_names

        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            params = torch.tensor(
                [delta, delta_g, phi], dtype=torch.float32
            ).unsqueeze(0).to(device)
            pred = self.forward(params, edge_index)
            names = observable_names()
            return {n: pred[i].item() for i, n in enumerate(names)}


class SurrogateGNNWithUncertainty(SurrogateGNN):
    """Surrogate GNN with MC-Dropout uncertainty estimation.

    At inference, run multiple forward passes with dropout enabled
    to estimate epistemic uncertainty (mean ± std).
    """

    def predict_with_uncertainty(
        self,
        delta: float,
        delta_g: float,
        phi: float,
        edge_index: torch.Tensor,
        n_samples: int = 50,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Predict with uncertainty via MC-Dropout.

        Returns
        -------
        (means, stds) : tuple of dicts
            Mean predictions and standard deviations per observable.
        """
        from ..data.dag_builder import observable_names

        self.train()  # enable dropout
        device = next(self.parameters()).device
        params = torch.tensor(
            [delta, delta_g, phi], dtype=torch.float32
        ).unsqueeze(0).to(device)

        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(params, edge_index)
                preds.append(pred.cpu().numpy())

        import numpy as np
        preds = np.stack(preds, axis=0)  # (n_samples, n_nodes)
        means = preds.mean(axis=0)
        stds = preds.std(axis=0)

        names = observable_names()
        return (
            {n: means[i] for i, n in enumerate(names)},
            {n: stds[i] for i, n in enumerate(names)},
        )
