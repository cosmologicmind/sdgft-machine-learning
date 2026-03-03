"""DeepONet for learning scale-dependent SDGFT functions.

Learns continuous operator mappings like:
- D*(r; Δ, δ_g)  — effective dimension as function of scale
- Ω_DE(r; D*, δ_g) — dark energy running
- η(k/aH; n)      — gravitational slip

Architecture (DeepONet)
-----------------------
Branch net:  encodes the "function parameters" (Δ, δ_g, φ, ...)
Trunk net:   encodes the "query point" (r, or k/aH)
Output:      dot product of branch and trunk embeddings → scalar
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    """Simple feedforward MLP with SiLU activation."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        n_hidden: int = 3,
    ):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_hidden - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CurveDeepONet(nn.Module):
    """DeepONet for learning scale-dependent SDGFT curves.

    Parameters
    ----------
    n_params : int
        Dimension of the "function parameter" input (branch).
        E.g., 3 for (Δ, δ_g, φ) or 2 for (n, k_over_aH).
    n_query : int
        Dimension of the query point (trunk). Usually 1 (r or k/aH).
    basis_dim : int
        Number of basis functions (dot-product dimension).
    hidden_dim : int
        Width of hidden layers.
    n_hidden : int
        Depth of branch and trunk networks.
    """

    def __init__(
        self,
        n_params: int = 3,
        n_query: int = 1,
        basis_dim: int = 64,
        hidden_dim: int = 128,
        n_hidden: int = 3,
    ):
        super().__init__()
        self.basis_dim = basis_dim

        # Branch network: parameters → basis coefficients
        self.branch = MLP(n_params, basis_dim, hidden_dim, n_hidden)

        # Trunk network: query point → basis values
        self.trunk = MLP(n_query, basis_dim, hidden_dim, n_hidden)

        # Learnable bias
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        params: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the learned operator.

        Parameters
        ----------
        params : (B, n_params)
            Function parameters (e.g., Δ, δ_g, φ).
        query : (B, n_query)
            Query points (e.g., scale r in meters).

        Returns
        -------
        output : (B, 1) — predicted function value
        """
        b = self.branch(params)  # (B, basis_dim)
        t = self.trunk(query)    # (B, basis_dim)
        # DeepONet: dot product + bias
        out = (b * t).sum(dim=-1, keepdim=True) + self.bias
        return out

    def forward_grid(
        self,
        params: torch.Tensor,
        query_grid: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate on a grid of query points for a single parameter set.

        Parameters
        ----------
        params : (n_params,) — single parameter vector
        query_grid : (M, n_query) — M query points

        Returns
        -------
        output : (M,) — predicted values at each query point
        """
        params_expanded = params.unsqueeze(0).expand(
            query_grid.size(0), -1
        )  # (M, n_params)
        return self.forward(params_expanded, query_grid).squeeze(-1)


class MultiCurveDeepONet(nn.Module):
    """Multiple DeepONets for different scale-dependent functions.

    Manages separate DeepONets for:
    - d_star_of_r: D*(r) effective dimension flow
    - omega_de_rg: Ω_DE(r) dark energy running
    - grav_slip:   η(k/aH) gravitational slip function
    """

    CURVES = ["d_star_of_r", "omega_de_rg", "grav_slip"]

    def __init__(
        self,
        hidden_dim: int = 128,
        basis_dim: int = 64,
        n_hidden: int = 3,
    ):
        super().__init__()
        self.nets = nn.ModuleDict()

        # D*(r): params = (Δ, δ_g), query = log10(r)
        self.nets["d_star_of_r"] = CurveDeepONet(
            n_params=2, n_query=1,
            basis_dim=basis_dim, hidden_dim=hidden_dim, n_hidden=n_hidden,
        )

        # Ω_DE(r): params = (D*, δ_g), query = log10(r)
        self.nets["omega_de_rg"] = CurveDeepONet(
            n_params=2, n_query=1,
            basis_dim=basis_dim, hidden_dim=hidden_dim, n_hidden=n_hidden,
        )

        # η(k/aH): params = (n,), query = log10(k/aH)
        self.nets["grav_slip"] = CurveDeepONet(
            n_params=1, n_query=1,
            basis_dim=basis_dim, hidden_dim=hidden_dim, n_hidden=n_hidden,
        )

    def forward(
        self,
        curve_name: str,
        params: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate a specific curve model."""
        return self.nets[curve_name](params, query)


# ── Training data generation for curves ───────────────────────────

def generate_d_star_curve_data(
    n_params: int = 200,
    n_r_points: int = 100,
    delta_range: tuple[float, float] = (0.05, 0.40),
    delta_g_range: tuple[float, float] = (0.01, 0.08),
    log_r_range: tuple[float, float] = (-35, 27),
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate training data for D*(r) curve learning.

    Returns
    -------
    params : (N, 2) — (Δ, δ_g) parameter pairs
    queries : (N, 1) — log10(r) query points
    targets : (N, 1) — D*(r) values
    """
    from ..data.parameter_sweep import ParametricForward

    rng = np.random.default_rng(seed)
    r_p = 1.616255e-35

    all_params, all_queries, all_targets = [], [], []
    log_rs = np.linspace(*log_r_range, n_r_points)

    for _ in range(n_params):
        delta = rng.uniform(*delta_range)
        delta_g = rng.uniform(*delta_g_range)
        fwd = ParametricForward(delta=delta, delta_g=delta_g)

        for log_r in log_rs:
            r = 10.0 ** log_r
            try:
                d_star = fwd.d_star_of_r(r, r_p)
                if np.isfinite(d_star) and 0 < d_star < 10:
                    all_params.append([delta, delta_g])
                    all_queries.append([log_r])
                    all_targets.append([d_star])
            except (ValueError, OverflowError):
                continue

    return (
        np.array(all_params, dtype=np.float32),
        np.array(all_queries, dtype=np.float32),
        np.array(all_targets, dtype=np.float32),
    )


def generate_grav_slip_data(
    n_params: int = 200,
    n_k_points: int = 100,
    n_range: tuple[float, float] = (1.0, 2.0),
    log_k_range: tuple[float, float] = (-2, 4),
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate training data for gravitational slip η(k/aH; n).

    Returns
    -------
    params : (N, 1) — f(R) exponent n
    queries : (N, 1) — log10(k/aH)
    targets : (N, 1) — η values
    """
    from ..data.parameter_sweep import ParametricForward

    rng = np.random.default_rng(seed)

    all_params, all_queries, all_targets = [], [], []
    log_ks = np.linspace(*log_k_range, n_k_points)

    for _ in range(n_params):
        n_val = rng.uniform(*n_range)
        fwd = ParametricForward()  # params don't matter, just using grav_slip

        for log_k in log_ks:
            k = 10.0 ** log_k
            try:
                eta = fwd.grav_slip(n=n_val, k_over_aH=k)
                if np.isfinite(eta):
                    all_params.append([n_val])
                    all_queries.append([log_k])
                    all_targets.append([eta])
            except (ValueError, OverflowError):
                continue

    return (
        np.array(all_params, dtype=np.float32),
        np.array(all_queries, dtype=np.float32),
        np.array(all_targets, dtype=np.float32),
    )
