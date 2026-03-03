"""Build the SDGFT observable DAG as a PyTorch Geometric graph.

Each node = one observable, each directed edge = a dependency link.
Node features encode the observable's predicted value, level, and type.

Usage::

    from sdgft_ml.data.dag_builder import build_dag, dag_to_pyg
    adj, names = build_dag()
    data = dag_to_pyg(feature_dict, adj, names)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

try:
    from torch_geometric.data import Data
except ImportError:
    Data = None  # graceful degradation for tests without PyG

from .parameter_sweep import ParametricForward


# ── Canonical observable ordering ─────────────────────────────────

def observable_names() -> list[str]:
    """Return the canonical list of observable keys used as DAG nodes."""
    return list(ParametricForward.OBSERVABLE_KEYS)


# ── Dependency map ────────────────────────────────────────────────

# Hand-coded from the SDGFT computation chain.
# Key = observable, Value = list of observables it depends on.
_DEPENDENCY_MAP: dict[str, list[str]] = {
    # Level 2: dimension (depends on params only — root nodes)
    "d_star_tree": [],
    "d_star_fp": [],
    "n_tree": ["d_star_tree"],
    "n_fp": ["d_star_fp"],
    # Level 3: gravity
    "alpha_m_tree": ["n_tree"],
    "alpha_b_tree": ["alpha_m_tree"],
    "eta_slip_survey": ["n_tree"],
    "eta_slip_horizon": ["n_tree"],
    # Level 4: inflation
    "n_efolds_fp": ["d_star_fp"],
    "n_s": ["n_fp", "n_efolds_fp"],
    "r_tensor": ["n_fp", "n_efolds_fp"],
    "beta_iso": [],  # geometric constant
    "epsilon_sr": ["n_fp", "n_efolds_fp"],
    "eta_sr": ["n_fp", "n_efolds_fp"],
    # Level 5-6: cosmology
    "omega_b": [],  # depends on params directly
    "omega_c": [],
    "omega_de": ["omega_b", "omega_c"],
    "omega_m": ["omega_b", "omega_c"],
    "w_de_fp": ["d_star_fp"],
    "eta_b": [],  # depends on delta_g directly
    "s_8": ["omega_m"],
    # Level 5-6: particle physics
    "alpha_em_inv_tree": ["d_star_tree"],
    "alpha_em_inv_fp": ["d_star_fp"],
    "alpha_em_tree": ["alpha_em_inv_tree"],
    "alpha_s": [],  # geometric constant
    "sin2_theta_w": [],  # external anchor
    "mu_e_ratio": ["alpha_em_tree"],
    "tau_mu_ratio_tree": ["d_star_tree"],
    "lambda_geo": [],  # depends on params directly
    "higgs_mass": ["lambda_geo"],
    "n_generations": [],  # depends on params directly
    "theta_12": [],
    "theta_23": [],
    "theta_13": [],
    "v_us": ["omega_b"],
    "v_ub": ["mu_e_ratio", "tau_mu_ratio_tree"],
    "quark_hierarchy": [],  # geometric constant
}


def build_dag() -> tuple[dict[str, list[str]], list[str]]:
    """Build the adjacency list and canonical node ordering.

    Returns
    -------
    adj : dict mapping each observable to its dependency list
    names : ordered list of observable names (= node indices)
    """
    names = observable_names()
    # Filter to only observables in our canonical set
    adj = {}
    name_set = set(names)
    for name in names:
        deps = _DEPENDENCY_MAP.get(name, [])
        adj[name] = [d for d in deps if d in name_set]
    return adj, names


def build_edge_index(
    adj: dict[str, list[str]],
    names: list[str],
) -> np.ndarray:
    """Convert adjacency list to COO edge index array (2, E).

    Edges go from dependency → dependent (information flow direction).
    """
    name_to_idx = {n: i for i, n in enumerate(names)}
    sources, targets = [], []
    for node, deps in adj.items():
        if node not in name_to_idx:
            continue
        for dep in deps:
            if dep in name_to_idx:
                sources.append(name_to_idx[dep])
                targets.append(name_to_idx[node])
    return np.array([sources, targets], dtype=np.int64)


def node_features_from_dict(
    values: dict[str, float],
    names: list[str],
) -> np.ndarray:
    """Build node feature matrix (N, F) from a compute_all() output.

    Features per node:
    - [0] predicted value (log-scaled for values > 1)
    - [1] level indicator (0-6, normalized)
    - [2] is_root (1 if no dependencies, 0 otherwise)
    """
    adj, _ = build_dag()
    n_nodes = len(names)
    features = np.zeros((n_nodes, 3), dtype=np.float32)

    # Level assignment (approximate, from the dependency depth)
    level_map = _compute_levels(adj, names)

    for i, name in enumerate(names):
        val = values.get(name, 0.0)
        # Log-scale large values, sign-preserve
        if abs(val) > 1.0:
            features[i, 0] = np.sign(val) * np.log1p(abs(val))
        else:
            features[i, 0] = val
        features[i, 1] = level_map.get(name, 0) / 6.0
        features[i, 2] = 1.0 if not adj.get(name, []) else 0.0

    return features


def _compute_levels(adj: dict[str, list[str]], names: list[str]) -> dict[str, int]:
    """Compute topological depth (level) for each node in the DAG."""
    levels: dict[str, int] = {}

    def _depth(name: str, visited: set) -> int:
        if name in levels:
            return levels[name]
        if name in visited:
            return 0  # cycle guard
        visited.add(name)
        deps = adj.get(name, [])
        if not deps:
            levels[name] = 0
            return 0
        d = 1 + max(_depth(dep, visited) for dep in deps)
        levels[name] = d
        return d

    for name in names:
        _depth(name, set())
    return levels


# ── PyG Data construction ─────────────────────────────────────────

def dag_to_pyg(
    values: dict[str, float],
    params: np.ndarray | None = None,
) -> Any:
    """Convert a single SDGFT sample to a PyG ``Data`` object.

    Parameters
    ----------
    values : dict
        Output of ``ParametricForward.compute_all()``.
    params : np.ndarray | None
        Input parameter vector [delta, delta_g, phi].
        If None, extracted from values dict.

    Returns
    -------
    torch_geometric.data.Data with:
        - x: node features (N, 3)
        - edge_index: (2, E) directed edges
        - y: raw observable values (N,)
        - params: input parameters (3,)
    """
    if Data is None:
        raise ImportError(
            "torch_geometric is required for dag_to_pyg(). "
            "Install with: pip install torch-geometric"
        )

    adj, names = build_dag()
    edge_index = build_edge_index(adj, names)
    node_feats = node_features_from_dict(values, names)
    raw_values = np.array(
        [values.get(n, 0.0) for n in names], dtype=np.float32
    )

    if params is None:
        params = np.array([
            values.get("param_delta", 5.0 / 24.0),
            values.get("param_delta_g", 1.0 / 24.0),
            values.get("param_phi", (1.0 + 5 ** 0.5) / 2.0),
        ], dtype=np.float32)

    return Data(
        x=torch.from_numpy(node_feats),
        edge_index=torch.from_numpy(edge_index),
        y=torch.from_numpy(raw_values),
        params=torch.from_numpy(params.astype(np.float32)),
    )


def sweep_to_pyg_list(
    samples: list[dict[str, float]],
) -> list[Any]:
    """Convert a list of sweep samples to a list of PyG Data objects."""
    return [dag_to_pyg(s) for s in samples]
