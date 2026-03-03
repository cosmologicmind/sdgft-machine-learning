"""Tests for the data pipeline: registry export, parameter sweep, DAG builder."""

from __future__ import annotations

import math

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════
# Test ParametricForward (parameter_sweep.py)
# ═══════════════════════════════════════════════════════════════════

class TestParametricForward:
    """Test the parametric forward model at the SDGFT axiom point."""

    def setup_method(self):
        from sdgft_ml.data.parameter_sweep import ParametricForward
        self.fwd = ParametricForward()  # default: Δ=5/24, δ=1/24

    # ── Level 2: Dimension ────────────────────────────────────

    def test_d_star_tree(self):
        expected = 67.0 / 24.0
        assert abs(self.fwd.d_star_tree - expected) < 1e-10

    def test_d_star_fp_converges(self):
        d_fp = self.fwd.d_star_fp
        assert 2.79 < d_fp < 2.80
        # Self-consistency: calling again returns cached value
        assert d_fp == self.fwd.d_star_fp

    def test_n_tree(self):
        assert abs(self.fwd.n_tree - 67.0 / 48.0) < 1e-10

    def test_n_fp(self):
        assert abs(self.fwd.n_fp - self.fwd.d_star_fp / 2.0) < 1e-15

    # ── Level 3: Gravity ──────────────────────────────────────

    def test_alpha_m(self):
        n = self.fwd.n_tree
        expected = (n - 1) / (2 * n - 1)
        assert abs(self.fwd.alpha_m() - expected) < 1e-10

    def test_alpha_b_is_half_alpha_m(self):
        assert abs(self.fwd.alpha_b() + self.fwd.alpha_m() / 2) < 1e-10

    def test_grav_slip_limits(self):
        # GR limit (n=1): η → 1
        assert abs(self.fwd.grav_slip(n=1.0, k_over_aH=10.0) - 1.0) < 1e-10
        # Deep sub-horizon (x→∞): η → 1/2
        eta_deep = self.fwd.grav_slip(k_over_aH=1e10)
        assert abs(eta_deep - 0.5) < 0.01

    # ── Level 4: Inflation ────────────────────────────────────

    def test_e_folds_reasonable(self):
        n_e = self.fwd.e_folds()
        assert 50 < n_e < 70, f"N_e = {n_e} outside [50, 70]"

    def test_spectral_index(self):
        n_s = self.fwd.spectral_index()
        assert 0.95 < n_s < 0.98, f"n_s = {n_s} outside [0.95, 0.98]"

    def test_tensor_to_scalar(self):
        r = self.fwd.tensor_to_scalar()
        assert 0.001 < r < 0.1, f"r = {r} outside [0.001, 0.1]"

    def test_beta_iso(self):
        assert abs(self.fwd.beta_iso - 1.0 / 36.0) < 1e-10

    # ── Level 5-6: Cosmology ──────────────────────────────────

    def test_omega_b(self):
        expected = (5.0 / 24.0 / 4.0) * (1.0 - 1.0 / 24.0)
        assert abs(self.fwd.omega_b - expected) < 1e-12

    def test_omega_c(self):
        expected = 6.0 * (5.0 / 24.0) ** 2
        assert abs(self.fwd.omega_c - expected) < 1e-12

    def test_flatness(self):
        total = self.fwd.omega_b + self.fwd.omega_c + self.fwd.omega_de
        assert abs(total - 1.0) < 1e-12

    def test_w_de(self):
        w = self.fwd.w_de()
        assert -1.0 < w < -0.9, f"w_DE = {w} outside (-1, -0.9)"

    def test_eta_b(self):
        eta = self.fwd.eta_b
        assert 5e-10 < eta < 8e-10, f"eta_B = {eta}"

    # ── Level 5-6: Particle Physics ───────────────────────────

    def test_alpha_em_inv(self):
        a_inv = self.fwd.alpha_em_inv()
        assert 130 < a_inv < 140, f"α_em⁻¹ = {a_inv}"

    def test_alpha_s(self):
        expected = math.sqrt(2) / 12
        assert abs(self.fwd.alpha_s - expected) < 1e-10

    def test_sin2_theta_w(self):
        expected = 1 / 9 + 0.12011
        assert abs(self.fwd.sin2_theta_w - expected) < 1e-10

    def test_higgs_mass(self):
        m_h = self.fwd.higgs_mass
        assert 120 < m_h < 130, f"m_H = {m_h} outside [120, 130]"

    def test_n_generations(self):
        assert self.fwd.n_generations == 3

    def test_theta_12(self):
        t12 = self.fwd.theta_12()
        assert 30 < t12 < 37, f"θ₁₂ = {t12}"

    def test_theta_23(self):
        t23 = self.fwd.theta_23()
        assert 44 < t23 < 50, f"θ₂₃ = {t23}"

    def test_theta_13(self):
        t13 = self.fwd.theta_13()
        assert 7 < t13 < 10, f"θ₁₃ = {t13}"

    def test_v_us(self):
        v = self.fwd.v_us()
        assert 0.2 < v < 0.25, f"|V_us| = {v}"

    def test_v_ub(self):
        v = self.fwd.v_ub()
        assert 0.001 < v < 0.01, f"|V_ub| = {v}"

    # ── Compute all ───────────────────────────────────────────

    def test_compute_all_keys(self):
        result = self.fwd.compute_all()
        assert "d_star_tree" in result
        assert "n_s" in result
        assert "higgs_mass" in result
        assert "param_delta" in result

    def test_compute_all_no_nan(self):
        result = self.fwd.compute_all()
        for key, val in result.items():
            assert not math.isnan(val), f"NaN in {key}"

    def test_feature_vector_shape(self):
        vec = self.fwd.feature_vector()
        from sdgft_ml.data.parameter_sweep import ParametricForward
        assert vec.shape == (len(ParametricForward.OBSERVABLE_KEYS),)
        assert vec.dtype == np.float64

    def test_param_vector_shape(self):
        vec = self.fwd.param_vector()
        assert vec.shape == (3,)


class TestParametricForwardEdgeCases:
    """Test edge cases and extreme parameters."""

    def test_very_small_delta(self):
        from sdgft_ml.data.parameter_sweep import ParametricForward
        fwd = ParametricForward(delta=0.01, delta_g=0.01)
        result = fwd.compute_all()
        # Should not crash
        assert "d_star_tree" in result

    def test_large_delta(self):
        from sdgft_ml.data.parameter_sweep import ParametricForward
        fwd = ParametricForward(delta=0.4, delta_g=0.05)
        result = fwd.compute_all()
        assert "d_star_tree" in result

    def test_axiom_constraint(self):
        from sdgft_ml.data.parameter_sweep import ParametricForward
        fwd = ParametricForward()
        assert abs(fwd.axiom_sum - 0.25) < 1e-10


# ═══════════════════════════════════════════════════════════════════
# Test Sweep Functions
# ═══════════════════════════════════════════════════════════════════

class TestSweep:
    """Test grid and LHS sweep functions."""

    def test_sweep_grid_basic(self):
        from sdgft_ml.data.parameter_sweep import sweep_grid
        samples = sweep_grid(n_delta=5, n_delta_g=5)
        assert len(samples) > 0
        assert "param_delta" in samples[0]
        assert "n_s" in samples[0]

    def test_sweep_constrained(self):
        from sdgft_ml.data.parameter_sweep import sweep_constrained
        df = sweep_constrained(n_points=20)
        assert len(df) > 0
        # Check axiom constraint: delta + delta_g ≈ 0.25
        sums = df["param_delta"] + df["param_delta_g"]
        assert all(abs(s - 0.25) < 1e-10 for s in sums)

    def test_sweep_lhs_shape(self):
        from sdgft_ml.data.parameter_sweep import sweep_latin_hypercube
        df = sweep_latin_hypercube(n_samples=50, seed=42)
        assert len(df) > 0
        from sdgft_ml.data.parameter_sweep import ParametricForward
        expected_cols = ParametricForward.PARAM_KEYS + ParametricForward.OBSERVABLE_KEYS
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_sweep_lhs_reproducible(self):
        from sdgft_ml.data.parameter_sweep import sweep_latin_hypercube
        df1 = sweep_latin_hypercube(n_samples=20, seed=123)
        df2 = sweep_latin_hypercube(n_samples=20, seed=123)
        np.testing.assert_array_equal(df1.values, df2.values)


# ═══════════════════════════════════════════════════════════════════
# Test DAG Builder
# ═══════════════════════════════════════════════════════════════════

class TestDAGBuilder:
    """Test the DAG construction for PyG."""

    def test_build_dag(self):
        from sdgft_ml.data.dag_builder import build_dag
        adj, names = build_dag()
        assert len(names) > 30
        assert "n_s" in names
        assert "d_star_tree" in names

    def test_edge_index_shape(self):
        from sdgft_ml.data.dag_builder import build_dag, build_edge_index
        adj, names = build_dag()
        ei = build_edge_index(adj, names)
        assert ei.shape[0] == 2
        assert ei.shape[1] > 0

    def test_edge_index_valid(self):
        from sdgft_ml.data.dag_builder import build_dag, build_edge_index
        adj, names = build_dag()
        ei = build_edge_index(adj, names)
        n_nodes = len(names)
        assert (ei >= 0).all()
        assert (ei < n_nodes).all()

    def test_node_features(self):
        from sdgft_ml.data.dag_builder import node_features_from_dict, observable_names
        from sdgft_ml.data.parameter_sweep import ParametricForward
        fwd = ParametricForward()
        values = fwd.compute_all()
        names = observable_names()
        feats = node_features_from_dict(values, names)
        assert feats.shape == (len(names), 3)
        assert feats.dtype == np.float32

    def test_dependency_map_consistency(self):
        """All dependencies should reference valid observable names."""
        from sdgft_ml.data.dag_builder import _DEPENDENCY_MAP, observable_names
        valid = set(observable_names())
        for node, deps in _DEPENDENCY_MAP.items():
            if node in valid:
                for dep in deps:
                    assert dep in valid, f"{node} depends on unknown {dep}"

    def test_dag_is_acyclic(self):
        """The DAG should have no cycles."""
        from sdgft_ml.data.dag_builder import build_dag
        adj, names = build_dag()

        # Topological sort via DFS
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in names}
        has_cycle = [False]

        def dfs(node):
            color[node] = GRAY
            for dep in adj.get(node, []):
                if color[dep] == GRAY:
                    has_cycle[0] = True
                    return
                if color[dep] == WHITE:
                    dfs(dep)
            color[node] = BLACK

        for n in names:
            if color[n] == WHITE:
                dfs(n)

        assert not has_cycle[0], "DAG contains a cycle!"


# ═══════════════════════════════════════════════════════════════════
# Test PyG Integration (requires torch + torch_geometric)
# ═══════════════════════════════════════════════════════════════════

_torch_available = True
try:
    import torch
except ImportError:
    _torch_available = False

_pyg_available = True
try:
    from torch_geometric.data import Data
except ImportError:
    _pyg_available = False


@pytest.mark.skipif(not _torch_available, reason="PyTorch not installed")
@pytest.mark.skipif(not _pyg_available, reason="torch_geometric not installed")
class TestPyGIntegration:
    """Test PyG Data construction."""

    def test_dag_to_pyg(self):
        from sdgft_ml.data.dag_builder import dag_to_pyg
        from sdgft_ml.data.parameter_sweep import ParametricForward
        fwd = ParametricForward()
        values = fwd.compute_all()
        data = dag_to_pyg(values)
        assert hasattr(data, "x")
        assert hasattr(data, "edge_index")
        assert hasattr(data, "y")
        assert hasattr(data, "params")
        assert data.x.dim() == 2
        assert data.edge_index.shape[0] == 2

    def test_sweep_to_pyg_list(self):
        from sdgft_ml.data.dag_builder import sweep_to_pyg_list
        from sdgft_ml.data.parameter_sweep import sweep_grid
        samples = sweep_grid(n_delta=3, n_delta_g=3)
        if samples:
            dataset = sweep_to_pyg_list(samples)
            assert len(dataset) == len(samples)
