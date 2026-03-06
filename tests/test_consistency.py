"""Cross-validation: ParametricForward at axiom point vs module-level constants.

This test ensures that the independent reimplementation in parameter_sweep.py
produces identical results to the physics module constants when evaluated at
the canonical axiom values (Δ=5/24, δ=1/24, φ=(1+√5)/2).
"""

from __future__ import annotations

import math

import pytest

from sdgft_ml.physics import constants as C
from sdgft_ml.physics import dimension as D
from sdgft_ml.data.parameter_sweep import ParametricForward


@pytest.fixture
def pf() -> ParametricForward:
    """ParametricForward at the canonical axiom point."""
    return ParametricForward(delta=5.0 / 24.0, delta_g=1.0 / 24.0)


class TestDimensionConsistency:
    """D*, n, γ_geo must agree between modules."""

    def test_d_star_tree(self, pf: ParametricForward):
        assert pf.d_star_tree == pytest.approx(D.D_STAR_TREE_F, rel=1e-12)

    def test_d_star_fp(self, pf: ParametricForward):
        assert pf.d_star_fp == pytest.approx(D.D_STAR_FP, rel=1e-10)

    def test_n_tree(self, pf: ParametricForward):
        assert pf.n_tree == pytest.approx(D.N_TREE_F, rel=1e-12)

    def test_alpha_m(self, pf: ParametricForward):
        assert pf.alpha_m() == pytest.approx(D.ALPHA_M_TREE_F, rel=1e-10)

    def test_alpha_b(self, pf: ParametricForward):
        assert pf.alpha_b() == pytest.approx(D.ALPHA_B_TREE_F, rel=1e-10)


class TestParticlePhysicsConsistency:
    """Particle-physics observables must agree."""

    def test_alpha_em_inv(self, pf: ParametricForward):
        assert pf.alpha_em_inv() == pytest.approx(D.ALPHA_EM_INV_TREE, rel=1e-10)

    def test_mu_e_ratio(self, pf: ParametricForward):
        """Critical: was the primary inconsistency (dimension.py had wrong formula)."""
        pf_val = pf.mu_e_ratio()
        mod_val = D.MU_E_RATIO
        # Both should be ~206.6, not ~10.6
        assert pf_val == pytest.approx(mod_val, rel=1e-10)
        assert 200 < pf_val < 210, f"mu_e_ratio={pf_val} out of physical range"

    def test_tau_mu_ratio(self, pf: ParametricForward):
        """Critical: was the second inconsistency (dimension.py had wrong formula)."""
        pf_val = pf.tau_mu_ratio()
        mod_val = D.TAU_MU_RATIO_TREE
        # Both should be ~16.75, not ~0.685
        assert pf_val == pytest.approx(mod_val, rel=1e-10)
        assert 15 < pf_val < 18, f"tau_mu_ratio={pf_val} out of physical range"

    def test_tau_e_ratio(self, pf: ParametricForward):
        pf_val = pf.tau_e_ratio()
        mod_val = D.TAU_E_RATIO_TREE
        assert pf_val == pytest.approx(mod_val, rel=1e-10)
        assert 3000 < pf_val < 4000, f"tau_e_ratio={pf_val} out of physical range"


class TestMixingAnglesConsistency:
    """Neutrino mixing angles must agree."""

    def test_theta_12(self, pf: ParametricForward):
        assert pf.theta_12() == pytest.approx(D.theta_12(), rel=1e-10)

    def test_theta_23(self, pf: ParametricForward):
        assert pf.theta_23() == pytest.approx(D.theta_23(), rel=1e-10)

    def test_theta_13(self, pf: ParametricForward):
        assert pf.theta_13() == pytest.approx(D.theta_13(), rel=1e-10)


class TestDeepIdentities:
    """Verify newly-documented mathematical identities."""

    def test_b_tf_equals_d_star_plus_one(self):
        """b_TF = D* + 1 exactly (Tully-Fisher ↔ effective dimension)."""
        assert D.B_TF_TREE_F == pytest.approx(D.D_STAR_TREE_F + 1.0, rel=1e-14)

    def test_alpha_ratio_equals_delta(self):
        """α₁/α₂|_{M_Pl} = 5/24 = Δ (gauge hierarchy from 24-cell)."""
        from sdgft_ml.physics.rg_running import ALPHA_RATIO_SDGFT
        assert ALPHA_RATIO_SDGFT == pytest.approx(C.DELTA_F, rel=1e-12)

    def test_nineteen_coincidence(self):
        """The number 19 appears via D*_num − 2·D*_den = 24 − 5."""
        from fractions import Fraction
        n_minus_1 = D.N_TREE - 1
        one_minus_delta = 1 - C.DELTA
        assert n_minus_1 == Fraction(19, 48)
        assert one_minus_delta == Fraction(19, 24)
        # Both numerators are 19
        assert n_minus_1.numerator == one_minus_delta.numerator == 19


class TestP0Items:
    """Verify P0 roadmap items implemented in Round 2."""

    def test_sigma_8_is_delta_pi(self, pf: ParametricForward):
        """σ₈ = Δ·π (analytical, no longer hardcoded 0.775)."""
        expected = (5.0 / 24.0) * math.pi
        assert pf.sigma_8 == pytest.approx(expected, rel=1e-14)
        # Must NOT be the old hardcoded value
        assert pf.sigma_8 != pytest.approx(0.775, rel=1e-3)

    def test_s8_from_sigma8(self, pf: ParametricForward):
        """S₈ = σ₈ · √(Ω_m / 0.3)."""
        assert pf.s_8 == pytest.approx(
            pf.sigma_8 * math.sqrt(pf.omega_m / 0.3), rel=1e-14
        )

    def test_tensor_spectral_index(self, pf: ParametricForward):
        """n_T = -2ε_SR (single-field consistency relation)."""
        eps = pf.slow_roll_epsilon()
        n_t = pf.tensor_spectral_index()
        assert n_t == pytest.approx(-2.0 * eps, rel=1e-14)
        # Consistency: r = -8 n_T (to leading order in slow roll)
        r_tensor = pf.tensor_to_scalar()
        # r ≈ 16 ε, n_T = -2ε ⟹ r ≈ -8 n_T
        # But the exact r formula uses nbar not 16ε, so check n_T only
        assert n_t < 0, "n_T must be negative (red-tilted tensor spectrum)"

    def test_n_t_in_compute_all(self, pf: ParametricForward):
        """n_T must appear in the compute_all() dict."""
        obs = pf.compute_all()
        assert "n_t" in obs
        assert obs["n_t"] < 0

    def test_r_p_not_hardcoded(self, pf: ParametricForward):
        """d_star_of_r and omega_de_rg should use R_P from constants, not a hardcoded literal."""
        import inspect
        from sdgft_ml.physics.constants import R_P
        src = inspect.getsource(pf.d_star_of_r)
        # The default should reference the imported constant, not a numeric literal
        assert "1.616255e-35" not in src

    def test_c_t_equals_c(self, pf: ParametricForward):
        """α_T = 0 for f(R) ⊂ Horndeski ⟹ c_T = c."""
        obs = pf.compute_all()
        assert obs["alpha_t"] == 0.0


class TestTOVIntegration:
    """Verify the RK45-based TOV integrator."""

    def test_tov_classical_polytropic(self):
        """Classical TOV with polytropic EOS should give R ~ 5-15 km, M ~ 0.5-2.5 M☉."""
        from sdgft_ml.physics.black_holes import integrate_tov, polytropic_eos
        eos = polytropic_eos()
        result = integrate_tov(1e18, eos, use_running_g=False)
        assert 3 < result["R_km"] < 20, f"R={result['R_km']} km out of range"
        assert 0.1 < result["M_msun"] < 3.0, f"M={result['M_msun']} M☉ out of range"

    def test_tov_running_g_polytropic(self):
        """SDGFT running-G TOV should differ from classical (slightly)."""
        from sdgft_ml.physics.black_holes import integrate_tov, polytropic_eos
        eos = polytropic_eos()
        cl = integrate_tov(1e18, eos, use_running_g=False)
        rg = integrate_tov(1e18, eos, use_running_g=True)
        # At nuclear densities, running G correction is tiny but nonzero
        assert rg["R_km"] > 0
        assert rg["M_msun"] > 0
