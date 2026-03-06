"""Tests for sdgft_ml.physics.rg_running module."""

import math

import pytest

from sdgft_ml.physics import rg_running as rg


class TestBetaCoefficients:

    def test_b1_positive(self):
        assert rg.B1 > 0  # U(1) asymptotically free? No — B1>0 means α₁ grows

    def test_b2_negative(self):
        assert rg.B2 < 0  # SU(2) asymptotically free

    def test_b3_negative(self):
        assert rg.B3 < 0  # SU(3) asymptotically free


class TestCouplingsFromObservables:

    def test_returns_three(self):
        ia1, ia2, ia3 = rg.couplings_from_observables(
            rg.ALPHA_EM_INV_MZ, rg.SIN2_THETA_W_MZ, rg.ALPHA_S_MZ)
        assert ia1 > 0
        assert ia2 > 0
        assert ia3 > 0

    def test_alpha_s_at_mz(self):
        _, _, ia3 = rg.couplings_from_observables(
            rg.ALPHA_EM_INV_MZ, rg.SIN2_THETA_W_MZ, rg.ALPHA_S_MZ)
        assert 1 / ia3 == pytest.approx(rg.ALPHA_S_MZ, rel=1e-6)


class TestRunToScale:

    def test_at_mz(self):
        """t=0 should return M_Z scale values."""
        r = rg.run_to_scale(0.0)
        assert r["scale_gev"] == pytest.approx(rg.M_Z, rel=1e-6)
        assert r["alpha_s"] == pytest.approx(rg.ALPHA_S_MZ, rel=1e-4)

    def test_at_higher_scale(self):
        r = rg.run_to_scale(10.0)
        assert r["scale_gev"] > rg.M_Z
        # α_s should decrease (asymptotic freedom)
        assert r["alpha_s"] < rg.ALPHA_S_MZ

    def test_sin2_theta_w_present(self):
        r = rg.run_to_scale(5.0)
        assert "sin2_theta_w" in r
        assert 0 < r["sin2_theta_w"] < 1


class TestUnification:

    def test_find_unification_scale(self):
        t_gut, m_gut = rg.find_unification_scale()
        assert t_gut > 0
        assert m_gut > 1e10  # well above TeV

    def test_gut_scale_order(self):
        _, m_gut = rg.find_unification_scale()
        assert 1e12 < m_gut < 1e18


class TestTrajectory:

    def test_trajectory_length(self):
        traj = rg.rg_trajectory(n_points=50)
        assert len(traj) == 51  # 0..50 inclusive

    def test_trajectory_monotone_alpha_s(self):
        traj = rg.rg_trajectory(n_points=50)
        alpha_s_vals = [t["alpha_s"] for t in traj]
        # α_s should decrease monotonically (asymptotic freedom)
        for i in range(1, len(alpha_s_vals)):
            assert alpha_s_vals[i] <= alpha_s_vals[i - 1] + 1e-10


class TestModuleLevelValues:

    def test_sin2_theta_w_sm_planck(self):
        assert 0 < rg.SIN2_THETA_W_SM_PLANCK < 1

    def test_gamma_ew(self):
        expected = rg.SIN2_THETA_W_MZ - 1 / 9
        assert rg.GAMMA_EW_ARITHMETIC == pytest.approx(expected, rel=1e-6)

    def test_t_gut_positive(self):
        assert rg.T_GUT > 0

    def test_m_gut_gev(self):
        assert rg.M_GUT > 1e10
