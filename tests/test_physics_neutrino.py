"""Tests for sdgft_ml.physics.neutrino module."""

import math

import pytest

from sdgft_ml.physics import neutrino as nu


class TestMassSplitting:

    def test_ratio_tree_exact(self):
        from fractions import Fraction
        assert nu.R_TREE == Fraction(67, 2)

    def test_ratio_tree_float(self):
        assert nu.R_TREE_F == pytest.approx(33.5, rel=1e-10)

    def test_mass_splitting_ratio_matches_tree(self):
        r = nu.mass_splitting_ratio()
        assert r == pytest.approx(nu.R_TREE_F, rel=1e-10)

    def test_ratio_near_observed(self):
        r_obs = nu.RATIO_OBS
        r_tree = nu.R_TREE_F
        # should be within ~5 sigma
        assert abs(r_tree - r_obs) / nu.RATIO_OBS_UNC < 10


class TestNeutrinoMasses:

    def test_m1_zero(self):
        assert nu.M1 == 0.0

    def test_m2_positive(self):
        assert nu.M2 > 0

    def test_m3_greater_than_m2(self):
        assert nu.M3 > nu.M2

    def test_normal_ordering(self):
        m1, m2, m3 = nu.neutrino_masses()
        assert m1 <= m2 <= m3

    def test_sum_consistent(self):
        m1, m2, m3 = nu.neutrino_masses()
        assert m1 + m2 + m3 == pytest.approx(nu.SUM_M_NU, rel=1e-8)

    def test_sum_below_cosmological_bound(self):
        assert nu.SUM_M_NU < 0.12  # Planck bound


class TestMassSquaredDifferences:

    def test_dm2_21_positive(self):
        assert nu.DM2_21 > 0

    def test_dm2_31_positive(self):
        assert nu.DM2_31 > 0

    def test_dm2_21_order_of_magnitude(self):
        assert 1e-6 < nu.DM2_21 < 1e-3

    def test_dm2_31_order_of_magnitude(self):
        assert 1e-4 < nu.DM2_31 < 1e-1


class TestPMNS:

    def test_three_angles(self):
        t12, t23, t13 = nu.pmns_angles_deg()
        assert 30 < t12 < 40
        assert 40 < t23 < 55
        assert 5 < t13 < 12

    def test_delta_cp(self):
        d = nu.delta_cp_pmns()
        assert d == pytest.approx(5 * math.pi / 4, rel=1e-10)

    def test_pmns_matrix_unitary(self):
        U = nu.pmns_matrix()
        # Check |U† U|_ii ≈ 1
        for i in range(3):
            norm = sum(abs(U[k][i]) ** 2 for k in range(3))
            assert norm == pytest.approx(1.0, rel=1e-8)

    def test_jarlskog_nonzero(self):
        J = nu.jarlskog_pmns()
        assert abs(J) > 0


class TestOscillation:

    def test_survival_probability_bounded(self):
        P = nu.oscillation_probability(0, 0, L_km=295, E_GeV=0.6)
        assert 0 <= P <= 1

    def test_appearance_probability_bounded(self):
        P = nu.oscillation_probability(1, 0, L_km=1300, E_GeV=2.5)
        assert 0 <= P <= 1

    def test_unitarity_sum(self):
        """Sum over β of P(α→β) ≈ 1."""
        total = sum(
            nu.oscillation_probability(0, b, L_km=800, E_GeV=1.0)
            for b in range(3)
        )
        assert total == pytest.approx(1.0, rel=1e-6)


class TestExperimentPredictions:

    def test_dune(self):
        p = nu.predict_dune()
        assert p.name == "DUNE"
        assert p.baseline_km == pytest.approx(1300, rel=0.1)
        assert 0 <= p.probability <= 1

    def test_t2k(self):
        p = nu.predict_t2k()
        assert p.name == "T2K"
        assert 0 <= p.probability <= 1

    def test_juno(self):
        p = nu.predict_juno()
        assert p.name == "JUNO"
        assert 0 <= p.probability <= 1

    def test_nova(self):
        p = nu.predict_nova()
        assert p.name == "NOvA"
        assert 0 <= p.probability <= 1


class TestEffectiveMass:

    def test_m_bb_positive(self):
        assert nu.M_BB >= 0

    def test_m_bb_below_limit(self):
        assert nu.M_BB < nu.M_BB_OBS_LIMIT
