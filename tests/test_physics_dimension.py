"""Tests for sdgft_ml.physics.dimension module."""

import math
from fractions import Fraction

import pytest

from sdgft_ml.physics import constants as C
from sdgft_ml.physics import dimension as D


class TestDimensionalFlowConstants:
    """Verify D*, gamma_geo and derived constants."""

    def test_d_star_tree_exact(self):
        assert D.D_STAR_TREE == Fraction(67, 24)

    def test_d_star_tree_float(self):
        assert D.D_STAR_TREE_F == pytest.approx(67 / 24, rel=1e-12)

    def test_d_star_between_2_and_4(self):
        assert 2.0 < D.D_STAR_TREE_F < 4.0

    def test_n_tree_exact(self):
        assert D.N_TREE == Fraction(67, 48)

    def test_gamma_geo_tree_exact(self):
        expected = Fraction(1, 24)**2 / Fraction(67, 24)
        assert D.GAMMA_GEO_TREE == expected

    def test_gamma_geo_tree_value(self):
        assert D.GAMMA_GEO_TREE_F == pytest.approx(1 / 1608, rel=1e-6)

    def test_gamma_geo_sq_small(self):
        assert D.GAMMA_GEO_TREE_SQ_F < 1e-5

    def test_alpha_m_tree_exact(self):
        assert D.ALPHA_M_TREE == Fraction(19, 86)

    def test_alpha_b_tree_relation(self):
        assert D.ALPHA_B_TREE == -D.ALPHA_M_TREE / 2

    def test_b_tf_tree_exact(self):
        assert D.B_TF_TREE == Fraction(91, 24)

    def test_b_tf_near_four(self):
        assert D.B_TF_TREE_F == pytest.approx(91 / 24, rel=1e-10)
        assert 3.5 < D.B_TF_TREE_F < 4.5  # close to observed ~3.8-4.0


class TestFixedPoint:
    """Test fixed-point iteration."""

    def test_compute_d_star_fp_converges(self):
        d_fp, history = D.compute_d_star_fp()
        assert len(history) > 1
        assert abs(history[-1] - history[-2]) < 1e-14

    def test_d_star_fp_value(self):
        assert D.D_STAR_FP == pytest.approx(2.797, rel=1e-2)

    def test_d_star_fp_close_to_tree(self):
        assert abs(D.D_STAR_FP - D.D_STAR_TREE_F) < 0.01


class TestAlphaEm:
    """Tree-level fine-structure constant."""

    def test_alpha_em_tree_positive(self):
        assert D.ALPHA_EM_TREE > 0

    def test_alpha_em_inv_tree_near_137(self):
        assert D.ALPHA_EM_INV_TREE == pytest.approx(137, rel=0.02)


class TestMixingAngles:
    """Neutrino mixing angles from dimension module."""

    def test_theta_12_range(self):
        val = D.theta_12()
        assert 30 < val < 40  # observed ~33.4°

    def test_theta_23_range(self):
        val = D.theta_23()
        assert 40 < val < 55  # observed ~49°

    def test_theta_13_range(self):
        val = D.theta_13()
        assert 5 < val < 12  # observed ~8.6°


class TestGalacticParameters:
    """Galaxy rotation parameters."""

    def test_transition_radius_positive(self):
        assert D.R_TRANS_KPC > 0

    def test_epsilon_gal_positive(self):
        assert D.EPSILON_GAL > 0
        assert D.EPSILON_GAL < 1
