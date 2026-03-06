"""Tests for sdgft_ml.physics.atomic and sdgft_ml.physics.qed modules."""

import math

import pytest

from sdgft_ml.physics import atomic, qed
from sdgft_ml.physics import constants as C
from sdgft_ml.physics import dimension as D


# ── Lamb shift ────────────────────────────────────────────────────

class TestLambShift:

    def test_tree_within_1pct(self):
        obs = atomic.LAMB_SHIFT_OBS_MHZ
        tree = atomic.lamb_shift_tree()
        assert abs(tree - obs) / obs < 0.01

    def test_fp_within_1pct(self):
        obs = atomic.LAMB_SHIFT_OBS_MHZ
        fp = atomic.lamb_shift_fp()
        assert abs(fp - obs) / obs < 0.01

    def test_lamb_shift_geo_formula(self):
        L = atomic.lamb_shift_geo(D.GAMMA_GEO_TREE_SQ_F)
        assert L > 0

    def test_d_star_from_lamb_invertible(self):
        d = atomic.d_star_from_lamb_shift(atomic.LAMB_SHIFT_OBS_MHZ)
        assert 2.5 < d < 3.0

    def test_rydberg_correction_small(self):
        corr = atomic.rydberg_geo_correction()
        assert abs(corr) < 0.01


# ── g − 2 ─────────────────────────────────────────────────────────

class TestG2:

    def test_electron_correction_zero(self):
        assert qed.delta_a_electron() == 0.0

    def test_muon_correction_positive(self):
        da = qed.delta_a_muon()
        assert da > 0

    def test_tau_correction_larger_than_muon(self):
        assert qed.delta_a_tau() > qed.delta_a_muon()

    def test_muon_correction_order_of_magnitude(self):
        da = qed.delta_a_muon()
        assert 1e-12 < da < 1e-8

    def test_predict_electron_dataclass(self):
        p = qed.predict_electron()
        assert p.lepton == "e"
        assert p.delta_a_geo == 0.0
        assert p.mass_ratio == pytest.approx(1.0)

    def test_predict_muon_dataclass(self):
        p = qed.predict_muon()
        assert p.lepton == "μ"
        assert p.delta_a_geo > 0
        assert p.a_sm is not None
        assert p.a_exp is not None

    def test_predict_muon_sigma(self):
        p = qed.predict_muon()
        # SDGFT should reduce the SM-vs-experiment tension
        assert p.sigma_vs_exp is not None

    def test_predict_tau_dataclass(self):
        p = qed.predict_tau()
        assert p.lepton == "τ"
        assert p.delta_a_geo > 0


class TestXiD:
    """Dimensional Schwinger factor."""

    def test_xi_4_is_one(self):
        assert qed.xi_d(4.0) == pytest.approx(1.0, rel=0.01)

    def test_xi_d_star(self):
        val = qed.xi_d(D.D_STAR_TREE_F)
        assert val > 0
