"""Tests for sdgft_ml.physics.galaxy module."""

import pytest

from sdgft_ml.physics import galaxy as gal
from sdgft_ml.physics import constants as C
from sdgft_ml.physics import dimension as D


class TestFreemanDisk:

    def test_v2_positive(self):
        v2 = gal.v2_freeman_disk(5.0, 1e11, 3.0)
        assert v2 > 0

    def test_v2_small_at_zero(self):
        v2 = gal.v2_freeman_disk(0.01, 1e11, 3.0)
        # should be small but positive
        assert v2 >= 0


class TestEpsilonCandidates:

    def test_candidates_nonempty(self):
        assert len(gal.EPSILON_CANDIDATES) > 0

    def test_best_epsilon_near_obs(self):
        assert abs(gal.EPSILON_BEST.value - gal.EPSILON_OBS) < gal.EPSILON_OBS_UNC

    def test_all_epsilon_positive(self):
        for c in gal.EPSILON_CANDIDATES:
            assert c.value > 0

    def test_build_epsilon_candidates(self):
        cands = gal.build_epsilon_candidates()
        assert len(cands) > 0
        # sorted by |value - obs|
        diffs = [abs(c.value - gal.EPSILON_OBS) for c in cands]
        assert diffs == sorted(diffs)


class TestGeff:

    def test_g_eff_at_small_r(self):
        g = gal.g_eff_galactic(0.1)
        assert g == pytest.approx(C.G_N, rel=0.5)

    def test_g_eff_enhanced_at_large_r(self):
        g = gal.g_eff_galactic(20.0)
        assert g > C.G_N  # modified gravity enhancement

    def test_g_eff_profile_monotone(self):
        radii = [1, 5, 10, 20, 50]
        profile = gal.g_eff_profile(radii)
        # G_eff/G_N should generally increase with radius
        assert profile[-1] >= profile[0]


class TestRotationCurve:

    def test_ngc3198_model(self):
        model = gal.NGC3198
        assert model.name == "NGC 3198"
        assert len(model.components) > 0

    def test_rotation_curve_positive(self):
        radii = [2, 5, 10, 15, 20]
        v = gal.rotation_curve(gal.NGC3198, radii)
        for vi in v:
            assert vi > 0

    def test_rotation_curve_flat_region(self):
        """Outer rotation curve should be approximately flat."""
        radii = [15, 20, 25, 30]
        v = gal.rotation_curve(gal.NGC3198, radii)
        # check flatness: max/min < 1.5
        assert max(v) / min(v) < 1.5


class TestScreening:

    def test_screening_factor_no_screen(self):
        cfg = gal.ScreeningConfig(sigma_screen=0.0, steepness=2.0)
        s = gal.screening_factor(100.0, cfg)
        # no screening → factor ≈ 1
        assert s == pytest.approx(1.0, rel=0.01)

    def test_screening_factor_bounded(self):
        cfg = gal.ScreeningConfig(sigma_screen=50.0, steepness=2.0)
        s = gal.screening_factor(1.0, cfg)
        assert 0 <= s <= 1


class TestTullyFisher:

    def test_luminosity_positive(self):
        log_L = gal.tully_fisher_luminosity(200)
        assert log_L > 0

    def test_faster_rotation_brighter(self):
        L1 = gal.tully_fisher_luminosity(100)
        L2 = gal.tully_fisher_luminosity(200)
        assert L2 > L1
