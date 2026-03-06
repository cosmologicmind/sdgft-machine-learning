"""Tests for sdgft_ml.physics.black_holes module."""

import math

import pytest

from sdgft_ml.physics import black_holes as bh
from sdgft_ml.physics import constants as C


class TestRunningG:

    def test_g_running_at_zero(self):
        """G(k=0) = G_N."""
        assert bh.g_running(0.0) == pytest.approx(C.G_N, rel=1e-10)

    def test_g_running_decreases(self):
        """G(k) should decrease with increasing k."""
        g_low = bh.g_running(1e10)
        g_high = bh.g_running(1e30)
        assert g_high < g_low

    def test_g_running_at_planck_scale(self):
        """G(k_P) ≈ G_N/2 (halfway)."""
        g_kp = bh.g_running(C.K_P)
        assert g_kp == pytest.approx(C.G_N / 2, rel=0.1)

    def test_g_of_r_at_large_r(self):
        """G(r→∞) → G_N."""
        g = bh.g_of_r(1e10)  # 10 billion meters
        assert g == pytest.approx(C.G_N, rel=1e-6)

    def test_g_of_r_decreases_at_small_r(self):
        g_far = bh.g_of_r(1.0)
        g_near = bh.g_of_r(C.R_P)
        assert g_near < g_far


class TestSchwarzschildRadius:

    def test_solar_mass(self):
        rs = bh.schwarzschild_radius(C.M_SUN)
        assert rs == pytest.approx(2950, rel=0.01)  # ~2.95 km

    def test_planck_mass(self):
        rs = bh.schwarzschild_radius(C.M_P)
        assert rs > 0
        assert rs == pytest.approx(2 * C.G_N * C.M_P / C.C**2, rel=1e-10)


class TestHawkingTemperature:

    def test_solar_positive(self):
        T = bh.hawking_temperature(C.M_SUN)
        assert T > 0

    def test_solar_very_cold(self):
        T = bh.hawking_temperature(C.M_SUN)
        assert T < 1e-5  # << 1 K

    def test_small_mass_hotter(self):
        T_big = bh.hawking_temperature(10 * C.M_SUN)
        T_small = bh.hawking_temperature(C.M_SUN)
        assert T_small > T_big

    def test_max_temperature_finite(self):
        assert bh.T_HAWKING_MAX > 0
        assert math.isfinite(bh.T_HAWKING_MAX)


class TestQNMCorrection:

    def test_30_msun_small(self):
        corr = bh.qnm_correction(30 * C.M_SUN)
        assert abs(corr) < 0.01  # small correction

    def test_planck_mass_larger(self):
        corr = bh.qnm_correction(C.M_P)
        assert abs(corr) > abs(bh.qnm_correction(C.M_SUN))


class TestEntropy:

    def test_solar_positive(self):
        S = bh.bekenstein_hawking_entropy(C.M_SUN)
        assert S > 0

    def test_entropy_scales_with_mass_squared(self):
        S1 = bh.bekenstein_hawking_entropy(C.M_SUN)
        S2 = bh.bekenstein_hawking_entropy(2 * C.M_SUN)
        assert S2 == pytest.approx(4 * S1, rel=0.01)


class TestTOV:

    def test_polytropic_eos_callable(self):
        eos = bh.polytropic_eos()
        p = eos(1e15)
        assert p > 0

    def test_integrate_tov_returns_dict(self):
        eos = bh.polytropic_eos(K=5e-3, gamma=2.0)
        result = bh.integrate_tov(1e15, eos)
        assert "R_km" in result
        assert "M_msun" in result
        assert result["R_km"] > 0
        assert result["M_msun"] > 0

    def test_tov_reasonable_ns(self):
        eos = bh.polytropic_eos(K=5e-3, gamma=2.0)
        result = bh.integrate_tov(1e15, eos)
        # Compact object: positive radius and mass
        assert result["R_km"] > 0
        assert result["M_msun"] > 0
