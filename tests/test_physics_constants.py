"""Tests for sdgft_ml.physics.constants module."""

import math
from fractions import Fraction

import pytest

from sdgft_ml.physics import constants as C


class TestFundamentalAxioms:
    """Verify exact-fraction axiom identities."""

    def test_delta_value(self):
        assert C.DELTA == Fraction(5, 24)

    def test_delta_g_value(self):
        assert C.DELTA_G == Fraction(1, 24)

    def test_delta_plus_delta_g_equals_sin2_30(self):
        assert C.DELTA + C.DELTA_G == Fraction(1, 4)

    def test_delta_ratio(self):
        assert C.DELTA / C.DELTA_G == 5

    def test_golden_ratio_identity(self):
        assert abs(C.PHI - (1 + math.sqrt(5)) / 2) < 1e-15

    def test_golden_ratio_from_delta(self):
        r = float(C.DELTA / C.DELTA_G)
        assert abs(C.PHI - (1 + math.sqrt(r)) / 2) < 1e-15


class TestPhysicalConstants:
    """Sanity-check physical constants against PDG / CODATA values."""

    def test_speed_of_light(self):
        assert C.C == pytest.approx(2.99792458e8, rel=1e-9)

    def test_gravitational_constant(self):
        assert C.G_N == pytest.approx(6.67430e-11, rel=1e-4)

    def test_planck_constant(self):
        assert C.HBAR == pytest.approx(1.054571817e-34, rel=1e-9)

    def test_planck_mass_positive(self):
        assert C.M_P > 0
        assert C.M_P == pytest.approx(math.sqrt(C.HBAR * C.C / C.G_N), rel=1e-10)

    def test_planck_length_positive(self):
        assert C.R_P > 0
        assert C.R_P == pytest.approx(math.sqrt(C.HBAR * C.G_N / C.C**3), rel=1e-10)

    def test_planck_energy(self):
        assert C.E_P == pytest.approx(C.M_P * C.C**2, rel=1e-10)

    def test_alpha_obs(self):
        assert C.ALPHA_OBS == pytest.approx(1 / 137.036, rel=1e-4)

    def test_alpha_inv_obs(self):
        assert C.ALPHA_INV_OBS == pytest.approx(137.036, rel=1e-4)

    def test_electron_mass(self):
        assert C.M_E_GEV == pytest.approx(0.000511, rel=1e-2)

    def test_muon_electron_ratio(self):
        assert C.M_MU_OVER_M_E == pytest.approx(206.768, rel=1e-4)


class TestConversions:
    """Verify unit conversion factors."""

    def test_kpc_to_m(self):
        assert C.KPC_M == pytest.approx(3.0857e19, rel=1e-3)

    def test_solar_mass(self):
        assert C.M_SUN == pytest.approx(1.989e30, rel=1e-3)

    def test_planck_mass_gev(self):
        assert C.M_PL_GEV == pytest.approx(1.22e19, rel=1e-2)
