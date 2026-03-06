"""Tests for sdgft_ml.physics.collider module."""

import math

import pytest

from sdgft_ml.physics import collider
from sdgft_ml.physics import constants as C
from sdgft_ml.physics import dimension as D


class TestModifiedRunning:

    def test_at_mz_returns_dict(self):
        r = collider.sdgft_modified_running(91.2)
        assert "sqrt_s_gev" in r
        assert "alpha_s" in r
        assert "inv_alpha_1" in r

    def test_above_mz(self):
        r = collider.sdgft_modified_running(1000.0)
        assert r["sqrt_s_gev"] == pytest.approx(1000.0)
        assert r["alpha_s"] > 0

    def test_alpha_s_decreases(self):
        r1 = collider.sdgft_modified_running(100.0)
        r2 = collider.sdgft_modified_running(10000.0)
        assert r2["alpha_s"] < r1["alpha_s"]


class TestDrellYan:

    def test_ratio_near_one(self):
        r = collider.drell_yan_ratio(1000.0)
        assert abs(r - 1.0) < 0.01

    def test_ratio_decreases_with_energy(self):
        r_low = collider.drell_yan_ratio(500.0)
        r_high = collider.drell_yan_ratio(10000.0)
        # SDGFT effect grows → ratio deviates more
        assert abs(r_high - 1.0) > abs(r_low - 1.0)

    def test_module_level_constant(self):
        assert collider.DY_RATIO_3TEV == pytest.approx(
            collider.drell_yan_ratio(3000.0), rel=1e-10)


class TestGravitonExchange:

    def test_amplitude_positive(self):
        A = collider.graviton_exchange_amplitude(14000.0)
        assert A > 0

    def test_amplitude_grows_with_energy(self):
        A_low = collider.graviton_exchange_amplitude(1000.0)
        A_high = collider.graviton_exchange_amplitude(100000.0)
        assert A_high > A_low

    def test_cross_section_positive(self):
        sigma = collider.graviton_exchange_cross_section_fb(14000.0)
        assert sigma > 0

    def test_module_level_constant(self):
        assert collider.GRAV_AMP_14TEV == pytest.approx(
            collider.graviton_exchange_amplitude(14000.0), rel=1e-10)


class TestKKSpectrum:

    def test_default_10_modes(self):
        modes = collider.kk_spectrum()
        assert len(modes) == 10

    def test_custom_n_max(self):
        modes = collider.kk_spectrum(n_max=5)
        assert len(modes) == 5

    def test_masses_increasing(self):
        modes = collider.kk_spectrum()
        masses = [m.mass_gev for m in modes]
        assert masses == sorted(masses)

    def test_couplings_decreasing(self):
        modes = collider.kk_spectrum()
        couplings = [m.coupling_ratio for m in modes]
        # coupling ratios should decrease with mode number
        for i in range(1, len(couplings)):
            assert couplings[i] <= couplings[i - 1]

    def test_mode_dataclass(self):
        modes = collider.kk_spectrum(n_max=1)
        m = modes[0]
        assert m.n == 1
        assert m.mass_gev > 0
        assert 0 < m.coupling_ratio <= 1


class TestHiggs:

    def test_gg_modification_near_one(self):
        r = collider.higgs_gg_modification()
        assert abs(r - 1.0) < 1e-3

    def test_width_modification_near_one(self):
        r = collider.higgs_width_modification()
        assert abs(r - 1.0) < 1e-3

    def test_module_level_higgs(self):
        assert collider.HIGGS_GG_MOD == pytest.approx(
            collider.higgs_gg_modification(), rel=1e-10)


class TestDijet:

    def test_f_chi_near_one(self):
        f = collider.dijet_f_chi(5.0)
        assert abs(f - 1.0) < 0.01

    def test_f_chi_positive(self):
        f = collider.dijet_f_chi(1.5)
        assert f > 0


class TestComputeReach:

    def test_returns_list(self):
        reach = collider.compute_reach()
        assert isinstance(reach, list)
        assert len(reach) > 0

    def test_reach_dataclass(self):
        reach = collider.compute_reach()[0]
        assert hasattr(reach, "name")
        assert hasattr(reach, "m_reach_tev")
        assert reach.m_reach_tev > 0


class TestEnergyScan:

    def test_default_scan(self):
        scan = collider.energy_scan()
        assert len(scan) == 40  # default

    def test_keys_present(self):
        scan = collider.energy_scan()
        for s in scan:
            assert "sqrt_s_gev" in s
            assert "alpha_s" in s
            assert "drell_yan_ratio" in s
            assert "graviton_amplitude" in s

    def test_custom_energies(self):
        scan = collider.energy_scan([100.0, 1000.0, 10000.0])
        assert len(scan) == 3
