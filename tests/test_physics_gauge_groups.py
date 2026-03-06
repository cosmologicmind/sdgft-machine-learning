"""Tests for sdgft_ml.physics.gauge_groups module."""

from fractions import Fraction

import pytest

from sdgft_ml.physics import gauge_groups as gg


class Test24Cell:

    def test_vertex_count(self):
        assert gg.N_VERTICES == 24

    def test_all_norms_equal(self):
        for v in gg.VERTICES_24CELL:
            ns = sum(x * x for x in v)
            assert ns == 4  # norm² = 4 (coords scaled ×2)

    def test_isomorphism(self):
        iso = gg.verify_24cell_d4_isomorphism()
        assert iso["is_24cell"]
        assert iso["count_24"]
        assert iso["vertex_degree"] == 8


class TestD4RootSystem:

    def test_root_count(self):
        assert gg.N_D4_ROOTS == 24

    def test_cartan_matrix(self):
        expected = ((2, -1, 0, 0), (-1, 2, -1, -1),
                    (0, -1, 2, 0), (0, -1, 0, 2))
        assert gg.D4_CARTAN_MATRIX == expected

    def test_cartan_matches_expected(self):
        assert gg.D4_CARTAN_MATRIX == gg.D4_CARTAN_EXPECTED

    def test_positive_roots_half(self):
        assert len(gg.D4_POSITIVE_ROOTS) == 12  # half of 24

    def test_root_system_valid(self):
        check = gg.verify_root_system(gg.D4_ROOTS)
        assert check.get("valid", True)  # should pass


class TestDecomposition:

    def test_gauge_bosons_12(self):
        d = gg.SM_DECOMPOSITION
        assert d.n_gauge_bosons == 12

    def test_su3_dim_8(self):
        d = gg.SM_DECOMPOSITION
        assert d.a2_dim == 8

    def test_su2_dim_3(self):
        d = gg.SM_DECOMPOSITION
        assert d.a1_dim == 3

    def test_u1_dim_1(self):
        d = gg.SM_DECOMPOSITION
        assert d.u1_dim == 1

    def test_coset_16(self):
        d = gg.SM_DECOMPOSITION
        assert d.n_coset == 16

    def test_total_roots(self):
        d = gg.SM_DECOMPOSITION
        assert d.n_total_roots == 24

    def test_sm_content(self):
        assert gg.SM_CONTENT.n_gluons == 8
        assert gg.SM_CONTENT.n_weak_bosons == 3
        assert gg.SM_CONTENT.n_photon == 1
        assert gg.SM_CONTENT.n_gauge_total == 12


class TestTriality:

    def test_verify_triality(self):
        t = gg.verify_triality()
        assert t["all_preserve_cartan"]
        assert t["n_automorphisms"] == 3

    def test_triality_permutation(self):
        perms = gg.triality_permutation()
        assert len(perms) == 3


class TestCoset:

    def test_coset_pairs_count(self):
        pairs = gg.coset_pairs()
        assert len(pairs) == 8  # 16 coset roots = 8 pairs

    def test_coset_pairs_opposite(self):
        """Each pair (r, -r) sums to zero."""
        for a, b in gg.coset_pairs():
            s = tuple(x + y for x, y in zip(a, b))
            assert all(c == 0 for c in s)


class TestDeltaFromRoots:

    def test_delta_g_from_roots(self):
        assert gg.DELTA_G_FROM_ROOTS == Fraction(1, 24)

    def test_delta_from_geometry(self):
        assert gg.DELTA_FROM_GEOMETRY == Fraction(5, 24)
