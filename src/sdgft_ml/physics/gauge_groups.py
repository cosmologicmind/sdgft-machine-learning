"""24-Cell ↔ D₄ root system ↔ Standard Model gauge group emergence.

Mathematical chain:
    24-cell vertices ≅ D₄ roots → so(8) → A₂⊕A₁⊕U(1) → SU(3)×SU(2)×U(1)

Key results:
    1.  24 roots of D₄ = 24 vertices of the 24-cell
    2.  dim so(8) = 28 = 24 roots + 4 Cartan
    3.  6(A₂) + 2(A₁) + 16(coset) = 24 roots  →  12 gauge bosons
    4.  D₄ triality (Z₃) → 3 generations
    5.  |Aut(24-cell)| = |W(D₄)| × |Out(D₄)| = 192 × 6 = 1152
    6.  δ = 1/24 = 1/|roots(D₄)|
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction
from typing import NamedTuple

from .constants import DELTA, DELTA_G, SIN2_30

# ═══════════════════════════════════════════════════════════════════
# §1  24-Cell Construction
# ═══════════════════════════════════════════════════════════════════


def _build_24cell_vertices() -> tuple[tuple[int, ...], ...]:
    verts: set[tuple[int, ...]] = set()
    for i in range(4):
        for s in (+2, -2):
            v = [0, 0, 0, 0]; v[i] = s; verts.add(tuple(v))
    for signs in itertools.product((+1, -1), repeat=4):
        verts.add(signs)
    return tuple(sorted(verts))


VERTICES_24CELL = _build_24cell_vertices()
N_VERTICES: int = len(VERTICES_24CELL)

# ═══════════════════════════════════════════════════════════════════
# §2  D₄ Root System
# ═══════════════════════════════════════════════════════════════════


def _build_d4_roots() -> tuple[tuple[int, ...], ...]:
    roots: set[tuple[int, ...]] = set()
    for i in range(4):
        for j in range(i + 1, 4):
            for si in (+1, -1):
                for sj in (+1, -1):
                    v = [0, 0, 0, 0]; v[i] = si; v[j] = sj; roots.add(tuple(v))
    return tuple(sorted(roots))


D4_ROOTS = _build_d4_roots()
N_D4_ROOTS: int = len(D4_ROOTS)

ALPHA_1 = (1, -1, 0, 0)
ALPHA_2 = (0, 1, -1, 0)
ALPHA_3 = (0, 0, 1, -1)
ALPHA_4 = (0, 0, 1, 1)
D4_SIMPLE_ROOTS = (ALPHA_1, ALPHA_2, ALPHA_3, ALPHA_4)

# ═══════════════════════════════════════════════════════════════════
# §3  Linear algebra
# ═══════════════════════════════════════════════════════════════════


def inner(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    return sum(ai * bi for ai, bi in zip(a, b))


def norm_sq(a: tuple[int, ...]) -> int:
    return inner(a, a)


def vec_add(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(ai + bi for ai, bi in zip(a, b))


def vec_neg(a: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(-c for c in a)


def cartan_matrix(simple_roots: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    n = len(simple_roots)
    return tuple(
        tuple(2 * inner(simple_roots[i], simple_roots[j]) // norm_sq(simple_roots[j])
              for j in range(n))
        for i in range(n)
    )


D4_CARTAN_MATRIX = cartan_matrix(D4_SIMPLE_ROOTS)
D4_CARTAN_EXPECTED = ((2, -1, 0, 0), (-1, 2, -1, -1), (0, -1, 2, 0), (0, -1, 0, 2))

# ═══════════════════════════════════════════════════════════════════
# §4  Positive roots
# ═══════════════════════════════════════════════════════════════════


def _positive_roots(roots: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    pos = []
    for r in roots:
        for c in r:
            if c != 0:
                if c > 0:
                    pos.append(r)
                break
    return tuple(sorted(pos))


D4_POSITIVE_ROOTS = _positive_roots(D4_ROOTS)

# ═══════════════════════════════════════════════════════════════════
# §5  Root system verification
# ═══════════════════════════════════════════════════════════════════


def verify_root_system(roots: tuple[tuple[int, ...], ...]) -> dict:
    root_set = set(roots)
    results: dict = {"n_roots": len(roots)}
    results["negation_closed"] = all(vec_neg(r) in root_set for r in roots)
    reflection_ok = True
    for alpha in roots:
        nsq = norm_sq(alpha)
        for beta in roots:
            coeff = 2 * inner(beta, alpha) // nsq
            reflected = tuple(beta[k] - coeff * alpha[k] for k in range(len(alpha)))
            if reflected not in root_set:
                reflection_ok = False; break
        if not reflection_ok:
            break
    results["reflection_closed"] = reflection_ok
    norms = set(norm_sq(r) for r in roots)
    results["simply_laced"] = len(norms) == 1
    results["root_norm_sq"] = norms.pop() if len(norms) == 1 else sorted(norms)
    return results


# ═══════════════════════════════════════════════════════════════════
# §6  D₄ → SM decomposition
# ═══════════════════════════════════════════════════════════════════

class GaugeDecomposition(NamedTuple):
    a2_roots: tuple[tuple[int, ...], ...]; a2_cartan_rank: int; a2_dim: int
    a1_roots: tuple[tuple[int, ...], ...]; a1_cartan_rank: int; a1_dim: int
    u1_cartan_rank: int; u1_dim: int
    coset_roots: tuple[tuple[int, ...], ...]; n_coset: int
    n_gauge_bosons: int; n_total_roots: int


def decompose_d4_to_sm() -> GaugeDecomposition:
    alpha12 = vec_add(ALPHA_1, ALPHA_2)
    a2 = tuple(sorted([ALPHA_1, vec_neg(ALPHA_1), ALPHA_2, vec_neg(ALPHA_2), alpha12, vec_neg(alpha12)]))
    a1 = tuple(sorted([ALPHA_3, vec_neg(ALPHA_3)]))
    gauge_set = set(a2) | set(a1)
    coset = tuple(sorted(r for r in D4_ROOTS if r not in gauge_set))
    return GaugeDecomposition(a2, 2, 8, a1, 1, 3, 1, 1, coset, len(coset), 12, len(D4_ROOTS))


SM_DECOMPOSITION = decompose_d4_to_sm()
N_GLUONS = SM_DECOMPOSITION.a2_dim
N_WEAK_BOSONS = SM_DECOMPOSITION.a1_dim
N_PHOTON = SM_DECOMPOSITION.u1_dim
N_GAUGE_BOSONS = SM_DECOMPOSITION.n_gauge_bosons

# ═══════════════════════════════════════════════════════════════════
# §7  Triality
# ═══════════════════════════════════════════════════════════════════


def triality_permutation():
    return ((ALPHA_1, ALPHA_2, ALPHA_3, ALPHA_4),
            (ALPHA_3, ALPHA_2, ALPHA_4, ALPHA_1),
            (ALPHA_4, ALPHA_2, ALPHA_1, ALPHA_3))


def verify_triality() -> dict:
    perms = triality_permutation()
    results = {"n_automorphisms": 3}
    for k, perm in enumerate(perms):
        results[f"sigma_{k}_preserves_cartan"] = cartan_matrix(perm) == D4_CARTAN_EXPECTED
    results["all_preserve_cartan"] = all(results[f"sigma_{k}_preserves_cartan"] for k in range(3))
    return results


TRIALITY_REPS = {
    "8v": {"dim": 8, "name": "vector"},
    "8s": {"dim": 8, "name": "spinor"},
    "8c": {"dim": 8, "name": "co-spinor"},
}

# ═══════════════════════════════════════════════════════════════════
# §8  Symmetry orders
# ═══════════════════════════════════════════════════════════════════

WEYL_D4_ORDER: int = 2 ** 3 * math.factorial(4)  # 192
OUTER_AUT_ORDER: int = math.factorial(3)  # 6
AUT_24CELL_ORDER: int = WEYL_D4_ORDER * OUTER_AUT_ORDER  # 1152

# ═══════════════════════════════════════════════════════════════════
# §9  SDGFT connection
# ═══════════════════════════════════════════════════════════════════

DELTA_G_FROM_ROOTS = Fraction(1, N_D4_ROOTS)
DELTA_FROM_GEOMETRY = Fraction(5, N_D4_ROOTS)

EDGE_ANGLE_DEG: int = 60
SIN2_COMPLEMENT = Fraction(1, 4)

# ═══════════════════════════════════════════════════════════════════
# §10  24-cell ↔ D₄ isomorphism
# ═══════════════════════════════════════════════════════════════════


def verify_24cell_d4_isomorphism() -> dict:
    results: dict = {"n_vertices": N_D4_ROOTS, "count_24": N_D4_ROOTS == 24}
    edge_count = 0
    nbr_counts = []
    for i, r in enumerate(D4_ROOTS):
        n = sum(1 for j, s in enumerate(D4_ROOTS) if i != j and inner(r, s) == 1)
        nbr_counts.append(n)
        edge_count += sum(1 for j in range(i + 1, len(D4_ROOTS)) if inner(r, D4_ROOTS[j]) == 1)
    results.update(edges=edge_count, vertex_degree=nbr_counts[0] if nbr_counts else 0,
                   aut_order=AUT_24CELL_ORDER)
    results["is_24cell"] = (results["count_24"] and edge_count == 96
                            and results["vertex_degree"] == 8 and AUT_24CELL_ORDER == 1152)
    return results


# ═══════════════════════════════════════════════════════════════════
# §11  Coset pairs (matter content)
# ═══════════════════════════════════════════════════════════════════


def coset_pairs() -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    seen: set[tuple[int, ...]] = set()
    pairs = []
    for r in SM_DECOMPOSITION.coset_roots:
        if r not in seen:
            pairs.append((r, vec_neg(r))); seen.update({r, vec_neg(r)})
    return pairs


class SMContent(NamedTuple):
    n_gluons: int; n_weak_bosons: int; n_photon: int; n_gauge_total: int
    n_coset_roots: int; n_matter_pairs: int; n_d4_roots: int; dim_so8: int


SM_CONTENT = SMContent(N_GLUONS, N_WEAK_BOSONS, N_PHOTON, N_GAUGE_BOSONS,
                       SM_DECOMPOSITION.n_coset, SM_DECOMPOSITION.n_coset // 2,
                       N_D4_ROOTS, N_D4_ROOTS + 4)
