r"""Hydrogen Lamb shift and fine structure from SDGFT 24-cell geometry.

L_geo = (Δ/(Δ+δ)) · γ²_geo · R∞c

Tree-level:  L ≈ 1060.3 MHz     (D* = 67/24)
Fixed-point: L ≈ 1056.1 MHz     (D*_fp ≈ 2.797)
Observed:    1057.845 ± 0.009 MHz

Same γ_geo = δ²/D* = 1/1608 governs dark energy, muon g-2, and Lamb shift.
"""

from __future__ import annotations

import math
from fractions import Fraction

from .constants import DELTA, DELTA_G, DELTA_G_F, SIN2_30, R_INF_C_MHZ, ALPHA_OBS
from .dimension import (
    D_STAR_TREE, D_STAR_TREE_F, D_STAR_FP,
    GAMMA_GEO_TREE, GAMMA_GEO_TREE_F,
    GAMMA_GEO_FP, GAMMA_GEO_TREE_SQ, GAMMA_GEO_TREE_SQ_F,
    GAMMA_GEO_FP_SQ,
)

# ── Observed Lamb shift ──────────────────────────────────────────

LAMB_SHIFT_OBS_MHZ: float = 1057.845
LAMB_SHIFT_OBS_UNCERT_MHZ: float = 0.009

# ── Projection factor ────────────────────────────────────────────

PROJECTION_FACTOR: Fraction = DELTA / (DELTA + DELTA_G)  # 5/6
PROJECTION_FACTOR_F: float = float(PROJECTION_FACTOR)

assert PROJECTION_FACTOR == Fraction(5, 6)

# ── Lamb shift functions ─────────────────────────────────────────


def lamb_shift_geo(gamma_geo_sq: float,
                   proj: float = PROJECTION_FACTOR_F,
                   r_inf_c: float = R_INF_C_MHZ) -> float:
    """L_geo = (Δ/(Δ+δ)) · γ²_geo · R∞c  [MHz]."""
    return proj * gamma_geo_sq * r_inf_c


def lamb_shift_tree() -> float:
    """Tree-level Lamb shift ≈ 1060.3 MHz."""
    return lamb_shift_geo(GAMMA_GEO_TREE_SQ_F)


def lamb_shift_fp() -> float:
    """Fixed-point Lamb shift ≈ 1056.1 MHz."""
    return lamb_shift_geo(GAMMA_GEO_FP_SQ)


LAMB_SHIFT_TREE: float = lamb_shift_tree()
LAMB_SHIFT_FP: float = lamb_shift_fp()
LAMB_SHIFT_WEIGHTED: float = (LAMB_SHIFT_TREE + LAMB_SHIFT_FP) / 2.0

LAMB_SHIFT_TREE_DEV_MHZ: float = LAMB_SHIFT_TREE - LAMB_SHIFT_OBS_MHZ
LAMB_SHIFT_FP_DEV_MHZ: float = LAMB_SHIFT_FP - LAMB_SHIFT_OBS_MHZ
LAMB_SHIFT_TREE_DEV_PCT: float = abs(LAMB_SHIFT_TREE_DEV_MHZ) / LAMB_SHIFT_OBS_MHZ * 100
LAMB_SHIFT_FP_DEV_PCT: float = abs(LAMB_SHIFT_FP_DEV_MHZ) / LAMB_SHIFT_OBS_MHZ * 100

# ── D* extraction from Lamb shift ────────────────────────────────


def d_star_from_lamb_shift(lamb_mhz: float = LAMB_SHIFT_OBS_MHZ) -> float:
    """Invert L = (5/6)·(δ²/D*)²·R∞c to extract D*."""
    gamma_geo_sq = lamb_mhz / (PROJECTION_FACTOR_F * R_INF_C_MHZ)
    gamma_geo = math.sqrt(gamma_geo_sq)
    return DELTA_G_F ** 2 / gamma_geo


D_STAR_FROM_LAMB: float = d_star_from_lamb_shift()

# ── Rydberg geometric correction ─────────────────────────────────


def rydberg_geo_correction() -> float:
    """δR∞/R∞ = (D*−3)/3 · α²."""
    return (D_STAR_TREE_F - 3.0) / 3.0 * ALPHA_OBS ** 2


RYDBERG_GEO_CORRECTION: float = rydberg_geo_correction()

# ── Fine-structure interval ───────────────────────────────────────

FINE_STRUCTURE_2P: float = ALPHA_OBS ** 2 * R_INF_C_MHZ / 16.0
FINE_STRUCTURE_2P_OBS: float = 10_969.04

# ── Muonic Lamb shift (geometric part) ────────────────────────
#
# The muonic hydrogen Lamb shift scales approximately as (m_μ/m_e)^3
# for the leading self-energy contribution, NOT linearly.
# The observed value is ~202 meV = ~49 THz, dominated by nuclear structure.
# Here we provide the geometric SDGFT correction only (same γ_geo form).

_MU_E_MASS_RATIO: float = 206.7682830

LAMB_SHIFT_MUONIC_TREE: float = LAMB_SHIFT_TREE * _MU_E_MASS_RATIO ** 3
"""Muonic Lamb shift ~ L_tree × (m_μ/m_e)³ [MHz].  Leading-order geometric scaling.

Note: this is a rough (m/m_e)^3 scaling of the self-energy contribution.
The full muonic Lamb shift is dominated by proton charge-radius effects
and requires QED+nuclear structure beyond the geometric SDGFT piece.
"""

LAMB_SHIFT_MUONIC_OBS_MEV: float = 202.3706
"""Observed muonic hydrogen 2S-2P Lamb shift [meV].  (Pohl et al. 2010)."""

# Conversion: 1 meV ≈ 241.799 GHz = 241799 MHz
_MEV_TO_MHZ: float = 241_799.0
LAMB_SHIFT_MUONIC_OBS_MHZ: float = LAMB_SHIFT_MUONIC_OBS_MEV * _MEV_TO_MHZ
