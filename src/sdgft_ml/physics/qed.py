r"""QED vertex corrections: anomalous magnetic moments (g−2) in SDGFT.

    Δa_ℓ = (α/2π) · γ²_geo · ln(m_ℓ/m_e)

Electron: Δa_e = 0 (exactly — lightest charged fermion, no RG running)
Muon:     Δa_μ ≈ 2.39 × 10⁻⁹  (obs anomaly: 2.49 ± 0.48 × 10⁻⁹)
Tau:      Δa_τ ≈ 3.66 × 10⁻⁹  (prediction for Belle II / FCC-ee)

Same γ_geo = δ²/D* = 1/1608 as dark energy and Lamb shift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .constants import ALPHA_OBS, M_MU_OVER_M_E, M_TAU_OVER_M_E, M_E_MEV, M_MU_MEV, M_TAU_MEV
from .dimension import (
    GAMMA_GEO_TREE, GAMMA_GEO_TREE_F,
    GAMMA_GEO_TREE_SQ, GAMMA_GEO_TREE_SQ_F,
    ALPHA_EM_TREE, ALPHA_EM_INV_TREE, MU_E_RATIO, TAU_E_RATIO_TREE,
)

# ── Experimental values ───────────────────────────────────────────

A_MU_EXP: float = 116_592_059e-11
A_MU_EXP_UNCERT: float = 22e-11

A_E_EXP: float = 0.001_159_652_180_59
A_E_EXP_UNCERT: float = 1.3e-13

A_MU_SM_WP: float = 116_591_810e-11
A_MU_SM_WP_UNCERT: float = 43e-11

DELTA_A_MU_OBS: float = A_MU_EXP - A_MU_SM_WP
DELTA_A_MU_OBS_UNCERT: float = math.sqrt(A_MU_EXP_UNCERT ** 2 + A_MU_SM_WP_UNCERT ** 2)

# ── Core formula ──────────────────────────────────────────────────


def delta_a_lepton(m_ell_over_m_e: float,
                   alpha: float = ALPHA_OBS,
                   gamma_sq: float = GAMMA_GEO_TREE_SQ_F) -> float:
    """Δa_ℓ = (α/2π) · γ²_geo · ln(m_ℓ/m_e).  Returns 0 if m_ℓ ≤ m_e."""
    if m_ell_over_m_e <= 1.0:
        return 0.0
    return (alpha / (2.0 * math.pi)) * gamma_sq * math.log(m_ell_over_m_e)


def delta_a_electron() -> float:
    return 0.0


def delta_a_muon(alpha: float = ALPHA_OBS,
                 gamma_sq: float = GAMMA_GEO_TREE_SQ_F,
                 mass_ratio: float = M_MU_OVER_M_E) -> float:
    return delta_a_lepton(mass_ratio, alpha, gamma_sq)


def delta_a_tau(alpha: float = ALPHA_OBS,
                gamma_sq: float = GAMMA_GEO_TREE_SQ_F,
                mass_ratio: float = M_TAU_OVER_M_E) -> float:
    return delta_a_lepton(mass_ratio, alpha, gamma_sq)


# ── Module-level values ───────────────────────────────────────────

DELTA_A_E: float = 0.0
DELTA_A_MU: float = delta_a_muon()
DELTA_A_TAU: float = delta_a_tau()

A_MU_SDGFT_TOTAL: float = A_MU_SM_WP + DELTA_A_MU
A_MU_SDGFT_SIGMA: float = (
    abs(A_MU_SDGFT_TOTAL - A_MU_EXP)
    / math.sqrt(A_MU_EXP_UNCERT ** 2 + A_MU_SM_WP_UNCERT ** 2)
)

# Pure SDGFT (tree α and mass ratios)
DELTA_A_MU_PURE: float = delta_a_muon(alpha=ALPHA_EM_TREE, mass_ratio=MU_E_RATIO)
DELTA_A_TAU_PURE: float = delta_a_tau(alpha=ALPHA_EM_TREE, mass_ratio=TAU_E_RATIO_TREE)

TAU_MU_GEO_RATIO: float = math.log(M_TAU_OVER_M_E) / math.log(M_MU_OVER_M_E)
"""Δa_τ/Δa_μ = ln(m_τ/m_e)/ln(m_μ/m_e) ≈ 1.529 (pure geometry)."""

# ── d-dimensional Schwinger Ξ(d) (diagnostic) ────────────────────


def xi_d(d: float) -> float:
    """⚠ Diagnostic only — NOT the physical SDGFT prediction."""
    if d <= 2.0:
        return 0.0
    return (
        (d - 2.0)
        * math.gamma(3.0 - d / 2.0)
        * math.gamma(d / 2.0 - 1.0) ** 2
        * (4.0 * math.pi) ** (2.0 - d / 2.0)
        / (2.0 * math.gamma(d - 2.0))
    )


# ── G2Prediction dataclass ───────────────────────────────────────

@dataclass(frozen=True)
class G2Prediction:
    """SDGFT g-2 prediction for a lepton."""
    lepton: str
    mass_mev: float
    mass_ratio: float
    delta_a_geo: float
    a_sm: float | None
    a_exp: float | None
    a_exp_uncert: float | None

    @property
    def a_sdgft(self) -> float | None:
        return None if self.a_sm is None else self.a_sm + self.delta_a_geo

    @property
    def sigma_vs_exp(self) -> float | None:
        a = self.a_sdgft
        if a is None or self.a_exp is None or self.a_exp_uncert is None:
            return None
        if self.a_exp_uncert <= 0.0:
            return None
        if self.lepton == "μ" and A_MU_SM_WP_UNCERT > 0:
            combined = math.sqrt(self.a_exp_uncert ** 2 + A_MU_SM_WP_UNCERT ** 2)
            return abs(a - self.a_exp) / combined
        return abs(a - self.a_exp) / self.a_exp_uncert

    @property
    def fraction_of_anomaly(self) -> float | None:
        if self.a_sm is None or self.a_exp is None:
            return None
        anomaly = self.a_exp - self.a_sm
        return None if anomaly == 0.0 else self.delta_a_geo / anomaly


def predict_electron() -> G2Prediction:
    return G2Prediction("e", M_E_MEV, 1.0, DELTA_A_E, None, A_E_EXP, A_E_EXP_UNCERT)


def predict_muon() -> G2Prediction:
    return G2Prediction("μ", M_MU_MEV, M_MU_OVER_M_E, DELTA_A_MU,
                        A_MU_SM_WP, A_MU_EXP, A_MU_EXP_UNCERT)


def predict_tau() -> G2Prediction:
    return G2Prediction("τ", M_TAU_MEV, M_TAU_OVER_M_E, DELTA_A_TAU, None, None, None)
