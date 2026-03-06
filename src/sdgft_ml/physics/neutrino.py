r"""Neutrino masses, PMNS matrix, and oscillation probabilities.

Mass-splitting ratio:  R = D*/(2δ) = 67/2 = 33.5  (obs ≈ 33.6 ± 0.9)
Normal ordering:  m₁ = 0, m₂ ≈ 8.5 meV, m₃ ≈ 49.4 meV
CP phase:  δ_CP = 5π/4  (testable by DUNE, T2K, JUNO)
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from fractions import Fraction

from .constants import (
    DELTA, DELTA_F, DELTA_G, DELTA_G_F, M_E_GEV, V_HIGGS_GEV, M_PL_GEV,
)
from .dimension import (
    D_STAR_TREE, D_STAR_TREE_F, D_STAR_FP,
    theta_12, theta_13, theta_23,
)

# ── Oscillation unit conversion ───────────────────────────────────

HBAR_C_EV_M: float = 1.973_269_804e-7
_KM_TO_INV_EV: float = 1e3 / HBAR_C_EV_M
_GEV_TO_EV: float = 1e9
OSC_PHASE_COEFF: float = _KM_TO_INV_EV / (2.0 * _GEV_TO_EV)
OSC_REDUCED_COEFF: float = OSC_PHASE_COEFF / 2.0

_FLAVOR_MAP: dict[str, int] = {
    "e": 0, "electron": 0, "nu_e": 0,
    "mu": 1, "muon": 1, "nu_mu": 1,
    "tau": 2, "nu_tau": 2,
}

# ── Observed values (PDG 2024, NO) ───────────────────────────────

DM2_21_OBS: float = 7.53e-5
DM2_21_OBS_UNC: float = 0.18e-5
DM2_32_OBS: float = 2.453e-3
DM2_32_OBS_UNC: float = 0.033e-3
DM2_31_OBS: float = DM2_32_OBS + DM2_21_OBS
DM2_31_OBS_UNC: float = math.sqrt(DM2_32_OBS_UNC ** 2 + DM2_21_OBS_UNC ** 2)
RATIO_OBS: float = DM2_31_OBS / DM2_21_OBS
RATIO_OBS_UNC: float = RATIO_OBS * math.sqrt(
    (DM2_31_OBS_UNC / DM2_31_OBS) ** 2 + (DM2_21_OBS_UNC / DM2_21_OBS) ** 2
)
M_BB_OBS_LIMIT: float = 0.036

# ── Neutrino mass sum ─────────────────────────────────────────────


def neutrino_mass_sum(delta_g: float = DELTA_G_F,
                      m_e: float = M_E_GEV,
                      v: float = V_HIGGS_GEV,
                      m_pl: float = M_PL_GEV) -> float:
    """Σm_ν = δ · m_e · (v/M_Pl)^{1/3}  [eV]."""
    return delta_g * m_e * (v / m_pl) ** (1.0 / 3.0) * 1e9


SUM_M_NU: float = neutrino_mass_sum()

# ── Mass-splitting ratio ─────────────────────────────────────────


def mass_splitting_ratio_exact() -> Fraction:
    """R = D*/(2δ) = 67/2 = 33.5."""
    return D_STAR_TREE / (2 * DELTA_G)


def mass_splitting_ratio(d_star: float = D_STAR_TREE_F,
                         delta_g: float = DELTA_G_F) -> float:
    return d_star / (2.0 * delta_g)


R_TREE: Fraction = mass_splitting_ratio_exact()
R_TREE_F: float = float(R_TREE)
R_FP: float = mass_splitting_ratio(D_STAR_FP, DELTA_G_F)

# ── Mass spectrum ─────────────────────────────────────────────────


def neutrino_masses(sum_m_nu: float | None = None,
                    r: float | None = None) -> tuple[float, float, float]:
    """(m₁, m₂, m₃) in eV.  m₁ = 0 (normal ordering)."""
    if sum_m_nu is None:
        sum_m_nu = SUM_M_NU
    if r is None:
        r = R_TREE_F
    sqrt_r = math.sqrt(r)
    m2 = sum_m_nu / (1.0 + sqrt_r)
    m3 = sum_m_nu * sqrt_r / (1.0 + sqrt_r)
    return 0.0, m2, m3


def delta_m2_21(sum_m_nu: float | None = None, r: float | None = None) -> float:
    _, m2, _ = neutrino_masses(sum_m_nu, r)
    return m2 ** 2


def delta_m2_31(sum_m_nu: float | None = None, r: float | None = None) -> float:
    _, _, m3 = neutrino_masses(sum_m_nu, r)
    return m3 ** 2


def delta_m2_32(sum_m_nu: float | None = None, r: float | None = None) -> float:
    _, m2, m3 = neutrino_masses(sum_m_nu, r)
    return m3 ** 2 - m2 ** 2


# ── CP phase ──────────────────────────────────────────────────────


def delta_cp_pmns() -> float:
    """δ_CP = 5π/4 ≈ 225°."""
    return 5.0 * math.pi / 4.0


DELTA_CP: float = delta_cp_pmns()

# ── PMNS matrix ───────────────────────────────────────────────────


def pmns_angles_deg() -> tuple[float, float, float]:
    """(θ₁₂, θ₂₃, θ₁₃) in degrees."""
    return theta_12(DELTA_G_F), theta_23(DELTA_F), theta_13(DELTA_F)


def pmns_matrix(theta_12_deg: float | None = None,
                theta_23_deg: float | None = None,
                theta_13_deg: float | None = None,
                delta_cp_rad: float | None = None) -> list[list[complex]]:
    """3×3 PMNS matrix (PDG parametrisation)."""
    if theta_12_deg is None or theta_23_deg is None or theta_13_deg is None:
        t12, t23, t13 = pmns_angles_deg()
        theta_12_deg = theta_12_deg if theta_12_deg is not None else t12
        theta_23_deg = theta_23_deg if theta_23_deg is not None else t23
        theta_13_deg = theta_13_deg if theta_13_deg is not None else t13
    if delta_cp_rad is None:
        delta_cp_rad = DELTA_CP

    s12, c12 = math.sin(math.radians(theta_12_deg)), math.cos(math.radians(theta_12_deg))
    s23, c23 = math.sin(math.radians(theta_23_deg)), math.cos(math.radians(theta_23_deg))
    s13, c13 = math.sin(math.radians(theta_13_deg)), math.cos(math.radians(theta_13_deg))
    eid = cmath.exp(1j * delta_cp_rad)
    emid = cmath.exp(-1j * delta_cp_rad)

    return [
        [complex(c12 * c13), complex(s12 * c13), s13 * emid],
        [-s12 * c23 - c12 * s23 * s13 * eid, c12 * c23 - s12 * s23 * s13 * eid, complex(s23 * c13)],
        [s12 * s23 - c12 * c23 * s13 * eid, -c12 * s23 - s12 * c23 * s13 * eid, complex(c23 * c13)],
    ]


def jarlskog_pmns(theta_12_deg: float | None = None,
                  theta_23_deg: float | None = None,
                  theta_13_deg: float | None = None,
                  delta_cp_rad: float | None = None) -> float:
    """J = c₁₂ s₁₂ c₂₃ s₂₃ c²₁₃ s₁₃ sin(δ_CP)."""
    if theta_12_deg is None or theta_23_deg is None or theta_13_deg is None:
        t12, t23, t13 = pmns_angles_deg()
        theta_12_deg = theta_12_deg or t12; theta_23_deg = theta_23_deg or t23; theta_13_deg = theta_13_deg or t13
    if delta_cp_rad is None:
        delta_cp_rad = DELTA_CP
    s12, c12 = math.sin(math.radians(theta_12_deg)), math.cos(math.radians(theta_12_deg))
    s23, c23 = math.sin(math.radians(theta_23_deg)), math.cos(math.radians(theta_23_deg))
    s13, c13 = math.sin(math.radians(theta_13_deg)), math.cos(math.radians(theta_13_deg))
    return c12 * s12 * c23 * s23 * c13 ** 2 * s13 * math.sin(delta_cp_rad)


def effective_majorana_mass(masses: tuple[float, float, float] | None = None,
                            U: list[list[complex]] | None = None) -> float:
    """m_ββ = |Σ U²_{ei} · mᵢ|."""
    if masses is None:
        masses = neutrino_masses()
    if U is None:
        U = pmns_matrix()
    return abs(sum(U[0][i] ** 2 * masses[i] for i in range(3)))


# ── Oscillation probability ───────────────────────────────────────

def _resolve_flavor(flavor: str | int) -> int:
    if isinstance(flavor, int):
        if flavor not in (0, 1, 2):
            raise ValueError(f"Flavor index must be 0, 1, or 2, got {flavor}")
        return flavor
    key = flavor.lower().strip()
    if key not in _FLAVOR_MAP:
        raise ValueError(f"Unknown flavor {flavor!r}")
    return _FLAVOR_MAP[key]


def oscillation_probability(
    alpha: str | int, beta: str | int,
    L_km: float, E_GeV: float,
    *, antineutrino: bool = False,
    masses: tuple[float, float, float] | None = None,
    U: list[list[complex]] | None = None,
) -> float:
    """P(ν_α → ν_β) in vacuum (full 3-flavour)."""
    a, b = _resolve_flavor(alpha), _resolve_flavor(beta)
    if masses is None:
        masses = neutrino_masses()
    if U is None:
        U = pmns_matrix()
    if antineutrino:
        U = [[elem.conjugate() for elem in row] for row in U]
    phase_coeff = OSC_PHASE_COEFF * L_km / E_GeV
    amplitude = 0j
    for i in range(3):
        phi_i = masses[i] ** 2 * phase_coeff
        amplitude += U[b][i] * cmath.exp(-1j * phi_i) * U[a][i].conjugate()
    return max(0.0, min(1.0, abs(amplitude) ** 2))


def cp_asymmetry(alpha: str | int, beta: str | int,
                 L_km: float, E_GeV: float, **kw) -> float:
    """A_CP = P(ν) − P(ν̄)."""
    return (oscillation_probability(alpha, beta, L_km, E_GeV, antineutrino=False, **kw)
            - oscillation_probability(alpha, beta, L_km, E_GeV, antineutrino=True, **kw))


# ── Experiment predictions ────────────────────────────────────────

@dataclass(frozen=True)
class ExperimentPrediction:
    name: str; baseline_km: float; energy_GeV: float; channel: str
    probability: float; probability_anti: float; cp_asymmetry: float


def predict_dune() -> ExperimentPrediction:
    L, E = 1285.0, 2.5
    p = oscillation_probability("mu", "e", L, E)
    pb = oscillation_probability("mu", "e", L, E, antineutrino=True)
    return ExperimentPrediction("DUNE", L, E, "ν_μ → ν_e", p, pb, p - pb)


def predict_t2k() -> ExperimentPrediction:
    L, E = 295.0, 0.6
    p = oscillation_probability("mu", "e", L, E)
    pb = oscillation_probability("mu", "e", L, E, antineutrino=True)
    return ExperimentPrediction("T2K", L, E, "ν_μ → ν_e", p, pb, p - pb)


def predict_juno() -> ExperimentPrediction:
    L, E = 52.5, 0.0035
    p = oscillation_probability("e", "e", L, E)
    pb = oscillation_probability("e", "e", L, E, antineutrino=True)
    return ExperimentPrediction("JUNO", L, E, "ν̄_e → ν̄_e", p, pb, p - pb)


def predict_nova() -> ExperimentPrediction:
    L, E = 810.0, 2.0
    p = oscillation_probability("mu", "e", L, E)
    pb = oscillation_probability("mu", "e", L, E, antineutrino=True)
    return ExperimentPrediction("NOvA", L, E, "ν_μ → ν_e", p, pb, p - pb)


# ── Module-level values ───────────────────────────────────────────

M1, M2, M3 = neutrino_masses()
DM2_21: float = delta_m2_21()
DM2_31: float = delta_m2_31()
DM2_32: float = delta_m2_32()
THETA_12_DEG, THETA_23_DEG, THETA_13_DEG = pmns_angles_deg()
U_PMNS: list[list[complex]] = pmns_matrix()
J_PMNS: float = jarlskog_pmns()
M_BB: float = effective_majorana_mass()
