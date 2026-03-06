r"""Galaxy rotation curves from scale-dependent gravity G(r).

Freeman (1970) exact thin-disk model + SDGFT running G + chameleon screening.

ε_gal candidates:
    A: α_M = 19/86 ≈ 0.221
    B: α_M(1−α_M) ≈ 0.172
    C: (1/3+δ)/(D*−1) ≈ 0.198
    D: α_M(1−Δ) ≈ 0.175
Obs:  ε = 0.16 ± 0.05
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .constants import G_N, KPC_M, M_SUN, DELTA_F, DELTA_G_F, PHI
from .dimension import (
    D_STAR_TREE_F, D_STAR_FP, N_TREE_F,
    ALPHA_M_TREE_F, R_TRANS_KPC, EPSILON_GAL,
)

KMS_TO_MS: float = 1.0e3

# ═══════════════════════════════════════════════════════════════════
# Modified Bessel functions (Abramowitz & Stegun §9.8)
# ═══════════════════════════════════════════════════════════════════


def _besseli0(x: float) -> float:
    ax = abs(x)
    if ax <= 3.75:
        t = (ax / 3.75) ** 2
        return (1.0 + t * (3.5156229 + t * (3.0899424 + t * (1.2067492
                + t * (0.2659732 + t * (0.0360768 + t * 0.0045813))))))
    t = 3.75 / ax
    poly = (0.39894228 + t * (0.01328592 + t * (0.00225319 + t * (-0.00157565
            + t * (0.00916281 + t * (-0.02057706 + t * (0.02635537
            + t * (-0.01647633 + t * 0.00392377))))))))
    return poly * math.exp(ax) / math.sqrt(ax)


def _besseli1(x: float) -> float:
    ax = abs(x)
    if ax <= 3.75:
        t = (ax / 3.75) ** 2
        return ax * (0.5 + t * (0.87890594 + t * (0.51498869 + t * (0.15084934
                     + t * (0.02658733 + t * (0.00301532 + t * 0.00032411))))))
    t = 3.75 / ax
    poly = (0.39894228 + t * (-0.03988024 + t * (-0.00362018 + t * (0.00163801
            + t * (-0.01031555 + t * (0.02282967 + t * (-0.02895312
            + t * (0.01787654 + t * (-0.00420059)))))))))
    return poly * math.exp(ax) / math.sqrt(ax)


def _besselk0(x: float) -> float:
    if x <= 2.0:
        t = (x / 2.0) ** 2
        poly = (-0.57721566 + t * (0.42278420 + t * (0.23069756 + t * (0.03488590
                + t * (0.00262698 + t * (0.00010750 + t * 0.00000740))))))
        return poly - math.log(x / 2.0) * _besseli0(x)
    t = 2.0 / x
    poly = (1.25331414 + t * (-0.07832358 + t * (0.02189568 + t * (-0.01062446
            + t * (0.00587872 + t * (-0.00251540 + t * 0.00053208))))))
    return poly * math.exp(-x) / math.sqrt(x)


def _besselk1(x: float) -> float:
    if x <= 2.0:
        t = (x / 2.0) ** 2
        poly = (1.0 + t * (0.15443144 + t * (-0.67278579 + t * (-0.18156897
                + t * (-0.01919402 + t * (-0.00110404 + t * (-0.00004686)))))))
        return poly / x + math.log(x / 2.0) * _besseli1(x)
    t = 2.0 / x
    poly = (1.25331414 + t * (0.23498619 + t * (-0.03655620 + t * (0.01504268
            + t * (-0.00780353 + t * (0.00325614 + t * (-0.00068245)))))))
    return poly * math.exp(-x) / math.sqrt(x)


def freeman_factor(y: float) -> float:
    """y² · [I₀(y)K₀(y) − I₁(y)K₁(y)]  — core of Freeman (1970)."""
    if y < 1e-10:
        return 0.0
    return y ** 2 * (_besseli0(y) * _besselk0(y) - _besseli1(y) * _besselk1(y))


# ═══════════════════════════════════════════════════════════════════
# Freeman thin-disk rotation curve
# ═══════════════════════════════════════════════════════════════════


def v2_freeman_disk(r_kpc: float, m_total_msun: float, h_kpc: float) -> float:
    """v²(R) for an exponential thin disk (Freeman 1970)  [m²/s²]."""
    if r_kpc <= 0 or h_kpc <= 0:
        return 0.0
    y = r_kpc / (2.0 * h_kpc)
    h_m = h_kpc * KPC_M
    return 2.0 * G_N * m_total_msun * M_SUN / h_m * freeman_factor(y)


# ═══════════════════════════════════════════════════════════════════
# ε_gal candidates
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class EpsilonCandidate:
    name: str; label: str; value: float; formula: str


EPSILON_OBS: float = 0.16
EPSILON_OBS_UNC: float = 0.05


def build_epsilon_candidates(n: float = N_TREE_F,
                             alpha_m: float = ALPHA_M_TREE_F,
                             delta: float = DELTA_F,
                             delta_g: float = DELTA_G_F,
                             d_star: float = D_STAR_TREE_F) -> list[EpsilonCandidate]:
    candidates = [
        EpsilonCandidate("alpha_m", "A: α_M=(n−1)/(2n−1)", alpha_m, r"\alpha_M"),
        EpsilonCandidate("braiding_damped", "B: α_M(1−α_M)",
                         alpha_m * (1 - alpha_m), r"\alpha_M(1-\alpha_M)"),
        EpsilonCandidate("chameleon", "C: (1/3+δ)/(D*−1)",
                         (1.0 / 3.0 + delta_g) / (d_star - 1.0), r"(1/3+\delta)/(D^*-1)"),
        EpsilonCandidate("fibonacci_screened", "D: α_M(1−Δ)",
                         alpha_m * (1 - delta), r"\alpha_M(1-\Delta)"),
    ]
    return sorted(candidates, key=lambda c: abs(c.value - EPSILON_OBS))


EPSILON_CANDIDATES: list[EpsilonCandidate] = build_epsilon_candidates()
EPSILON_BEST: EpsilonCandidate = EPSILON_CANDIDATES[0]

# ═══════════════════════════════════════════════════════════════════
# G_eff(r) at galactic scales
# ═══════════════════════════════════════════════════════════════════


def g_eff_galactic(r_kpc: float, epsilon: float = EPSILON_GAL,
                   r_trans_kpc: float = R_TRANS_KPC,
                   g_n: float = G_N) -> float:
    if r_kpc <= r_trans_kpc:
        return g_n
    return g_n * (1.0 + epsilon * math.log(r_kpc / r_trans_kpc))


def g_eff_profile(radii_kpc: list[float], epsilon: float = EPSILON_GAL,
                  r_trans_kpc: float = R_TRANS_KPC) -> list[float]:
    return [g_eff_galactic(r, epsilon, r_trans_kpc) / G_N for r in radii_kpc]


# ═══════════════════════════════════════════════════════════════════
# Baryonic mass models
# ═══════════════════════════════════════════════════════════════════


def enclosed_mass_exponential(r_kpc: float, m_total_msun: float,
                              h_kpc: float) -> float:
    """Spherical approximation: M(<r) = M·[1−(1+r/h)e^{−r/h}]  [kg]."""
    x = r_kpc / h_kpc
    return m_total_msun * M_SUN * (1.0 - (1.0 + x) * math.exp(-x))


def surface_density_exponential(r_kpc: float, m_total_msun: float,
                                h_kpc: float) -> float:
    """Σ(R) = (M/2πh²) exp(−R/h) [kg/m²]."""
    h_m = h_kpc * KPC_M
    return m_total_msun * M_SUN / (2.0 * math.pi * h_m ** 2) * math.exp(-r_kpc / h_kpc)


# ═══════════════════════════════════════════════════════════════════
# Chameleon screening
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ScreeningConfig:
    sigma_screen: float = 0.0   # [kg/m²] — auto-computed if 0
    steepness: float = 2.0


def screening_factor(sigma: float, cfg: ScreeningConfig) -> float:
    """S(R) = 1/(1 + (Σ/Σ_screen)^p)."""
    if cfg.sigma_screen <= 0:
        return 1.0
    return 1.0 / (1.0 + (sigma / cfg.sigma_screen) ** cfg.steepness)


# ═══════════════════════════════════════════════════════════════════
# Full rotation-curve model
# ═══════════════════════════════════════════════════════════════════

@dataclass
class GalaxyModel:
    """Multi-component galaxy for rotation-curve fitting."""
    name: str
    components: list[dict]  # each: {mass_msun, h_kpc, label}
    distance_mpc: float = 0.0
    v_obs_kpc: list[float] | None = None
    v_obs_kms: list[float] | None = None
    v_obs_err: list[float] | None = None


def rotation_curve(model: GalaxyModel, radii_kpc: list[float],
                   epsilon: float = EPSILON_GAL,
                   r_trans_kpc: float = R_TRANS_KPC,
                   exact: bool = True,
                   screening: ScreeningConfig | None = None) -> list[float]:
    """Compute v(R) [km/s] for a multi-component galaxy.

    Args:
        model: Galaxy model with baryonic components.
        radii_kpc: Radii to evaluate.
        epsilon: Logarithmic G modification strength.
        r_trans_kpc: Transition radius.
        exact: Use Freeman thin-disk (True) or spherical approx.
        screening: Optional chameleon screening configuration.
    """
    v_kms = []
    for r in radii_kpc:
        v2_bary = 0.0
        sigma_total = 0.0
        for comp in model.components:
            m, h = comp["mass_msun"], comp["h_kpc"]
            if exact:
                v2_bary += v2_freeman_disk(r, m, h)
            else:
                m_enc = enclosed_mass_exponential(r, m, h)
                v2_bary += G_N * m_enc / (r * KPC_M) if r > 0 else 0.0
            sigma_total += surface_density_exponential(r, m, h)

        # G_eff modification
        if r > r_trans_kpc:
            log_fac = math.log(r / r_trans_kpc)
            if screening is not None:
                sigma_scr = screening.sigma_screen
                if sigma_scr <= 0:
                    # Auto: Σ at r_trans (compute locally, don't mutate config)
                    sigma_scr = sum(
                        surface_density_exponential(r_trans_kpc, c["mass_msun"], c["h_kpc"])
                        for c in model.components
                    )
                s = screening_factor(sigma_total,
                                     ScreeningConfig(sigma_scr, screening.steepness))
                g_ratio = 1.0 + s * epsilon * log_fac
            else:
                g_ratio = 1.0 + epsilon * log_fac
        else:
            g_ratio = 1.0

        v2_total = v2_bary * g_ratio
        v_kms.append(math.sqrt(max(0.0, v2_total)) / KMS_TO_MS)

    return v_kms


# ═══════════════════════════════════════════════════════════════════
# NGC 3198 demo model
# ═══════════════════════════════════════════════════════════════════

NGC3198 = GalaxyModel(
    name="NGC 3198",
    distance_mpc=13.8,
    components=[
        {"mass_msun": 2.6e10, "h_kpc": 2.72, "label": "Stellar disk"},
        {"mass_msun": 8.3e9, "h_kpc": 6.50, "label": "Gas (HI+He)"},
    ],
)

# ═══════════════════════════════════════════════════════════════════
# Tully-Fisher relation
# ═══════════════════════════════════════════════════════════════════


def tully_fisher_luminosity(v_flat_kms: float, b: float = 3.792) -> float:
    """log₁₀(L/L☉) from Tully-Fisher: L ∝ v^b with b = 91/24."""
    return b * math.log10(v_flat_kms)
