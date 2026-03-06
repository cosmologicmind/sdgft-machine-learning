r"""Collider signatures from SDGFT dimensional flow.

Completely new code — not ported from V2.

Predictions for high-energy colliders (LHC, FCC-hh, FCC-ee):
    1. Running couplings to M_Pl with SDGFT boundary conditions
    2. Modified Drell-Yan cross-section from dimensional flow
    3. Virtual graviton exchange (KK-like) at √s ≫ M_Z
    4. Effective Kaluza-Klein modes from D* > 3
    5. Higgs production modification from geometric vertex
    6. Dijet angular distributions from extra-dimensional effects
    7. BSM exclusion / discovery reach
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .constants import ALPHA_OBS, ALPHA_INV_OBS, G_N, C, HBAR, M_PL_GEV
from .dimension import (
    D_STAR_TREE_F, D_STAR_FP, N_TREE_F,
    GAMMA_GEO_TREE_F, GAMMA_GEO_TREE_SQ_F,
    ALPHA_M_TREE_F, ALPHA_EM_TREE, ALPHA_EM_INV_TREE,
)
from .rg_running import (
    couplings_from_observables, run_inverse_couplings,
    ALPHA_EM_INV_MZ, SIN2_THETA_W_MZ, ALPHA_S_MZ, M_Z, M_PL, T_PL,
    B1, B2, B3,
)

# ═══════════════════════════════════════════════════════════════════
# §1  Running couplings with SDGFT modifications
# ═══════════════════════════════════════════════════════════════════


def sdgft_modified_running(sqrt_s_gev: float) -> dict[str, float]:
    """SM 1-loop running + SDGFT γ_geo correction at scale √s.

    The SDGFT correction modifies the beta functions at O(γ_geo²):
        b_i → b_i · (1 + c_i · γ²_geo)
    where c_i depends on the gauge group representation content.

    For the SM: c_1 = 1, c_2 = 1, c_3 = D*/3
    (the colour sector gets a stronger correction from D* > 3).
    """
    if sqrt_s_gev <= M_Z:
        from .rg_running import run_to_scale
        r = run_to_scale(0.0)
        r["sqrt_s_gev"] = sqrt_s_gev
        return r

    t = math.log(sqrt_s_gev / M_Z)
    ia1_0, ia2_0, ia3_0 = couplings_from_observables(
        ALPHA_EM_INV_MZ, SIN2_THETA_W_MZ, ALPHA_S_MZ
    )

    # SDGFT-modified beta coefficients
    g2 = GAMMA_GEO_TREE_SQ_F
    b1_mod = B1 * (1.0 + g2)
    b2_mod = B2 * (1.0 + g2)
    b3_mod = B3 * (1.0 + D_STAR_TREE_F / 3.0 * g2)

    TWO_PI = 2.0 * math.pi
    ia1 = ia1_0 - b1_mod * t / TWO_PI
    ia2 = ia2_0 - b2_mod * t / TWO_PI
    ia3 = ia3_0 - b3_mod * t / TWO_PI

    return {
        "sqrt_s_gev": sqrt_s_gev,
        "inv_alpha_1": ia1, "inv_alpha_2": ia2, "inv_alpha_3": ia3,
        "alpha_s": 1.0 / ia3 if ia3 > 0 else float("inf"),
        "alpha_em_inv": (5.0 / 3.0) * ia1 + ia2,
    }


# ═══════════════════════════════════════════════════════════════════
# §2  Modified Drell-Yan cross-section
# ═══════════════════════════════════════════════════════════════════


def drell_yan_ratio(m_ll_gev: float, d_star: float = D_STAR_TREE_F) -> float:
    r"""σ_SDGFT / σ_SM for Drell-Yan at invariant mass m_ll.

    In D*-dimensional spacetime, the propagator picks up a correction:
        |D(q²)|² → |D(q²)|² · (1 + (D*−3) · α · ln(m_ll/M_Z) / π)

    This gives a subtle enhancement at high m_ll.
    """
    if m_ll_gev <= M_Z:
        return 1.0
    correction = (d_star - 3.0) * ALPHA_OBS * math.log(m_ll_gev / M_Z) / math.pi
    return 1.0 + correction


# ═══════════════════════════════════════════════════════════════════
# §3  Virtual graviton exchange
# ═══════════════════════════════════════════════════════════════════


def graviton_exchange_amplitude(sqrt_s_gev: float,
                                m_pl_gev: float = M_PL_GEV,
                                d_star: float = D_STAR_TREE_F) -> float:
    r"""Dimensionless amplitude for virtual graviton exchange.

    In SDGFT, the effective Planck mass runs with scale:
        M_Pl(k) ~ M_Pl · (k/k_0)^{α_M}   for k > k_trans

    The graviton exchange amplitude at √s:
        A_grav ~ (√s / M_Pl)^{D*-2}

    For D* ≈ 2.79: A ~ (√s / M_Pl)^0.79  (slower than D=4 gravity).
    """
    if sqrt_s_gev <= 0 or m_pl_gev <= 0:
        return 0.0
    return (sqrt_s_gev / m_pl_gev) ** (d_star - 2.0)


def graviton_exchange_cross_section_fb(sqrt_s_gev: float,
                                        m_pl_gev: float = M_PL_GEV) -> float:
    """σ(graviton exchange) in fb — virtual graviton t-channel.

    σ ~ π α² / s · A_grav²
    Converted to femtobarns.
    """
    s = sqrt_s_gev ** 2
    a_grav = graviton_exchange_amplitude(sqrt_s_gev, m_pl_gev)
    # Convert GeV⁻² to fb: 1 GeV⁻² = 0.3894e12 fb
    gev2_to_fb = 0.3894e12
    return math.pi * ALPHA_OBS ** 2 / s * a_grav ** 2 * gev2_to_fb


# ═══════════════════════════════════════════════════════════════════
# §4  Effective Kaluza-Klein spectrum
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class KKMode:
    """Effective KK mode from D* > 3 dimensional flow."""
    n: int
    mass_gev: float
    coupling_ratio: float  # relative to zero-mode


def kk_spectrum(n_max: int = 10, m_compactification_gev: float | None = None,
                d_star: float = D_STAR_TREE_F) -> list[KKMode]:
    """Effective KK tower from the fractal dimension D* > 3.

    In SDGFT, D* ≈ 2.79 < 3 means there are NO true extra dimensions.
    However, the fractal geometry produces an effective KK-like spectrum
    with spacing:
        M_n = n · M_c · (D*/3)^{n/2}

    where M_c is the compactification scale (set by r_trans in SDGFT,
    or by the dimensional flow).  Default: M_c ~ M_Pl · (D*−2).
    """
    if m_compactification_gev is None:
        m_compactification_gev = M_PL_GEV * abs(d_star - 2.0)

    modes = []
    for n in range(1, n_max + 1):
        m_n = n * m_compactification_gev * (d_star / 3.0) ** (n / 2.0)
        # Coupling falls off geometrically
        coupling = (d_star / 3.0) ** n
        modes.append(KKMode(n=n, mass_gev=m_n, coupling_ratio=coupling))
    return modes


# ═══════════════════════════════════════════════════════════════════
# §5  Higgs production modification
# ═══════════════════════════════════════════════════════════════════


def higgs_gg_modification(d_star: float = D_STAR_TREE_F) -> float:
    """Ratio σ(gg→H)_SDGFT / σ(gg→H)_SM.

    The top-quark loop in gg→H picks up a geometric correction:
        δσ/σ = 2 · γ²_geo · ln(m_t/m_b)

    where the factor 2 comes from the two gluon legs, and the log
    runs between top and bottom masses (the dominant loop contributors).
    """
    m_t = 172.76  # GeV
    m_b = 4.18    # GeV
    return 1.0 + 2.0 * GAMMA_GEO_TREE_SQ_F * math.log(m_t / m_b)


def higgs_width_modification(d_star: float = D_STAR_TREE_F) -> float:
    """Ratio Γ(H)_SDGFT / Γ(H)_SM.

    Total width modded by average geometric correction over all channels.
    Dominant: H→bb̄ (58%), H→WW* (21%), H→gg (8.5%).
    """
    # Weighted average of channel-specific corrections
    # bb̄: correction ~ γ²_geo · ln(m_b/m_e)  (small)
    # WW: correction ~ γ²_geo · ln(M_W/M_Z)  (tiny)
    # gg: same as production
    g2 = GAMMA_GEO_TREE_SQ_F
    delta_bb = g2 * math.log(4.18e3 / 0.511)  # m_b/m_e
    delta_ww = g2 * math.log(80.379 / 91.1876)
    delta_gg = 2.0 * g2 * math.log(172.76 / 4.18)
    return 1.0 + 0.58 * delta_bb + 0.21 * abs(delta_ww) + 0.085 * delta_gg


# ═══════════════════════════════════════════════════════════════════
# §6  Dijet angular distributions
# ═══════════════════════════════════════════════════════════════════


def dijet_f_chi(chi: float, d_star: float = D_STAR_TREE_F) -> float:
    r"""Normalized dijet angular distribution F(χ).

    In SM QCD (Rutherford scattering): dσ/dχ ∝ 1/χ²
    With graviton exchange: dσ/dχ ∝ 1/χ² + c · (√s/M_Pl)^{2(D*-2)}

    χ = exp(|y₁ − y₂|) = (1 + cos θ*)/(1 − cos θ*)

    Returns the SDGFT/SM ratio at given χ.
    """
    sm_part = 1.0 / chi ** 2
    # Graviton exchange contributes isotropically (spin-2 contact term)
    # The correction depends on √s; we parametrise at LHC 14 TeV
    sqrt_s = 14_000.0  # GeV
    a_grav = graviton_exchange_amplitude(sqrt_s)
    grav_part = a_grav ** 2  # isotropic in chi
    return (sm_part + grav_part) / sm_part


# ═══════════════════════════════════════════════════════════════════
# §7  BSM reach / exclusion limits
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ColliderReach:
    """Discovery/exclusion reach for a specific collider."""
    name: str
    sqrt_s_tev: float
    luminosity_ifb: float
    process: str
    m_reach_tev: float
    significance_sigma: float


def compute_reach(sqrt_s_tev: float = 14.0,
                  luminosity_ifb: float = 3000.0) -> list[ColliderReach]:
    """Estimate SDGFT discovery reach at a hadron collider.

    Returns reach estimates for different BSM signatures.
    """
    sqrt_s = sqrt_s_tev * 1000  # GeV

    reaches = []

    # 1. Graviton exchange in dileptons
    # σ_signal ~ α² / s · (√s / M_Pl)^{2(D*-2)}
    # For 5σ discovery: N_signal = 5 · √N_bkg
    sigma_grav = graviton_exchange_cross_section_fb(sqrt_s)
    n_signal = sigma_grav * luminosity_ifb
    # Background: SM Drell-Yan at high m_ll
    sigma_bkg_fb = 0.1  # rough: σ(DY, m_ll > 3 TeV) ~ 0.1 fb at 14 TeV
    n_bkg = sigma_bkg_fb * luminosity_ifb
    significance = n_signal / math.sqrt(max(n_bkg, 1.0)) if n_signal > 0 else 0

    # Effective reach: scale where significance = 5
    # M_reach ~ M_Pl · (5·√(σ_bkg·L) / (π·α²·L))^{1/(D*-2)} · √s^{...}
    # Simplified: find M_eff where A_grav gives 5σ
    if significance > 0:
        # A ~ (√s / M_eff)^(D*-2), need A² · πα²/s · L · gev2fb = 25 · √(σ_bkg·L)
        target_a2 = 25.0 * math.sqrt(sigma_bkg_fb * luminosity_ifb) / (
            math.pi * ALPHA_OBS ** 2 / (sqrt_s ** 2) * 0.3894e12 * luminosity_ifb
        )
        if target_a2 > 0:
            m_reach_gev = sqrt_s / target_a2 ** (0.5 / (D_STAR_TREE_F - 2.0))
            m_reach_tev = m_reach_gev / 1000.0
        else:
            m_reach_tev = 0.0
    else:
        m_reach_tev = 0.0

    reaches.append(ColliderReach(
        f"LHC-{sqrt_s_tev:.0f}",
        sqrt_s_tev, luminosity_ifb,
        "Graviton exchange (dileptons)",
        m_reach_tev, significance,
    ))

    # 2. Drell-Yan shape distortion
    # At m_ll ~ 5 TeV, the SDGFT correction to DY is:
    m_ll_probe = 5000.0  # GeV
    dy_ratio = drell_yan_ratio(m_ll_probe)
    dy_deviation_pct = (dy_ratio - 1.0) * 100
    reaches.append(ColliderReach(
        f"LHC-{sqrt_s_tev:.0f}",
        sqrt_s_tev, luminosity_ifb,
        f"Drell-Yan shape (m_ll={m_ll_probe/1000:.0f} TeV)",
        m_ll_probe / 1000.0, dy_deviation_pct,
    ))

    # 3. Higgs coupling modification
    higgs_mod = higgs_gg_modification()
    higgs_dev_pct = (higgs_mod - 1.0) * 100
    reaches.append(ColliderReach(
        f"LHC-{sqrt_s_tev:.0f}",
        sqrt_s_tev, luminosity_ifb,
        "Higgs gg→H coupling shift",
        0.125,  # m_H
        higgs_dev_pct * 1e6,  # in ppm
    ))

    return reaches


# ═══════════════════════════════════════════════════════════════════
# §8  Energy scan: running coupling trajectory
# ═══════════════════════════════════════════════════════════════════


def energy_scan(energies_gev: list[float] | None = None) -> list[dict]:
    """Compute SDGFT-modified observables at multiple collider energies.

    Default: logarithmic scan from M_Z to 10⁵ GeV.
    """
    if energies_gev is None:
        energies_gev = [M_Z * 10 ** (i * 0.1) for i in range(40)]

    results = []
    for e in energies_gev:
        r = sdgft_modified_running(e)
        r["drell_yan_ratio"] = drell_yan_ratio(e)
        r["graviton_amplitude"] = graviton_exchange_amplitude(e)
        results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════════
# §9  Module-level values
# ═══════════════════════════════════════════════════════════════════

HIGGS_GG_MOD: float = higgs_gg_modification()
"""σ(gg→H) SDGFT/SM ratio ≈ 1 + O(10⁻⁶)."""

HIGGS_WIDTH_MOD: float = higgs_width_modification()
"""Γ(H) SDGFT/SM ratio."""

DY_RATIO_3TEV: float = drell_yan_ratio(3000.0)
"""Drell-Yan ratio at 3 TeV."""

GRAV_AMP_14TEV: float = graviton_exchange_amplitude(14000.0)
"""Graviton exchange amplitude at √s = 14 TeV."""
