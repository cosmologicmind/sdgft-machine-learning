"""Validation against real experimental data.

Compares SDGFT predictions (at the axiom point or arbitrary parameters)
against published PDG (Particle Data Group) and Planck 2018/2020 values
to produce a physics scorecard.

Usage::

    from sdgft_ml.training.validate_real import (
        EXPERIMENTAL_DATA,
        validate_at_axiom,
        scorecard,
    )
    results = validate_at_axiom()
    scorecard(results)

Data sources
-----------
- PDG 2024: https://pdg.lbl.gov
- Planck 2018 (TT,TE,EE+lowE+lensing): arXiv:1807.06209
- Planck 2020 (NPIPE): arXiv:2007.04997
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..data.parameter_sweep import ParametricForward


# ── Experimental reference data ───────────────────────────────────

@dataclass(frozen=True)
class ExperimentalValue:
    """A single experimental measurement with uncertainty."""
    name: str
    value: float
    sigma: float          # 1σ total uncertainty
    unit: str = ""
    source: str = ""
    category: str = ""    # "cosmology", "particle", "gravity", "inflation"
    theory_sigma: float = 0.0  # tree-level theory uncertainty (used if > sigma)


# Comprehensive experimental reference table
EXPERIMENTAL_DATA: dict[str, ExperimentalValue] = {
    # ── Cosmological parameters (Planck 2018 TT,TE,EE+lowE+lensing) ──
    "omega_b": ExperimentalValue(
        name="Ω_b h² → Ω_b",
        value=0.0493,       # Ω_b = ω_b/h² ≈ 0.02237/0.6736²
        sigma=0.0020,
        source="Planck 2018",
        category="cosmology",
    ),
    "omega_c": ExperimentalValue(
        name="Ω_c",
        value=0.265,        # Ω_c = ω_c/h² ≈ 0.1200/0.6736²
        sigma=0.007,
        source="Planck 2018",
        category="cosmology",
    ),
    "omega_m": ExperimentalValue(
        name="Ω_m",
        value=0.3153,
        sigma=0.0073,
        source="Planck 2018",
        category="cosmology",
    ),
    "omega_de": ExperimentalValue(
        name="Ω_Λ",
        value=0.6847,
        sigma=0.0073,
        source="Planck 2018",
        category="cosmology",
    ),
    "s_8": ExperimentalValue(
        name="S_8",
        value=0.832,
        sigma=0.013,
        source="Planck 2018",
        category="cosmology",
    ),
    "eta_b": ExperimentalValue(
        name="η_B (baryon asymmetry)",
        value=6.143e-10,
        sigma=0.190e-10,
        source="Planck 2018 + BBN",
        category="cosmology",
    ),
    # ── Inflationary observables ──
    "n_s": ExperimentalValue(
        name="n_s (scalar spectral index)",
        value=0.9649,
        sigma=0.0042,
        source="Planck 2018",
        category="inflation",
    ),
    "r_tensor": ExperimentalValue(
        name="r (tensor-to-scalar ratio)",
        value=0.0,          # upper limit r < 0.036 (BICEP/Keck 2021)
        sigma=0.036,        # using 95% CL as ~2σ → σ≈0.018
        source="BICEP/Keck 2021 (upper limit)",
        category="inflation",
    ),
    # ── Particle physics (PDG 2024) ──
    "alpha_em_inv_tree": ExperimentalValue(
        name="α_em⁻¹(0)",
        value=137.035999177,
        sigma=0.000000021,
        source="PDG 2024 (CODATA)",
        category="particle",
        theory_sigma=0.5,  # tree-level: ~0.4% loop corrections expected
    ),
    "alpha_s": ExperimentalValue(
        name="α_s(M_Z)",
        value=0.1180,
        sigma=0.0009,
        source="PDG 2024",
        category="particle",
    ),
    "sin2_theta_w": ExperimentalValue(
        name="sin²θ_W (MS-bar)",
        value=0.23122,
        sigma=0.00003,
        source="PDG 2024",
        category="particle",
    ),
    "higgs_mass": ExperimentalValue(
        name="m_H",
        value=125.25,
        sigma=0.17,
        unit="GeV",
        source="PDG 2024 (ATLAS+CMS)",
        category="particle",
    ),
    "mu_e_ratio": ExperimentalValue(
        name="m_μ/m_e",
        value=206.7682830,
        sigma=0.0000046,
        source="PDG 2024 (CODATA)",
        category="particle",
        theory_sigma=1.0,  # tree-level geometric ratio; ~0.5% radiative corrections
    ),
    "tau_mu_ratio_tree": ExperimentalValue(
        name="m_τ/m_μ",
        value=16.8170,
        sigma=0.0015,
        source="PDG 2024",
        category="particle",
        theory_sigma=0.1,  # tree-level geometric ratio
    ),
    "n_generations": ExperimentalValue(
        name="N_gen (fermion generations)",
        value=3.0,
        sigma=0.008,        # LEP: N_ν = 2.9840 ± 0.0082
        source="LEP (Z width)",
        category="particle",
    ),
    "theta_12": ExperimentalValue(
        name="θ₁₂ (solar mixing angle)",
        value=33.44,
        sigma=0.77,
        unit="deg",
        source="NuFIT 5.3 (2024)",
        category="particle",
    ),
    "theta_23": ExperimentalValue(
        name="θ₂₃ (atmospheric mixing angle)",
        value=49.2,
        sigma=1.0,
        unit="deg",
        source="NuFIT 5.3 (2024)",
        category="particle",
    ),
    "theta_13": ExperimentalValue(
        name="θ₁₃ (reactor mixing angle)",
        value=8.57,
        sigma=0.12,
        unit="deg",
        source="NuFIT 5.3 (2024)",
        category="particle",
    ),
    "v_us": ExperimentalValue(
        name="|V_us|",
        value=0.2243,
        sigma=0.0005,
        source="PDG 2024 (CKM)",
        category="particle",
    ),
    "v_ub": ExperimentalValue(
        name="|V_ub|",
        value=0.00382,
        sigma=0.00020,
        source="PDG 2024 (CKM)",
        category="particle",
    ),
    "lambda_geo": ExperimentalValue(
        name="λ (Higgs quartic coupling)",
        value=0.1291,       # from m_H=125.25, v=246.22 → λ = m²/(2v²)
        sigma=0.0020,
        source="Derived from PDG m_H, v",
        category="particle",
    ),
    # ── Gravity sector ──
    # Modified gravity: current constraints on Horndeski α_M, α_B
    # GW170817 constrains |c_T - 1| < 6e-16 → strong limit on α_T
    # But α_M, α_B are less constrained. DES Y1+Planck: α_M0 < 0.7 (95%)
    "w_de_fp": ExperimentalValue(
        name="w_DE (equation of state)",
        value=-1.03,
        sigma=0.03,
        source="Planck 2018 + BAO + SN",
        category="cosmology",
    ),
}


# ── Validation functions ──────────────────────────────────────────

def validate_at_point(
    delta: float,
    delta_g: float,
    phi: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute theory predictions at a parameter point and compare to data.

    Parameters
    ----------
    delta, delta_g : float
        SDGFT axiom parameters.
    phi : float | None
        Golden ratio (default: (1+√5)/2).

    Returns
    -------
    Dict mapping observable name → {theory, experiment, sigma, pull, tension}
    """
    if phi is None:
        phi = (1.0 + math.sqrt(5.0)) / 2.0

    fwd = ParametricForward(delta=delta, delta_g=delta_g, phi=phi)
    theory = fwd.compute_all()

    results: dict[str, dict[str, Any]] = {}
    for key, exp in EXPERIMENTAL_DATA.items():
        if key not in theory:
            continue
        th_val = theory[key]
        eff_sigma = max(exp.sigma, exp.theory_sigma)  # use larger of exp/theory σ
        pull = (th_val - exp.value) / eff_sigma if eff_sigma > 0 else float("inf")
        tension_sigma = abs(pull)
        status = (
            "excellent" if tension_sigma < 1.0
            else "good" if tension_sigma < 2.0
            else "tension" if tension_sigma < 3.0
            else "FAIL"
        )
        results[key] = {
            "theory": th_val,
            "experiment": exp.value,
            "exp_sigma": eff_sigma,
            "pull": pull,
            "tension_sigma": tension_sigma,
            "status": status,
            "source": exp.source,
            "category": exp.category,
            "name": exp.name,
        }

    return results


def validate_at_axiom() -> dict[str, dict[str, Any]]:
    """Validate SDGFT at the canonical axiom point (Δ=5/24, δ_g=1/24, φ=golden)."""
    return validate_at_point(5.0 / 24.0, 1.0 / 24.0)


def chi_squared(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Compute χ² from validation results.

    Returns
    -------
    dict with 'chi2', 'ndof', 'chi2_per_dof', 'p_value', 'per_category'
    """
    from scipy import stats

    pulls = [r["pull"] for r in results.values()]
    chi2 = sum(p**2 for p in pulls)
    ndof = len(pulls)

    # Per-category breakdown
    categories: dict[str, list[float]] = {}
    for r in results.values():
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r["pull"] ** 2)

    per_cat = {}
    for cat, chi2_list in categories.items():
        n = len(chi2_list)
        c = sum(chi2_list)
        per_cat[cat] = {
            "chi2": c,
            "ndof": n,
            "chi2_per_dof": c / n if n > 0 else 0,
        }

    return {
        "chi2": chi2,
        "ndof": ndof,
        "chi2_per_dof": chi2 / ndof if ndof > 0 else 0,
        "p_value": 1.0 - stats.chi2.cdf(chi2, ndof),
        "per_category": per_cat,
    }


def scorecard(
    results: dict[str, dict[str, Any]],
    title: str = "SDGFT Physics Scorecard",
) -> None:
    """Pretty-print a physics scorecard.

    Parameters
    ----------
    results : output of validate_at_point / validate_at_axiom
    """
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

    # Group by category
    by_cat: dict[str, list[tuple[str, dict]]] = {}
    for key, r in results.items():
        cat = r["category"]
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append((key, r))

    total_excellent = total_good = total_tension = total_fail = 0

    for cat in ["cosmology", "inflation", "particle", "gravity"]:
        items = by_cat.get(cat, [])
        if not items:
            continue
        print(f"\n  ── {cat.upper()} ──")
        print(f"  {'Observable':<28s} {'Theory':>12s} {'Experiment':>12s} "
              f"{'Pull (σ)':>8s} {'Status':>10s}")
        print(f"  {'-'*74}")

        for key, r in sorted(items, key=lambda x: abs(x[1]["pull"]), reverse=True):
            status_sym = {
                "excellent": "  ✓",
                "good": " ~✓",
                "tension": " ⚠",
                "FAIL": "  ✗",
            }[r["status"]]
            print(
                f"  {r['name']:<28s} {r['theory']:>12.6g} {r['experiment']:>12.6g} "
                f"{r['pull']:>+8.2f} {status_sym:>10s}"
            )
            if r["status"] == "excellent":
                total_excellent += 1
            elif r["status"] == "good":
                total_good += 1
            elif r["status"] == "tension":
                total_tension += 1
            else:
                total_fail += 1

    n = len(results)
    print(f"\n{'='*80}")
    print(f"  TOTAL: {n} observables")
    print(f"    Excellent (<1σ):  {total_excellent}")
    print(f"    Good      (<2σ):  {total_good}")
    print(f"    Tension   (<3σ):  {total_tension}")
    print(f"    FAIL      (≥3σ):  {total_fail}")

    pulls = [r["pull"] for r in results.values()]
    chi2 = sum(p**2 for p in pulls)
    print(f"    χ² = {chi2:.2f}  (N_dof = {n}, χ²/N = {chi2/n:.2f})")
    print(f"{'='*80}")


def validate_surrogate_vs_real(
    model: Any,
    edge_index: Any,
    device: str = "cpu",
    norm_mean: np.ndarray | None = None,
    norm_std: np.ndarray | None = None,
) -> dict[str, dict[str, Any]]:
    """Run the ML surrogate at axiom point and compare to real data.

    This is the final-stage validation: ML model → experimental data.

    Parameters
    ----------
    model : SurrogateGNN
    edge_index : DAG edge index
    norm_mean, norm_std : normalization arrays

    Returns
    -------
    Same format as validate_at_point, but with 'ml_prediction' added.
    """
    import torch
    from ..data.dag_builder import observable_names

    delta = 5.0 / 24.0
    delta_g = 1.0 / 24.0
    phi = (1.0 + math.sqrt(5.0)) / 2.0

    model.eval()
    model = model.to(device)

    if isinstance(edge_index, np.ndarray):
        edge_index = torch.from_numpy(edge_index)
    ei = edge_index.to(device)

    with torch.no_grad():
        params = torch.tensor([delta, delta_g, phi], dtype=torch.float32).unsqueeze(0).to(device)
        pred = model(params, ei).cpu().numpy()

    if norm_mean is not None and norm_std is not None:
        pred = pred * norm_std + norm_mean

    names = observable_names()
    ml_preds = {n: pred[i] for i, n in enumerate(names)}

    # Get theory predictions too
    results = validate_at_axiom()

    for key in results:
        if key in ml_preds:
            results[key]["ml_prediction"] = float(ml_preds[key])
            exp = EXPERIMENTAL_DATA[key]
            eff_sigma = max(exp.sigma, exp.theory_sigma)
            ml_pull = (ml_preds[key] - exp.value) / eff_sigma if eff_sigma > 0 else 0
            results[key]["ml_pull"] = float(ml_pull)

    return results
