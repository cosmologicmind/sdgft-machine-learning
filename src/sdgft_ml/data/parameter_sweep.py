"""Parametric forward model & grid sweep for SDGFT observables.

Re-computes every closed-form SDGFT observable for arbitrary
(Δ, δ_g, φ) input.  This is the **data generator** for ML training.

Usage::

    from sdgft_ml.data.parameter_sweep import sweep_grid, sweep_to_dataframe
    samples = sweep_grid(n_delta=50, n_delta_g=50)
    df = sweep_to_dataframe(samples)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from sdgft_ml.physics.constants import R_P as _R_P_DEFAULT


# ── Parametric Forward Model ─────────────────────────────────────

@dataclass
class ParametricForward:
    """Compute all SDGFT observables for given axiom parameters.

    This mirrors the SDGFT computation chain but accepts arbitrary
    (delta, delta_g, phi) instead of the hard-coded axiom values.

    Parameters
    ----------
    delta : float
        Fibonacci-lattice conflict (axiom: 5/24 ≈ 0.2083).
    delta_g : float
        Lattice tension (axiom: 1/24 ≈ 0.04167).
    phi : float
        Golden ratio (default: (1+√5)/2).
    gamma_ew : float
        Electroweak RG correction (external anchor: 0.12011).
    v_higgs : float
        Higgs VEV in GeV (external anchor: 246.22).
    """

    delta: float = 5.0 / 24.0
    delta_g: float = 1.0 / 24.0
    phi: float = (1.0 + math.sqrt(5.0)) / 2.0
    gamma_ew: float = 0.12011
    v_higgs: float = 246.22

    # Cached results
    _cache: dict[str, float] = field(default_factory=dict, repr=False)

    # ── Level 0-1: axiom-derived constants ────────────────────

    @property
    def sin2_30(self) -> float:
        return 0.25  # geometric constant

    @property
    def cos2_30(self) -> float:
        return 0.75

    @property
    def axiom_sum(self) -> float:
        """Δ + δ should equal sin²(30°) = 1/4 on the physical manifold."""
        return self.delta + self.delta_g

    # ── Level 2: effective dimension ──────────────────────────

    @property
    def d_star_tree(self) -> float:
        """D*_tree = 3 - sin²(30°) + δ_g = 11/4 + δ_g."""
        return 3.0 - self.sin2_30 + self.delta_g

    def compute_d_star_fp(
        self,
        d0: float = 3.0,
        tol: float = 1e-15,
        max_iter: int = 1000,
    ) -> float:
        """Fixed-point iteration: D_{k+1} = Δ^{-1/D_k} · φ · Δ^{Δ·δ_g}."""
        if "d_star_fp" in self._cache:
            return self._cache["d_star_fp"]
        d = d0
        correction = self.delta ** (self.delta * self.delta_g)
        for _ in range(max_iter):
            d_new = self.delta ** (-1.0 / d) * self.phi * correction
            if abs(d_new - d) < tol:
                break
            d = d_new
        self._cache["d_star_fp"] = d_new
        return d_new

    @property
    def d_star_fp(self) -> float:
        return self.compute_d_star_fp()

    @property
    def n_tree(self) -> float:
        """f(R) exponent: n = D*/2."""
        return self.d_star_tree / 2.0

    @property
    def n_fp(self) -> float:
        return self.d_star_fp / 2.0

    @property
    def two_n_minus_1_tree(self) -> float:
        return 2.0 * self.n_tree - 1.0

    @property
    def two_n_minus_1_fp(self) -> float:
        return 2.0 * self.n_fp - 1.0

    # ── Level 3: gravity ──────────────────────────────────────

    def alpha_m(self, n: float | None = None) -> float:
        """α_M = (n-1)/(2n-1)."""
        n = n if n is not None else self.n_tree
        return (n - 1.0) / (2.0 * n - 1.0)

    def alpha_b(self, n: float | None = None) -> float:
        """α_B = -α_M / 2."""
        return -self.alpha_m(n) / 2.0

    def grav_slip(self, n: float | None = None, k_over_aH: float = 10.0) -> float:
        """η = (1 + 2B·x²)/(1 + 4B·x²), B = n-1, x = k/(aH)."""
        n = n if n is not None else self.n_tree
        b = n - 1.0
        x2 = k_over_aH ** 2
        return (1.0 + 2.0 * b * x2) / (1.0 + 4.0 * b * x2)

    # ── Level 4: inflation ────────────────────────────────────

    def e_folds(self, d_star: float | None = None) -> float:
        """N_e = (D*/Δ) · ln[(D* - 2 - δ_g) / (Δ · δ_g)]."""
        ds = d_star if d_star is not None else self.d_star_fp
        arg = (ds - 2.0 - self.delta_g) / (self.delta * self.delta_g)
        if arg <= 0:
            return float("nan")
        return (ds / self.delta) * math.log(arg)

    def spectral_index(self, n: float | None = None, n_e: float | None = None) -> float:
        """n_s = 1 - 2(2n-1) / [N_e(2n-1) + n]."""
        n = n if n is not None else self.n_fp
        n_e = n_e if n_e is not None else self.e_folds()
        nbar = 2.0 * n - 1.0
        denom = n_e * nbar + n
        if denom == 0:
            return float("nan")
        return 1.0 - 2.0 * nbar / denom

    def tensor_to_scalar(self, n: float | None = None, n_e: float | None = None) -> float:
        """r = 48(2n-1)² / [N_e(2n-1) + n]²."""
        n = n if n is not None else self.n_fp
        n_e = n_e if n_e is not None else self.e_folds()
        nbar = 2.0 * n - 1.0
        denom = n_e * nbar + n
        if denom == 0:
            return float("nan")
        return 48.0 * nbar ** 2 / denom ** 2

    def slow_roll_epsilon(self, n: float | None = None, n_e: float | None = None) -> float:
        """ε = (2n-1)(n-2)² / [N_e(2n-1) + n]²."""
        n = n if n is not None else self.n_fp
        n_e = n_e if n_e is not None else self.e_folds()
        nbar = 2.0 * n - 1.0
        denom = n_e * nbar + n
        if denom == 0:
            return float("nan")
        return nbar * (n - 2.0) ** 2 / denom ** 2

    def slow_roll_eta(self, n: float | None = None, n_e: float | None = None) -> float:
        """η_SR = (2n-1)(2n²-7n+4)/[N_e(2n-1)+n]² - (2n-1)/[N_e(2n-1)+n]."""
        n = n if n is not None else self.n_fp
        n_e = n_e if n_e is not None else self.e_folds()
        nbar = 2.0 * n - 1.0
        denom = n_e * nbar + n
        if denom == 0:
            return float("nan")
        return nbar * (2.0 * n ** 2 - 7.0 * n + 4.0) / denom ** 2 - nbar / denom

    def tensor_spectral_index(self, n: float | None = None, n_e: float | None = None) -> float:
        """Tensor spectral index n_T = -2ε_SR (consistency relation).

        For single-field slow-roll inflation the consistency relation
        r = -8 n_T holds.  Since ε_SR is already computed, this is trivial.
        """
        return -2.0 * self.slow_roll_epsilon(n, n_e)

    @property
    def beta_iso(self) -> float:
        """β_iso = (1/6)² = 1/36 (geometric constant)."""
        return 1.0 / 36.0

    # ── Level 5-6: cosmology ──────────────────────────────────

    @property
    def omega_b(self) -> float:
        """Ω_b = (Δ/4)(1 - δ_g)."""
        return (self.delta / 4.0) * (1.0 - self.delta_g)

    @property
    def omega_c(self) -> float:
        """Ω_c = 6Δ² (lattice-closure prediction)."""
        return 6.0 * self.delta ** 2

    @property
    def omega_de(self) -> float:
        """Ω_DE = 1 - Ω_b - Ω_c (flatness closure)."""
        return 1.0 - self.omega_b - self.omega_c

    @property
    def omega_m(self) -> float:
        """Ω_m = Ω_b + Ω_c."""
        return self.omega_b + self.omega_c

    def w_de(self, d_star: float | None = None) -> float:
        """w_DE = -D*/3."""
        ds = d_star if d_star is not None else self.d_star_fp
        return -ds / 3.0

    @property
    def eta_b(self) -> float:
        """η_B = δ_g⁶ · (1-δ_g) / 8."""
        return self.delta_g ** 6 * (1.0 - self.delta_g) / 8.0

    @property
    def sigma_8(self) -> float:
        """σ_8 = Δ·π ≈ 0.6545 (analytical derivation).

        The monograph identifies σ_8 = Δπ as the analytical prediction.
        Previously hardcoded as 0.775 (MCMC anchor); now derived from
        the axiom parameter Δ.
        """
        return self.delta * math.pi

    @property
    def s_8(self) -> float:
        """S_8 = σ_8 · √(Ω_m / 0.3)."""
        return self.sigma_8 * math.sqrt(self.omega_m / 0.3)

    @property
    def m_dm(self) -> float:
        """Effective DM mass ~ 10^{-22} eV (qualitative)."""
        return 1e-22

    # ── Level 5-6: particle physics ───────────────────────────

    def alpha_em_inv(self, d_star: float | None = None) -> float:
        """α_em⁻¹ = 2π D*³ + δ_g · D*."""
        ds = d_star if d_star is not None else self.d_star_tree
        return 2.0 * math.pi * ds ** 3 + self.delta_g * ds

    def alpha_em(self, d_star: float | None = None) -> float:
        """α_em = 1 / α_em⁻¹."""
        return 1.0 / self.alpha_em_inv(d_star)

    @property
    def alpha_s(self) -> float:
        """α_s(M_Z) = √2/12 (geometric constant)."""
        return math.sqrt(2.0) / 12.0

    @property
    def sin2_theta_w(self) -> float:
        """sin²θ_W = 1/9 + γ_EW."""
        return 1.0 / 9.0 + self.gamma_ew

    def mu_e_ratio(self, d_star: float | None = None) -> float:
        """m_μ/m_e = 3/(2α_em) + 1 + Δ."""
        ae = self.alpha_em(d_star)
        return 3.0 / (2.0 * ae) + 1.0 + self.delta

    def tau_mu_ratio(self, d_star: float | None = None) -> float:
        """m_τ/m_μ = 6 D*."""
        ds = d_star if d_star is not None else self.d_star_tree
        return 6.0 * ds

    def tau_e_ratio(self, d_star: float | None = None) -> float:
        """m_τ/m_e = (m_μ/m_e)(m_τ/m_μ)."""
        return self.mu_e_ratio(d_star) * self.tau_mu_ratio(d_star)

    @property
    def lambda_geo(self) -> float:
        """λ_geo = Δ/φ."""
        return self.delta / self.phi

    @property
    def higgs_mass(self) -> float:
        """m_H = √(2λ_geo) · v_Higgs."""
        return math.sqrt(2.0 * self.lambda_geo) * self.v_higgs

    @property
    def n_generations(self) -> int:
        """N_gen = max{n : φ^n < Δ/δ_g}."""
        if self.delta_g == 0:
            return 0
        ratio = self.delta / self.delta_g
        n = 0
        while self.phi ** (n + 1) < ratio:
            n += 1
            if n > 10:
                break
        return n

    def theta_12(self) -> float:
        """θ₁₂ = arctan(1/√2) · (1 - δ_g) [degrees]."""
        tbm = math.atan(1.0 / math.sqrt(2.0))
        return math.degrees(tbm * (1.0 - self.delta_g))

    def theta_23(self) -> float:
        """θ₂₃ = 45(1 + Δ/√6) [degrees]."""
        return 45.0 * (1.0 + self.delta / math.sqrt(6.0))

    def theta_13(self) -> float:
        """θ₁₃ = arcsin(Δ/√2) [degrees]."""
        arg = self.delta / math.sqrt(2.0)
        if abs(arg) > 1.0:
            return float("nan")
        return math.degrees(math.asin(arg))

    def v_us(self) -> float:
        """|V_us| = √Ω_b."""
        return math.sqrt(self.omega_b)

    def v_ub(self) -> float:
        """|V_ub| = Δ^φ · δ_g · exp(δ_g · ln(m_τ/m_e) / φ²)."""
        tau_e = self.tau_e_ratio()
        if tau_e <= 0:
            return float("nan")
        return (
            self.delta ** self.phi
            * self.delta_g
            * math.exp(self.delta_g * math.log(tau_e) / self.phi ** 2)
        )

    @property
    def quark_hierarchy(self) -> float:
        """m_c/m_u ≈ exp(2π) (confinement-scale constant)."""
        return math.exp(2.0 * math.pi)

    # ── Scale-dependent functions (sampled at characteristic scales) ──

    def d_star_of_r(self, r: float, r_p: float = _R_P_DEFAULT) -> float:
        """D*(r) = D*_IR · (r/r_P)^{-Δ²}."""
        if r <= 0 or r_p <= 0:
            return float("nan")
        return self.d_star_tree * (r / r_p) ** (-self.delta ** 2)

    def omega_de_rg(self, r: float, r_p: float = _R_P_DEFAULT) -> float:
        """Ω_DE(r) = (3/4) · (r/r_P)^{-δ_g²/D*}."""
        ds = self.d_star_of_r(r, r_p)
        if ds == 0:
            return float("nan")
        return 0.75 * (r / r_p) ** (-self.delta_g ** 2 / ds)

    # ── Collect all observables ───────────────────────────────

    def compute_all(self) -> dict[str, float]:
        """Compute all observables and return as a flat dict.

        The keys match the SDGFT registry names where possible.
        """
        self._cache.clear()

        d_star_t = self.d_star_tree
        d_star_f = self.d_star_fp
        n_t = self.n_tree
        n_f = self.n_fp
        n_e_f = self.e_folds(d_star_f)
        n_e_t = self.e_folds(d_star_t)

        return {
            # Input parameters
            "param_delta": self.delta,
            "param_delta_g": self.delta_g,
            "param_phi": self.phi,
            "param_gamma_ew": self.gamma_ew,
            "param_v_higgs": self.v_higgs,
            # Level 2: dimension
            "d_star_tree": d_star_t,
            "d_star_fp": d_star_f,
            "n_tree": n_t,
            "n_fp": n_f,
            "two_n_minus_1_tree": self.two_n_minus_1_tree,
            "two_n_minus_1_fp": self.two_n_minus_1_fp,
            # Level 3: gravity
            "alpha_m_tree": self.alpha_m(n_t),
            "alpha_b_tree": self.alpha_b(n_t),
            "alpha_t": 0.0,   # c_T = c exactly: f(R) ⊂ Horndeski with G₅=0 ⇒ α_T=0
            "alpha_k": 0.0,
            "eta_slip_subhorizon": self.grav_slip(n_t, k_over_aH=1e6),
            "eta_slip_survey": self.grav_slip(n_t, k_over_aH=10.0),
            "eta_slip_horizon": self.grav_slip(n_t, k_over_aH=1.0),
            # Level 4: inflation
            "n_efolds_fp": n_e_f,
            "n_efolds_tree": n_e_t,
            "n_s": self.spectral_index(n_f, n_e_f),
            "r_tensor": self.tensor_to_scalar(n_f, n_e_f),
            "beta_iso": self.beta_iso,
            "epsilon_sr": self.slow_roll_epsilon(n_f, n_e_f),
            "n_t": self.tensor_spectral_index(n_f, n_e_f),
            "eta_sr": self.slow_roll_eta(n_f, n_e_f),
            # Level 5-6: cosmology
            "omega_b": self.omega_b,
            "omega_c": self.omega_c,
            "omega_de": self.omega_de,
            "omega_m": self.omega_m,
            "w_de_tree": self.w_de(d_star_t),
            "w_de_fp": self.w_de(d_star_f),
            "eta_b": self.eta_b,
            "sigma_8": self.sigma_8,
            "s_8": self.s_8,
            "m_dm": self.m_dm,
            # Level 5-6: particle physics
            "alpha_em_inv_tree": self.alpha_em_inv(d_star_t),
            "alpha_em_inv_fp": self.alpha_em_inv(d_star_f),
            "alpha_em_tree": self.alpha_em(d_star_t),
            "alpha_s": self.alpha_s,
            "sin2_theta_w": self.sin2_theta_w,
            "mu_e_ratio": self.mu_e_ratio(d_star_t),
            "tau_mu_ratio_tree": self.tau_mu_ratio(d_star_t),
            "tau_mu_ratio_fp": self.tau_mu_ratio(d_star_f),
            "tau_e_ratio_tree": self.tau_e_ratio(d_star_t),
            "lambda_geo": self.lambda_geo,
            "higgs_mass": self.higgs_mass,
            "n_generations": float(self.n_generations),
            "theta_12": self.theta_12(),
            "theta_23": self.theta_23(),
            "theta_13": self.theta_13(),
            "v_us": self.v_us(),
            "v_ub": self.v_ub(),
            "quark_hierarchy": self.quark_hierarchy,
        }

    # ── Feature vector (ordered, numeric-only) ────────────────

    # Observable keys for ML (excludes input params and constants)
    OBSERVABLE_KEYS: ClassVar[list[str]] = [
        "d_star_tree", "d_star_fp", "n_tree", "n_fp",
        "alpha_m_tree", "alpha_b_tree",
        "eta_slip_survey", "eta_slip_horizon",
        "n_efolds_fp", "n_s", "r_tensor", "beta_iso",
        "epsilon_sr", "eta_sr",
        "omega_b", "omega_c", "omega_de", "omega_m",
        "w_de_fp", "eta_b", "s_8",
        "alpha_em_inv_tree", "alpha_em_inv_fp",
        "alpha_em_tree", "alpha_s", "sin2_theta_w",
        "mu_e_ratio", "tau_mu_ratio_tree",
        "lambda_geo", "higgs_mass",
        "n_generations",
        "theta_12", "theta_23", "theta_13",
        "v_us", "v_ub", "quark_hierarchy",
    ]

    PARAM_KEYS: ClassVar[list[str]] = [
        "param_delta", "param_delta_g", "param_phi",
    ]

    def feature_vector(self) -> np.ndarray:
        """Return observable values as a 1-D numpy array (stable order)."""
        result = self.compute_all()
        return np.array([result[k] for k in self.OBSERVABLE_KEYS], dtype=np.float64)

    def param_vector(self) -> np.ndarray:
        """Return input parameters as a 1-D numpy array."""
        return np.array([self.delta, self.delta_g, self.phi], dtype=np.float64)


# ── Grid Sweep ────────────────────────────────────────────────────

def sweep_grid(
    n_delta: int = 50,
    n_delta_g: int = 50,
    delta_range: tuple[float, float] = (0.05, 0.40),
    delta_g_range: tuple[float, float] = (0.01, 0.08),
    phi: float = (1.0 + math.sqrt(5.0)) / 2.0,
    constrained: bool = False,
) -> list[dict[str, float]]:
    """Generate a grid of SDGFT observables over (Δ, δ_g) space.

    Parameters
    ----------
    n_delta, n_delta_g : int
        Grid resolution per axis.
    delta_range, delta_g_range : tuple
        Bounds for Δ and δ_g.
    phi : float
        Golden ratio (held fixed by default).
    constrained : bool
        If True, enforce axiom Δ + δ_g = 1/4.
        In that case only n_delta is used and δ_g = 1/4 - Δ.

    Returns
    -------
    list of dicts, each containing params + all observables.
    """
    samples: list[dict[str, float]] = []

    if constrained:
        deltas = np.linspace(*delta_range, n_delta)
        for d in deltas:
            dg = 0.25 - d
            if dg <= 0:
                continue
            fwd = ParametricForward(delta=float(d), delta_g=float(dg), phi=phi)
            try:
                row = fwd.compute_all()
                if any(math.isnan(v) for v in row.values() if isinstance(v, float)):
                    continue
                samples.append(row)
            except (ValueError, ZeroDivisionError, OverflowError):
                continue
    else:
        deltas = np.linspace(*delta_range, n_delta)
        delta_gs = np.linspace(*delta_g_range, n_delta_g)
        for d in deltas:
            for dg in delta_gs:
                fwd = ParametricForward(delta=float(d), delta_g=float(dg), phi=phi)
                try:
                    row = fwd.compute_all()
                    if any(
                        math.isnan(v) for v in row.values() if isinstance(v, float)
                    ):
                        continue
                    samples.append(row)
                except (ValueError, ZeroDivisionError, OverflowError):
                    continue

    return samples


def sweep_to_dataframe(samples: list[dict[str, float]]) -> pd.DataFrame:
    """Convert sweep results to a DataFrame."""
    return pd.DataFrame(samples)


def sweep_constrained(
    n_points: int = 200,
    delta_range: tuple[float, float] = (0.05, 0.24),
) -> pd.DataFrame:
    """1-D sweep along the axiom manifold Δ + δ_g = 1/4."""
    samples = sweep_grid(
        n_delta=n_points,
        delta_range=delta_range,
        constrained=True,
    )
    return sweep_to_dataframe(samples)


def sweep_latin_hypercube(
    n_samples: int = 1000,
    delta_range: tuple[float, float] = (0.05, 0.40),
    delta_g_range: tuple[float, float] = (0.01, 0.08),
    phi_range: tuple[float, float] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Latin-hypercube sampling for efficient space coverage.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    phi_range : tuple | None
        If None, φ is held fixed at the golden ratio.
        If provided, φ is varied in the given range.
    seed : int
        RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    phi_default = (1.0 + math.sqrt(5.0)) / 2.0

    n_dim = 2 if phi_range is None else 3

    # Latin-hypercube: stratified random in each dimension
    intervals = np.arange(n_samples) / n_samples
    points = np.empty((n_samples, n_dim))
    for i in range(n_dim):
        perm = rng.permutation(n_samples)
        points[:, i] = intervals[perm] + rng.uniform(0, 1.0 / n_samples, n_samples)

    # Scale to parameter ranges
    deltas = delta_range[0] + points[:, 0] * (delta_range[1] - delta_range[0])
    delta_gs = delta_g_range[0] + points[:, 1] * (delta_g_range[1] - delta_g_range[0])
    if phi_range is not None:
        phis = phi_range[0] + points[:, 2] * (phi_range[1] - phi_range[0])
    else:
        phis = np.full(n_samples, phi_default)

    samples: list[dict[str, float]] = []
    for d, dg, p in zip(deltas, delta_gs, phis):
        fwd = ParametricForward(delta=float(d), delta_g=float(dg), phi=float(p))
        try:
            row = fwd.compute_all()
            if any(math.isnan(v) for v in row.values() if isinstance(v, float)):
                continue
            samples.append(row)
        except (ValueError, ZeroDivisionError, OverflowError):
            continue

    return sweep_to_dataframe(samples)
