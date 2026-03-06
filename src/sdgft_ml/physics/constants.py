"""Level 0 axioms, Level 1 emergent constants, and physical constants (SI).

The entire SDGFT rests on two topological constants derived from
the 24-cell polytope (the unique self-dual regular 4D polytope):

    Δ = 5/24    Fibonacci-lattice conflict (F₅ = 5, capacity = 24)
    δ = 1/24    Elementary lattice tension (1 / vertex count)

From these, the golden ratio and the 6-cone geometry follow algebraically.
"""

from __future__ import annotations

import math
from fractions import Fraction

# ═══════════════════════════════════════════════════════════════════
# Level 0: Topological axioms
# ═══════════════════════════════════════════════════════════════════

DELTA = Fraction(5, 24)
"""Big Delta: Fibonacci-lattice conflict.  F₅ / 24 = 5/24."""

DELTA_F: float = float(DELTA)

DELTA_G = Fraction(1, 24)
"""Small delta (lattice tension): 1/24."""

DELTA_G_F: float = float(DELTA_G)

# ── Level 1: Emergent constants ───────────────────────────────────

PHI: float = (1.0 + math.sqrt(5.0)) / 2.0
"""Golden ratio φ = (1 + √5)/2 ≈ 1.61803."""

THETA_MAX: float = 30.0
"""Maximum cone half-opening angle in degrees."""

SIN2_30 = Fraction(1, 4)
SIN2_30_F: float = 0.25

COS2_30 = Fraction(3, 4)

# ── Import-time consistency ───────────────────────────────────────

assert DELTA + DELTA_G == SIN2_30
assert DELTA / DELTA_G == 5
assert abs(PHI - (1 + math.sqrt(float(DELTA / DELTA_G))) / 2) < 1e-15

# ═══════════════════════════════════════════════════════════════════
# Physical constants (SI, CODATA 2018)
# ═══════════════════════════════════════════════════════════════════

G_N: float = 6.67430e-11
"""Newtonian gravitational constant  [m³/(kg·s²)]."""

C: float = 2.99792458e8
"""Speed of light [m/s]."""

HBAR: float = 1.054571817e-34
"""Reduced Planck constant [J·s]."""

K_B: float = 1.380649e-23
"""Boltzmann constant [J/K]."""

# ── Planck units ──────────────────────────────────────────────────

M_P: float = math.sqrt(HBAR * C / G_N)
"""Planck mass [kg]."""

R_P: float = math.sqrt(HBAR * G_N / C ** 3)
"""Planck length [m]."""

T_P: float = math.sqrt(HBAR * G_N / C ** 5)
"""Planck time [s]."""

E_P: float = M_P * C ** 2
"""Planck energy [J]."""

K_P: float = 1.0 / R_P
"""Planck momentum scale [1/m]."""

M_PL_GEV: float = 1.2209e19
"""Planck mass [GeV]."""

# ── Astrophysical ─────────────────────────────────────────────────

R_H: float = 4.4e26
"""Hubble radius [m]  (c/H₀, H₀ ≈ 67.4 km/s/Mpc)."""

KPC_M: float = 3.0857e19
"""1 kiloparsec [m]."""

M_SUN: float = 1.989e30
"""Solar mass [kg]."""

# ── Conversions ───────────────────────────────────────────────────

GEV_TO_KG: float = 1.78266192e-27
GEV_TO_J: float = 1.602176634e-10

# ── Particle masses ───────────────────────────────────────────────

M_E_GEV: float = 0.000511
"""Electron mass [GeV]."""

M_E_MEV: float = 0.51099895000
"""Electron mass [MeV]."""

M_MU_MEV: float = 105.6583755
"""Muon mass [MeV]."""

M_TAU_MEV: float = 1776.86
"""Tau mass [MeV]."""

M_MU_OVER_M_E: float = 206.7682830
"""Muon-to-electron mass ratio (PDG 2024)."""

M_TAU_OVER_M_E: float = 3477.48
"""Tau-to-electron mass ratio (PDG 2024)."""

V_HIGGS_GEV: float = 246.22
"""Higgs VEV [GeV]."""

ALPHA_OBS: float = 1.0 / 137.035999177
"""Observed fine-structure constant (CODATA 2022)."""

ALPHA_INV_OBS: float = 137.035999177
"""Inverse fine-structure constant."""

# ── Rydberg ───────────────────────────────────────────────────────

R_INF_C_MHZ: float = 3_289_841_960.250
"""Rydberg frequency R∞·c [MHz]."""

R_INF_C_HZ: float = R_INF_C_MHZ * 1e6
"""Rydberg frequency R∞·c [Hz]."""
