"""SDGFT Physics — Standalone analytic predictions from the 24-cell geometry.

Modules
-------
constants      – Level 0/1 axioms (Δ, δ, φ) and physical constants (SI)
dimension      – Effective dimension D*, f(R) exponent n
atomic         – Lamb shift, Rydberg correction, fine structure
qed            – Anomalous magnetic moments (g−2) for e, μ, τ
neutrino       – Mass spectrum, PMNS matrix, oscillation probabilities
black_holes    – Running G(k), Hawking temperature, QNM corrections
galaxy         – Rotation curves (Freeman disk), chameleon screening
rg_running     – SM 1-loop gauge coupling running, sin²θ_W
gauge_groups   – 24-cell ↔ D₄ root system, SM gauge group emergence
collider       – Collider signatures: running couplings, graviton exchange, KK modes
"""

from . import constants, dimension
