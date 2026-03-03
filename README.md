# SDGFT-ML

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org)

Machine-learning surrogate, inverter & anomaly detector for
**Six-Dimensional Geometric Field Theory (SDGFT)**.

> **0 free parameters · 22+ observables · χ²/ndof = 1.48 · R² = 0.9995 round-trip**

---

## Highlights

| Metric | Value |
|--------|-------|
| Surrogate accuracy (canary) | 100 % observables within 5 % |
| Round-trip R² (Δ, δ_g) | 0.9995 |
| PDG 2024 scorecard | 20/22 within 2σ, χ²/ndof = 1.48 |
| Neutrino mass ratio R | 33.5 vs 33.6 ± 0.9 (−0.08σ) |
| Baryon asymmetry η_B | +0.66σ from Planck+BBN |
| S₈ tension | Resolved — 1.2σ vs KiDS weak lensing |
| SPARC Tully-Fisher slope | b = 3.79 (91/24), 1.7σ from ODR fit |
| Dark energy EoS | w_DE = −0.932, consistent with DESI 2024 |

## Overview

SDGFT derives all Standard Model + cosmological observables from **two axiom
parameters** (Δ = 5/24, δ_g = 1/24) plus the golden ratio φ. This project
builds a complete ML pipeline around the SDGFT computation DAG:

1. **Surrogate GNN** — GATv2-based graph neural network that mirrors the
   SDGFT computation DAG and predicts all 37 observables for arbitrary (Δ, δ_g, φ)
2. **Inverter CVAE** — Conditional Variational Autoencoder that solves the
   inverse problem: observables → axiom parameters
3. **Ensemble** — 5-member deep ensemble with calibrated uncertainty
4. **Active Learning** — Uncertainty-guided sampling to improve the surrogate
5. **Anomaly Detection** — Autoencoder on prediction residuals
6. **Experimental Validation** — 7 tests against Planck, PDG, NuFIT, SPARC, DESI, KiDS

### Pipeline Phases

| Phase | Description | Commit |
|-------|-------------|--------|
| **A** | Surrogate GNN + Inverter CVAE + canary tests | `504a654` |
| **B** | Surrogate v2 (100 % canary), CVAE v2, real-data validation | `5349d72` |
| **C** | 5-member ensemble, sensitivity analysis, round-trip pipeline | `0684a7d` |
| **D** | Enhanced surrogate (8.3× better), matched normalization, R² = 0.9995 | `7919ad1` |
| **E** | 7 experimental validation tests against real data | `364834a` |

## Project Structure

```
sdgft-machine-learning/
├── LICENSE
├── CITATION.cff
├── README.md
├── pyproject.toml
├── .zenodo.json
├── .gitignore
├── src/sdgft_ml/
│   ├── __init__.py
│   ├── data/
│   │   ├── registry_export.py      # Export SDGFT observables
│   │   ├── parameter_sweep.py      # ParametricForward model & Latin hypercube sweep
│   │   └── dag_builder.py          # DAG → PyG graph structure
│   ├── models/
│   │   ├── surrogate_gnn.py        # GATv2Conv surrogate (mirrors computation DAG)
│   │   ├── inverter.py             # Conditional VAE for inverse problem
│   │   └── curve_net.py            # DeepONet for scale-dependent functions
│   ├── training/
│   │   ├── train_surrogate.py      # GNN training loop with canary test
│   │   ├── train_inverter.py       # CVAE training loop
│   │   ├── train_ensemble.py       # Deep ensemble training (5 members)
│   │   ├── evaluate.py             # Evaluation metrics & canary test
│   │   ├── validate_real.py        # 22 experimental data points + χ² scoring
│   │   ├── sensitivity.py          # Jacobian-based sensitivity analysis
│   │   └── round_trip.py           # Forward → Inverse → Forward consistency
│   ├── loop/
│   │   ├── active_learner.py       # Uncertainty-guided sampling
│   │   ├── anomaly_detector.py     # Residual autoencoder
│   │   └── distillation.py         # Symbolic regression (PySR)
│   └── api/
│       ├── predict.py              # CLI tool
│       └── visualize.py            # Plotting utilities
├── tests/
│   ├── test_data.py                # Parameter sweep & DAG tests
│   ├── test_models.py              # Model architecture tests
│   └── test_training.py            # Training loop tests
├── notebooks/
│   ├── train_and_explore.ipynb     # Full pipeline: Phases A–E (interactive)
│   └── runs/                       # Trained model checkpoints
│       ├── ensemble/               # 5 ensemble members (~3.3 MB each)
│       ├── surrogate_v3/           # Best surrogate checkpoint
│       ├── inverter_v3/            # Best inverter checkpoint
│       └── inverter/               # Phase D inverter
└── data/
    └── .gitkeep
```

## Installation

```bash
# Clone
git clone https://github.com/cosmologicmind/sdgft-machine-learning.git
cd sdgft-machine-learning

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install with dev dependencies
pip install -e ".[dev]"

# Install SDGFT core library (required dependency)
# Option A: from local checkout
pip install -e /path/to/SDGFT_V2
# Option B: from GitHub (when available)
# pip install git+https://github.com/cosmologicmind/SDGFT_V2.git

# Verify installation
pytest
```

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1 (CUDA recommended for training)
- PyTorch Geometric ≥ 2.4
- SDGFT_V2 (core theory library)

## Quick Start

### Compute Observables

```python
from sdgft_ml.data.parameter_sweep import ParametricForward

# At the SDGFT axiom point
fwd = ParametricForward(delta=5/24, delta_g=1/24)
result = fwd.compute_all()

print(f"n_s = {result['n_s']:.6f}")            # 0.961035
print(f"m_H = {result['higgs_mass']:.2f} GeV")  # 125.46
print(f"η_B = {result['eta_b']:.3e}")           # 6.268e-10
```

### Train the Surrogate

```python
from sdgft_ml.training.train_surrogate import train_surrogate
model, history = train_surrogate()
```

### Validate Against Experiments

```python
from sdgft_ml.training.validate_real import scorecard, chi_squared
report = scorecard()  # 22 observables vs PDG/Planck/NuFIT
chi2, ndof, pval = chi_squared()
print(f"χ²/ndof = {chi2/ndof:.2f}, p = {pval:.4f}")
```

### CLI

```bash
# Predict observables at the SDGFT axiom point
sdgft-predict

# Custom parameters
sdgft-predict --delta 0.21 --delta-g 0.04

# JSON output
sdgft-predict --json
```

## Experimental Validation (Phase E)

The notebook executes 7 end-to-end tests comparing SDGFT predictions (0 free
parameters) against real experimental data:

| # | Test | Data Source | Key Result | Verdict |
|---|------|------------|------------|---------|
| 1 | Planck → Inverter | Planck 2018 | Δ err = 0.8 %, δ_g err = 0.5 % | ✓ PASS |
| 2 | Hubble Tension | Planck + DESI 2024 | w_DE = −0.932, +3.3σ | ⚠ TENSION |
| 3 | S₈ Tension | KiDS-1000, DES-Y3, HSC | S₈ = 0.788, 1.2σ vs KiDS | ✓ PASS |
| 4 | SPARC Tully-Fisher | 135 SPARC galaxies | b = 3.52 ± 0.16, 1.7σ | ✓ PASS |
| 5 | PDG Precision | PDG 2024 | 20/22 within 2σ, χ²/ndof = 1.48 | ✓ PASS |
| 6 | Neutrinos | NuFIT 5.3, KATRIN | R = 33.5 (−0.08σ), δ_CP = 225° | ✓ PASS |
| 7 | Antimatter & CKM | PDG 2024, Planck+BBN | η_B +0.66σ, all < 2σ | ✓ PASS |

**Result: 6 PASS · 1 TENSION · χ²/ndof = 1.48 · p = 0.069**

### Falsifiable Predictions

| Observable | SDGFT Prediction | Experiment | Timeline |
|-----------|-----------------|------------|----------|
| Σm_ν | 0.058 eV | KATRIN, JUNO, DUNE | ~2028 |
| δ_CP | 225° (5π/4) | DUNE | ~2030s |
| w_DE | −0.932 | DESI, Euclid, Rubin | ~2026–2030 |
| b_TF | 91/24 ≈ 3.792 | WALLABY survey | ~2027 |
| Mass ordering | Normal | JUNO | ~2026 |
| m_ββ | 2.8 meV | nEXO, LEGEND | ~2030s |

## Theory Background

SDGFT derives all Standard Model and cosmological observables from a
six-dimensional geometric flow on a Fibonacci lattice. The two axiom
parameters encode:

- **Δ = 5/24** — Fibonacci-lattice conflict (dimensional flow scale)
- **δ_g = 1/24** — Elementary lattice tension (gravitational coupling seed)

The computation DAG spans 7 levels:

```
Level 0: (Δ, δ_g, φ)
  → Level 1: D* (effective dimension)
    → Level 2: Gravity (G_N, Λ, modified dynamics)
      → Level 3: Inflation (n_s, r, N_e)
        → Level 4: Cosmology (Ω_m, Ω_Λ, H₀, S₈, w_DE)
        → Level 5: Particles (m_H, sin²θ_W, α_em, α_s, CKM, PMNS)
          → Level 6: Neutrinos (Δm², θ_ij, δ_CP, Σm_ν)
```

## Citing

If you use this code in your research, please cite:

```bibtex
@software{besemer_sdgft_ml_2026,
  author       = {Besemer, David A.},
  title        = {{SDGFT-ML}: Machine-Learning Surrogate and Inverter
                  for Six-Dimensional Geometric Field Theory},
  year         = 2026,
  url          = {https://github.com/cosmologicmind/sdgft-machine-learning},
  version      = {0.1.0}
}
```

## License

[MIT](LICENSE)
