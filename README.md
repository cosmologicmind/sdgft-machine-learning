# SDGFT-ML

Machine-learning surrogate, inverter & anomaly detector for
**Six-Dimensional Geometric Flow Theory (SDGFT)**.

## Overview

This project trains ML models on the full SDGFT observable space
(~36 parametrically-computable observables derived from 2 axiom parameters)
to enable:

1. **Surrogate GNN** — GATv2-based graph neural network that mirrors the
   SDGFT computation DAG and predicts all observables for arbitrary (Δ, δ_g, φ)
2. **Inverter CVAE** — Conditional Variational Autoencoder that solves the
   inverse problem: observables → axiom parameters
3. **Curve DeepONet** — Operator network for learning scale-dependent functions
   like D*(r) and Ω_DE(r)
4. **Active Learning** — Uncertainty-guided sampling to iteratively improve
   the surrogate in uncertain parameter regions
5. **Anomaly Detection** — Autoencoder on prediction residuals to discover
   interesting parameter space regions
6. **Symbolic Distillation** — PySR-based extraction of human-readable formulas
   from the trained models

## Project Structure

```
sdgft_ml/
├── pyproject.toml
├── src/sdgft_ml/
│   ├── data/
│   │   ├── registry_export.py    # Export SDGFT observables
│   │   ├── parameter_sweep.py    # Parametric forward model & grid sweep
│   │   └── dag_builder.py        # DAG → PyG graph
│   ├── models/
│   │   ├── surrogate_gnn.py      # GATv2Conv surrogate
│   │   ├── inverter.py           # Conditional VAE
│   │   └── curve_net.py          # DeepONet
│   ├── training/
│   │   ├── train_surrogate.py    # GNN training loop
│   │   ├── train_inverter.py     # CVAE training loop
│   │   └── evaluate.py           # Evaluation & canary test
│   ├── loop/
│   │   ├── active_learner.py     # Uncertainty-guided sampling
│   │   ├── anomaly_detector.py   # Residual autoencoder
│   │   └── distillation.py       # Symbolic regression (PySR)
│   └── api/
│       ├── predict.py            # CLI tool
│       └── visualize.py          # Plotting utilities
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   └── test_training.py
├── data/
└── notebooks/
```

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Install SDGFT (from local checkout)
pip install -e ../SDGFT_V2

# Run tests
pytest
```

## Quick Start

```python
# Compute all observables for custom parameters
from sdgft_ml.data.parameter_sweep import ParametricForward
fwd = ParametricForward(delta=0.21, delta_g=0.04)
result = fwd.compute_all()
print(f"n_s = {result['n_s']:.4f}, m_H = {result['higgs_mass']:.2f} GeV")

# Generate training data
from sdgft_ml.data.parameter_sweep import sweep_latin_hypercube
df = sweep_latin_hypercube(n_samples=5000)

# Train the surrogate
from sdgft_ml.training.train_surrogate import train_surrogate
model, history = train_surrogate()

# Canary test at axiomatic point
from sdgft_ml.training.evaluate import canary_test, print_canary_report
result = canary_test(model, edge_index)
print_canary_report(result)
```

## CLI

```bash
# Predict observables at the SDGFT axiom point
sdgft-predict

# Custom parameters
sdgft-predict --delta 0.21 --delta-g 0.04

# JSON output
sdgft-predict --json
```

## Theory

All observables trace back to two axiom parameters:
- **Δ = 5/24** — Fibonacci-lattice conflict
- **δ_g = 1/24** — elementary lattice tension

Plus the golden ratio φ = (1+√5)/2 and two external anchors
(γ_EW = 0.12011, v_Higgs = 246.22 GeV).

The computation DAG spans 7 levels:
Level 0-1 → Level 2 (D*) → Level 3 (gravity) → Level 4 (inflation) → Level 5-6 (cosmology + particles)

## License

MIT
