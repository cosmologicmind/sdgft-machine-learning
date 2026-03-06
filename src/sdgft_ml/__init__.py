"""SDGFT-ML: Machine-learning surrogate, inverter & anomaly detector for SDGFT.

Modules
-------
data        – Registry export, parameter sweeps, DAG construction
models      – GNN surrogate (GATv2), Conditional VAE inverter, DeepONet curve net
training    – Training loops and evaluation
loop        – Active learning, anomaly detection, symbolic distillation
api         – CLI prediction and visualization
physics     – Closed-form SDGFT physics modules (QED, neutrinos, black holes, …)
inference   – High-level prediction & Oracle database query API
validation  – Experimental reference data (PDG 2024, Planck 2018, NuFIT 5.3)
"""

__version__ = "1.1.0"
