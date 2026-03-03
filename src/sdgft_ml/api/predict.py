"""CLI tool for querying the trained SDGFT-ML models.

Usage::

    # After installing the package:
    sdgft-predict --delta 0.2083 --delta-g 0.04167

    # Or directly:
    python -m sdgft_ml.api.predict --delta 0.2083 --delta-g 0.04167
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _load_surrogate(model_path: str, device: str = "cpu"):
    """Load a trained surrogate model from checkpoint."""
    import torch
    from ..models.surrogate_gnn import SurrogateGNN
    from ..data.dag_builder import build_dag, build_edge_index, observable_names

    n_nodes = len(observable_names())
    model = SurrogateGNN(n_params=3, n_nodes=n_nodes)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    adj, names = build_dag()
    edge_index = build_edge_index(adj, names)
    return model, edge_index, names


def predict_from_params(
    delta: float,
    delta_g: float,
    phi: float | None = None,
    model_path: str | None = None,
    use_exact: bool = True,
) -> dict[str, float]:
    """Predict all SDGFT observables for given parameters.

    Parameters
    ----------
    delta, delta_g, phi : float
        SDGFT axiom parameters.
    model_path : str | None
        Path to trained surrogate checkpoint. If None, uses exact computation.
    use_exact : bool
        If True (default), compute exact SDGFT values.
        Falls back to exact if no model_path is provided.
    """
    if phi is None:
        phi = (1.0 + math.sqrt(5.0)) / 2.0

    if model_path is not None and not use_exact:
        import torch
        model, edge_index, names = _load_surrogate(model_path)
        with torch.no_grad():
            params = torch.tensor([delta, delta_g, phi], dtype=torch.float32).unsqueeze(0)
            ei = torch.from_numpy(edge_index)
            pred = model(params, ei).numpy()
        return {n: float(pred[i]) for i, n in enumerate(names)}

    # Exact computation
    from ..data.parameter_sweep import ParametricForward
    fwd = ParametricForward(delta=delta, delta_g=delta_g, phi=phi)
    return fwd.compute_all()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="sdgft-predict",
        description="Predict SDGFT observables for given parameters",
    )
    parser.add_argument(
        "--delta", type=float, default=5.0 / 24.0,
        help="Fibonacci-lattice conflict Δ (default: 5/24)",
    )
    parser.add_argument(
        "--delta-g", type=float, default=1.0 / 24.0,
        help="Lattice tension δ_g (default: 1/24)",
    )
    parser.add_argument(
        "--phi", type=float, default=None,
        help="Golden ratio φ (default: (1+√5)/2)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to trained surrogate model checkpoint",
    )
    parser.add_argument(
        "--json", action="store_true", default=False,
        help="Output as JSON instead of table",
    )
    parser.add_argument(
        "--observables", nargs="*", default=None,
        help="Specific observables to display (default: all)",
    )

    args = parser.parse_args()

    result = predict_from_params(
        delta=args.delta,
        delta_g=args.delta_g,
        phi=args.phi,
        model_path=args.model,
        use_exact=args.model is None,
    )

    # Filter observables if requested
    if args.observables:
        result = {k: v for k, v in result.items() if k in args.observables}

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\nSDGFT-ML Prediction")
        print(f"  Δ    = {args.delta:.6f}")
        print(f"  δ_g  = {args.delta_g:.6f}")
        print(f"  φ    = {args.phi or (1+5**0.5)/2:.6f}")
        print(f"  Mode = {'surrogate' if args.model else 'exact'}")
        print(f"\n{'Observable':<30s} {'Value':>15s}")
        print("-" * 47)
        for name, value in sorted(result.items()):
            if name.startswith("param_"):
                continue
            if isinstance(value, float) and abs(value) < 0.01 and value != 0:
                print(f"{name:<30s} {value:>15.6e}")
            else:
                print(f"{name:<30s} {value:>15.6f}")


if __name__ == "__main__":
    main()
