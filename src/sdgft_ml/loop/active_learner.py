"""Active learning loop with uncertainty-guided sampling.

The active learner identifies parameter regions where the surrogate
is most uncertain and generates new training data there.
This iterative process improves the model where it matters most.

Strategy
--------
1. Train surrogate on initial data
2. Generate candidate parameter points
3. Estimate uncertainty via MC-Dropout
4. Select top-K most uncertain points
5. Compute exact SDGFT values for those points
6. Add to training set and retrain
7. Repeat until convergence

Usage::

    from sdgft_ml.loop.active_learner import ActiveLearner
    learner = ActiveLearner(model, edge_index)
    learner.run(n_rounds=10, n_candidates=500, n_acquire=50)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import math
import numpy as np
import torch


@dataclass
class AcquisitionResult:
    """Result of an acquisition round."""
    round_idx: int
    n_acquired: int
    mean_uncertainty: float
    max_uncertainty: float
    acquired_params: np.ndarray  # (n_acquired, 3)
    acquired_targets: np.ndarray  # (n_acquired, n_nodes)


@dataclass
class ActiveLearningHistory:
    """Metrics over active learning rounds."""
    rounds: list[AcquisitionResult] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    canary_pass_rates: list[float] = field(default_factory=list)


class ActiveLearner:
    """Uncertainty-guided active learning for the SDGFT surrogate.

    Parameters
    ----------
    model : SurrogateGNNWithUncertainty
        Surrogate model with MC-Dropout uncertainty.
    edge_index : np.ndarray
        DAG edge index (2, E).
    train_params : np.ndarray
        Current training parameters (N, 3).
    train_targets : np.ndarray
        Current training targets (N, n_nodes).
    device : str
        Torch device.
    """

    def __init__(
        self,
        model: Any,
        edge_index: np.ndarray,
        train_params: np.ndarray,
        train_targets: np.ndarray,
        device: str = "cpu",
    ):
        self.model = model
        self.edge_index = edge_index
        self.edge_index_t = torch.from_numpy(edge_index).to(device)
        self.train_params = train_params.copy()
        self.train_targets = train_targets.copy()
        self.device = device
        self.history = ActiveLearningHistory()

    def _generate_candidates(
        self,
        n_candidates: int,
        delta_range: tuple[float, float] = (0.05, 0.40),
        delta_g_range: tuple[float, float] = (0.01, 0.08),
        seed: int | None = None,
    ) -> np.ndarray:
        """Generate random candidate parameter vectors."""
        rng = np.random.default_rng(seed)
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        candidates = np.empty((n_candidates, 3), dtype=np.float32)
        candidates[:, 0] = rng.uniform(*delta_range, n_candidates)
        candidates[:, 1] = rng.uniform(*delta_g_range, n_candidates)
        candidates[:, 2] = phi
        return candidates

    def _estimate_uncertainty(
        self,
        candidates: np.ndarray,
        n_mc_samples: int = 30,
    ) -> np.ndarray:
        """Estimate model uncertainty for candidate points via MC-Dropout.

        Returns
        -------
        uncertainties : (n_candidates,) — mean std across observables
        """
        self.model.train()  # enable dropout
        uncertainties = np.empty(len(candidates), dtype=np.float32)

        with torch.no_grad():
            for i, cand in enumerate(candidates):
                params = torch.tensor(cand, dtype=torch.float32).unsqueeze(0).to(self.device)
                preds = []
                for _ in range(n_mc_samples):
                    pred = self.model(params, self.edge_index_t).cpu().numpy()
                    preds.append(pred)
                preds = np.stack(preds, axis=0)  # (n_mc, n_nodes)
                uncertainties[i] = preds.std(axis=0).mean()

        return uncertainties

    def _compute_exact(self, params: np.ndarray) -> np.ndarray:
        """Compute exact SDGFT values for given parameter vectors.

        Parameters
        ----------
        params : (N, 3) — [delta, delta_g, phi]

        Returns
        -------
        targets : (N, n_nodes)
        """
        from ..data.parameter_sweep import ParametricForward
        from ..data.dag_builder import observable_names

        names = observable_names()
        targets = []

        for row in params:
            fwd = ParametricForward(
                delta=float(row[0]),
                delta_g=float(row[1]),
                phi=float(row[2]),
            )
            try:
                result = fwd.compute_all()
                vec = [result.get(n, 0.0) for n in names]
                if any(math.isnan(v) for v in vec):
                    continue
                targets.append(vec)
            except (ValueError, ZeroDivisionError, OverflowError):
                continue

        return np.array(targets, dtype=np.float32)

    def acquire(
        self,
        n_candidates: int = 500,
        n_acquire: int = 50,
        n_mc_samples: int = 30,
        seed: int | None = None,
    ) -> AcquisitionResult:
        """Run one acquisition round.

        1. Generate candidates
        2. Estimate uncertainty
        3. Select top-K most uncertain
        4. Compute exact values
        5. Add to training set
        """
        round_idx = len(self.history.rounds)

        # Step 1-2: Candidates + uncertainty
        candidates = self._generate_candidates(n_candidates, seed=seed)
        uncertainties = self._estimate_uncertainty(candidates, n_mc_samples)

        # Step 3: Select most uncertain
        top_k_idx = np.argsort(uncertainties)[-n_acquire:]
        selected = candidates[top_k_idx]

        # Step 4: Compute exact SDGFT values
        new_targets = self._compute_exact(selected)

        # Step 5: Add to training set
        n_valid = len(new_targets)
        if n_valid > 0:
            self.train_params = np.vstack([self.train_params, selected[:n_valid]])
            self.train_targets = np.vstack([self.train_targets, new_targets])

        result = AcquisitionResult(
            round_idx=round_idx,
            n_acquired=n_valid,
            mean_uncertainty=float(uncertainties[top_k_idx].mean()),
            max_uncertainty=float(uncertainties.max()),
            acquired_params=selected[:n_valid],
            acquired_targets=new_targets,
        )
        self.history.rounds.append(result)
        return result

    def run(
        self,
        n_rounds: int = 10,
        n_candidates: int = 500,
        n_acquire: int = 50,
        n_mc_samples: int = 30,
        retrain_epochs: int = 50,
        retrain_fn: Any | None = None,
    ) -> ActiveLearningHistory:
        """Run the full active learning loop.

        Parameters
        ----------
        n_rounds : int
            Number of acquisition rounds.
        retrain_fn : callable, optional
            Function to retrain the model given (model, train_params, train_targets).
            If None, skips retraining (useful for dry runs).

        Returns
        -------
        history
        """
        for r in range(n_rounds):
            print(f"\n--- Active Learning Round {r+1}/{n_rounds} ---")
            result = self.acquire(
                n_candidates=n_candidates,
                n_acquire=n_acquire,
                n_mc_samples=n_mc_samples,
                seed=42 + r,
            )
            print(f"  Acquired {result.n_acquired} new samples "
                  f"(mean unc: {result.mean_uncertainty:.4f}, "
                  f"max unc: {result.max_uncertainty:.4f})")
            print(f"  Total training set: {len(self.train_params)}")

            if retrain_fn is not None:
                val_loss = retrain_fn(
                    self.model,
                    self.train_params,
                    self.train_targets,
                    n_epochs=retrain_epochs,
                )
                self.history.val_losses.append(val_loss)
                print(f"  Retrained: val_loss = {val_loss:.6f}")

        return self.history
