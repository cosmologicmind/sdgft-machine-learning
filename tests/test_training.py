"""Tests for training infrastructure and evaluation."""

from __future__ import annotations

import math

import numpy as np
import pytest

_torch_available = True
try:
    import torch
except ImportError:
    _torch_available = False

_pyg_available = True
try:
    from torch_geometric.data import Data
except ImportError:
    _pyg_available = False

pytestmark = pytest.mark.skipif(not _torch_available, reason="PyTorch not installed")


class TestEvaluateCanaryTest:
    """Test the canary test evaluation at the SDGFT axiom point."""

    def test_exact_canary_values(self):
        """Verify the canary test uses correct exact SDGFT values."""
        from sdgft_ml.data.parameter_sweep import ParametricForward
        fwd = ParametricForward()
        result = fwd.compute_all()

        # Key observables at the axiom point
        assert abs(result["d_star_tree"] - 67 / 24) < 1e-10
        assert abs(result["omega_b"] - 115 / 2304) < 1e-10
        assert abs(result["beta_iso"] - 1 / 36) < 1e-10
        assert abs(result["alpha_s"] - math.sqrt(2) / 12) < 1e-10

    @pytest.mark.skipif(not _pyg_available, reason="torch_geometric not installed")
    def test_canary_test_structure(self):
        """Test that canary_test returns expected structure."""
        from sdgft_ml.training.evaluate import canary_test
        from sdgft_ml.models.surrogate_gnn import SurrogateGNN
        from sdgft_ml.data.dag_builder import (
            build_dag, build_edge_index, observable_names,
        )

        n_nodes = len(observable_names())
        adj, names = build_dag()
        edge_index = build_edge_index(adj, names)

        model = SurrogateGNN(
            n_params=3, n_nodes=n_nodes,
            hidden_dim=16, n_heads=2, n_layers=1, dropout=0.0,
        )

        result = canary_test(model, edge_index)
        assert "predictions" in result
        assert "targets" in result
        assert "relative_errors" in result
        assert "pass_rate" in result
        assert "status" in result
        assert result["n_total"] == n_nodes


class TestTrainSurrogateDataset:
    """Test the dataset construction for surrogate training."""

    def test_sdgft_graph_dataset(self):
        from sdgft_ml.training.train_surrogate import SDGFTGraphDataset

        params = np.random.randn(10, 3).astype(np.float32)
        targets = np.random.randn(10, 36).astype(np.float32)
        edge_index = np.array([[0, 1], [1, 2]], dtype=np.int64)

        ds = SDGFTGraphDataset(params, targets, edge_index)
        assert len(ds) == 10

        p, t = ds[0]
        assert p.shape == (3,)
        assert t.shape == (36,)


class TestTrainInverterDataset:
    """Test the dataset construction for inverter training."""

    def test_inverter_dataset(self):
        from sdgft_ml.training.train_inverter import InverterDataset

        obs = np.random.randn(10, 36).astype(np.float32)
        params = np.random.randn(10, 3).astype(np.float32)

        ds = InverterDataset(obs, params)
        assert len(ds) == 10

        o, p = ds[0]
        assert o.shape == (36,)
        assert p.shape == (3,)

    def test_inverter_dataset_normalized(self):
        from sdgft_ml.training.train_inverter import InverterDataset

        obs = np.random.randn(100, 36).astype(np.float32)
        params = np.random.randn(100, 3).astype(np.float32)
        mean = obs.mean(axis=0)
        std = obs.std(axis=0)
        std[std < 1e-12] = 1.0

        ds = InverterDataset(obs, params, mean, std)
        o, _ = ds[0]
        # Should be approximately normalized
        all_obs = torch.stack([ds[i][0] for i in range(len(ds))])
        assert abs(all_obs.mean()) < 0.5


class TestEvaluateInverter:
    """Test the inverter evaluation function."""

    def test_evaluate_inverter_structure(self):
        from sdgft_ml.training.evaluate import evaluate_inverter
        from sdgft_ml.models.inverter import InverterCVAE

        model = InverterCVAE(
            n_observables=37, n_params=3,
            hidden_dim=16, latent_dim=4, n_hidden=1,
        )
        result = evaluate_inverter(model, n_test=5)
        assert "mean_abs_error" in result
        assert "std_abs_error" in result
        assert "delta" in result["mean_abs_error"]
        assert "delta_g" in result["mean_abs_error"]
        assert "phi" in result["mean_abs_error"]


# ═══════════════════════════════════════════════════════════════════
# Test Anomaly Detector
# ═══════════════════════════════════════════════════════════════════

class TestAnomalyDetector:
    """Test the anomaly detection module."""

    def test_residual_autoencoder(self):
        from sdgft_ml.loop.anomaly_detector import ResidualAutoencoder
        ae = ResidualAutoencoder(n_features=36, hidden_dim=16, bottleneck_dim=4)
        x = torch.randn(8, 36)
        recon, z = ae(x)
        assert recon.shape == (8, 36)
        assert z.shape == (8, 4)

    def test_anomaly_score(self):
        from sdgft_ml.loop.anomaly_detector import ResidualAutoencoder
        ae = ResidualAutoencoder(n_features=36, hidden_dim=16, bottleneck_dim=4)
        x = torch.randn(8, 36)
        scores = ae.anomaly_score(x)
        assert scores.shape == (8,)
        assert (scores >= 0).all()

    def test_anomaly_detector_fit(self):
        from sdgft_ml.loop.anomaly_detector import AnomalyDetector
        detector = AnomalyDetector(n_features=36, hidden_dim=16, bottleneck_dim=4)
        residuals = np.random.randn(50, 36).astype(np.float32)
        loss_history = detector.fit(residuals, n_epochs=5, batch_size=16)
        assert len(loss_history) == 5
        assert all(l > 0 for l in loss_history)

    def test_anomaly_detector_detect(self):
        from sdgft_ml.loop.anomaly_detector import AnomalyDetector
        detector = AnomalyDetector(n_features=36, hidden_dim=16, bottleneck_dim=4)
        residuals = np.random.randn(100, 36).astype(np.float32)
        params = np.random.randn(100, 3).astype(np.float32)
        detector.fit(residuals, n_epochs=5, batch_size=32)
        result = detector.detect(params, residuals, quantile=0.95)
        assert result.scores.shape == (100,)
        assert result.anomaly_mask.shape == (100,)
        assert result.anomaly_mask.sum() <= 10  # ~5% anomalous


# ═══════════════════════════════════════════════════════════════════
# Test Active Learner (basic)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _pyg_available, reason="torch_geometric not installed")
class TestActiveLearner:
    """Test the active learning loop (basic checks)."""

    def test_acquisition(self):
        from sdgft_ml.loop.active_learner import ActiveLearner
        from sdgft_ml.models.surrogate_gnn import SurrogateGNNWithUncertainty
        from sdgft_ml.data.dag_builder import (
            build_dag, build_edge_index, observable_names,
        )

        n_nodes = len(observable_names())
        adj, names = build_dag()
        edge_index = build_edge_index(adj, names)

        model = SurrogateGNNWithUncertainty(
            n_nodes=n_nodes, hidden_dim=16,
            n_heads=2, n_layers=1, dropout=0.1,
        )

        # Initial dummy training data
        train_params = np.random.randn(20, 3).astype(np.float32)
        train_targets = np.random.randn(20, n_nodes).astype(np.float32)

        learner = ActiveLearner(
            model=model,
            edge_index=edge_index,
            train_params=train_params,
            train_targets=train_targets,
        )

        result = learner.acquire(n_candidates=10, n_acquire=3, n_mc_samples=5)
        assert result.round_idx == 0
        assert result.n_acquired <= 3


# ═══════════════════════════════════════════════════════════════════
# Test CLI predict
# ═══════════════════════════════════════════════════════════════════

class TestPredict:
    """Test the predict CLI function."""

    def test_predict_exact(self):
        from sdgft_ml.api.predict import predict_from_params
        result = predict_from_params(
            delta=5.0 / 24.0,
            delta_g=1.0 / 24.0,
        )
        assert "d_star_tree" in result
        assert "n_s" in result
        assert abs(result["d_star_tree"] - 67 / 24) < 1e-10
