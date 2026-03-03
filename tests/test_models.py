"""Tests for ML models: SurrogateGNN, InverterCVAE, CurveDeepONet."""

from __future__ import annotations

import numpy as np
import pytest

_torch_available = True
try:
    import torch
    import torch.nn as nn
except ImportError:
    _torch_available = False

_pyg_available = True
try:
    from torch_geometric.nn import GATv2Conv
except ImportError:
    _pyg_available = False

pytestmark = pytest.mark.skipif(not _torch_available, reason="PyTorch not installed")


# ═══════════════════════════════════════════════════════════════════
# Test SurrogateGNN
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _pyg_available, reason="torch_geometric not installed")
class TestSurrogateGNN:
    """Test the GATv2-based surrogate model."""

    def setup_method(self):
        from sdgft_ml.models.surrogate_gnn import SurrogateGNN
        from sdgft_ml.data.dag_builder import build_dag, build_edge_index, observable_names

        self.n_nodes = len(observable_names())
        adj, names = build_dag()
        self.edge_index = torch.from_numpy(build_edge_index(adj, names))

        self.model = SurrogateGNN(
            n_params=3,
            n_nodes=self.n_nodes,
            hidden_dim=32,
            n_heads=2,
            n_layers=2,
            dropout=0.0,
        )

    def test_forward_shape(self):
        params = torch.randn(1, 3)
        pred = self.model(params, self.edge_index)
        assert pred.shape == (self.n_nodes,)

    def test_forward_batch(self):
        B = 4
        params = torch.randn(B, 3)
        # Expand edge index for batch
        n = self.n_nodes
        offsets = torch.arange(B).unsqueeze(1) * n
        ei = self.edge_index.unsqueeze(0).expand(B, -1, -1)
        ei_batch = (ei + offsets.unsqueeze(1)).reshape(2, -1)

        pred = self.model(params, ei_batch)
        assert pred.shape == (B * n,)

    def test_gradients_flow(self):
        params = torch.randn(1, 3, requires_grad=True)
        pred = self.model(params, self.edge_index)
        loss = pred.sum()
        loss.backward()
        assert params.grad is not None
        assert params.grad.abs().sum() > 0

    def test_parameter_count(self):
        n_params = sum(p.numel() for p in self.model.parameters())
        assert n_params > 1000, f"Model too small: {n_params} params"

    def test_predict_method(self):
        result = self.model.predict(
            delta=5.0 / 24.0,
            delta_g=1.0 / 24.0,
            phi=(1 + 5**0.5) / 2.0,
            edge_index=self.edge_index,
        )
        assert isinstance(result, dict)
        assert "n_s" in result
        assert "d_star_tree" in result


@pytest.mark.skipif(not _pyg_available, reason="torch_geometric not installed")
class TestSurrogateGNNWithUncertainty:
    """Test MC-Dropout uncertainty estimation."""

    def test_uncertainty(self):
        from sdgft_ml.models.surrogate_gnn import SurrogateGNNWithUncertainty
        from sdgft_ml.data.dag_builder import build_dag, build_edge_index, observable_names

        n_nodes = len(observable_names())
        adj, names = build_dag()
        edge_index = torch.from_numpy(build_edge_index(adj, names))

        model = SurrogateGNNWithUncertainty(
            n_nodes=n_nodes, hidden_dim=32,
            n_heads=2, n_layers=2, dropout=0.1,
        )
        means, stds = model.predict_with_uncertainty(
            delta=5.0 / 24.0,
            delta_g=1.0 / 24.0,
            phi=(1 + 5**0.5) / 2.0,
            edge_index=edge_index,
            n_samples=10,
        )
        assert isinstance(means, dict)
        assert isinstance(stds, dict)
        assert len(means) == n_nodes
        # With dropout, there should be some variance
        assert any(v > 0 for v in stds.values())


# ═══════════════════════════════════════════════════════════════════
# Test InverterCVAE
# ═══════════════════════════════════════════════════════════════════

class TestInverterCVAE:
    """Test the Conditional VAE inverter."""

    def setup_method(self):
        from sdgft_ml.models.inverter import InverterCVAE
        self.model = InverterCVAE(
            n_observables=36,
            n_params=3,
            hidden_dim=32,
            latent_dim=8,
            n_hidden=2,
        )

    def test_forward_shape(self):
        x = torch.randn(4, 36)
        params_pred, mu, logvar = self.model(x)
        assert params_pred.shape == (4, 3)
        assert mu.shape == (4, 8)
        assert logvar.shape == (4, 8)

    def test_output_in_range(self):
        x = torch.randn(10, 36)
        params_pred, _, _ = self.model(x)
        # Should be within param_min/max bounds
        assert (params_pred >= self.model.param_min).all()
        assert (params_pred <= self.model.param_max).all()

    def test_loss_computation(self):
        x = torch.randn(4, 36)
        params_pred, mu, logvar = self.model(x)
        params_true = torch.rand(4, 3)
        total, recon, kl = self.model.loss(params_pred, params_true, mu, logvar)
        assert total.dim() == 0  # scalar
        assert recon >= 0
        assert kl >= 0

    def test_invert(self):
        x = torch.randn(36)
        mean, std = self.model.invert(x, n_samples=20)
        assert mean.shape == (3,)
        assert std.shape == (3,)

    def test_reparameterize_training(self):
        mu = torch.zeros(4, 8)
        logvar = torch.zeros(4, 8)
        self.model.train()
        z1 = self.model.reparameterize(mu, logvar)
        z2 = self.model.reparameterize(mu, logvar)
        # In training mode, should sample differently each time
        # (probabilistic, but with high probability they differ)
        assert z1.shape == (4, 8)

    def test_reparameterize_eval(self):
        mu = torch.ones(4, 8) * 0.5
        logvar = torch.zeros(4, 8)
        self.model.eval()
        z = self.model.reparameterize(mu, logvar)
        # In eval mode, should return mu directly
        torch.testing.assert_close(z, mu)


class TestInverterEnsemble:
    """Test the ensemble of CVAEs."""

    def test_ensemble_forward(self):
        from sdgft_ml.models.inverter import InverterEnsemble
        ensemble = InverterEnsemble(
            n_models=3,
            n_observables=36,
            n_params=3,
            hidden_dim=16,
            latent_dim=4,
            n_hidden=1,
        )
        x = torch.randn(4, 36)
        params = ensemble(x)
        assert params.shape == (4, 3)

    def test_ensemble_invert(self):
        from sdgft_ml.models.inverter import InverterEnsemble
        ensemble = InverterEnsemble(
            n_models=3,
            n_observables=36,
            n_params=3,
            hidden_dim=16,
            latent_dim=4,
            n_hidden=1,
        )
        x = torch.randn(36)
        mean, std = ensemble.invert(x, n_samples_per_model=5)
        assert mean.shape == (3,)
        assert std.shape == (3,)


# ═══════════════════════════════════════════════════════════════════
# Test CurveDeepONet
# ═══════════════════════════════════════════════════════════════════

class TestCurveDeepONet:
    """Test the DeepONet for scale-dependent functions."""

    def setup_method(self):
        from sdgft_ml.models.curve_net import CurveDeepONet
        self.model = CurveDeepONet(
            n_params=3,
            n_query=1,
            basis_dim=16,
            hidden_dim=32,
            n_hidden=2,
        )

    def test_forward_shape(self):
        params = torch.randn(8, 3)
        query = torch.randn(8, 1)
        out = self.model(params, query)
        assert out.shape == (8, 1)

    def test_forward_grid(self):
        params = torch.randn(3)
        query_grid = torch.randn(50, 1)
        out = self.model.forward_grid(params, query_grid)
        assert out.shape == (50,)

    def test_gradients(self):
        params = torch.randn(4, 3, requires_grad=True)
        query = torch.randn(4, 1)
        out = self.model(params, query)
        loss = out.sum()
        loss.backward()
        assert params.grad is not None


class TestMultiCurveDeepONet:
    """Test the multi-curve DeepONet container."""

    def test_all_curves_exist(self):
        from sdgft_ml.models.curve_net import MultiCurveDeepONet
        multi = MultiCurveDeepONet(hidden_dim=16, basis_dim=8, n_hidden=1)
        assert "d_star_of_r" in multi.nets
        assert "omega_de_rg" in multi.nets
        assert "grav_slip" in multi.nets

    def test_forward_d_star(self):
        from sdgft_ml.models.curve_net import MultiCurveDeepONet
        multi = MultiCurveDeepONet(hidden_dim=16, basis_dim=8, n_hidden=1)
        params = torch.randn(4, 2)
        query = torch.randn(4, 1)
        out = multi("d_star_of_r", params, query)
        assert out.shape == (4, 1)

    def test_forward_grav_slip(self):
        from sdgft_ml.models.curve_net import MultiCurveDeepONet
        multi = MultiCurveDeepONet(hidden_dim=16, basis_dim=8, n_hidden=1)
        params = torch.randn(4, 1)
        query = torch.randn(4, 1)
        out = multi("grav_slip", params, query)
        assert out.shape == (4, 1)


# ═══════════════════════════════════════════════════════════════════
# Test ParameterEncoder / NodeDecoder independently
# ═══════════════════════════════════════════════════════════════════

class TestEncoderDecoder:
    """Test sub-components of the GNN."""

    def test_parameter_encoder(self):
        from sdgft_ml.models.surrogate_gnn import ParameterEncoder
        enc = ParameterEncoder(n_params=3, n_nodes=10, hidden_dim=16)
        params = torch.randn(2, 3)
        h = enc(params)
        assert h.shape == (20, 16)  # 2 batches * 10 nodes

    def test_node_decoder(self):
        from sdgft_ml.models.surrogate_gnn import NodeDecoder
        dec = NodeDecoder(hidden_dim=16)
        h = torch.randn(10, 16)
        out = dec(h)
        assert out.shape == (10, 1)

    def test_mlp(self):
        from sdgft_ml.models.curve_net import MLP
        mlp = MLP(in_dim=5, out_dim=3, hidden_dim=16, n_hidden=2)
        x = torch.randn(8, 5)
        out = mlp(x)
        assert out.shape == (8, 3)
