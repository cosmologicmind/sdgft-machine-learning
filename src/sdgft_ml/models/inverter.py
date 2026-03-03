"""Conditional Variational Autoencoder for parameter inversion.

Given a set of observable values, infer the fundamental parameters
(Δ, δ_g, φ) that produced them.

The CVAE learns a latent distribution p(z | observables) and decodes
z → (Δ, δ_g, φ).  This solves the inverse problem: from measurements
back to axiom parameters.

Architecture
-----------
Encoder:  observables → μ, log_σ² (latent distribution)
Decoder:  z (sampled) → (Δ, δ_g, φ)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InverterCVAE(nn.Module):
    """Conditional VAE for SDGFT parameter inversion.

    Parameters
    ----------
    n_observables : int
        Dimension of the observable input vector (default 36).
    n_params : int
        Number of output parameters (default 3: Δ, δ_g, φ).
    hidden_dim : int
        Hidden layer width.
    latent_dim : int
        Dimensionality of the latent space.
    n_hidden : int
        Number of hidden layers in encoder/decoder.
    """

    def __init__(
        self,
        n_observables: int = 36,
        n_params: int = 3,
        hidden_dim: int = 128,
        latent_dim: int = 16,
        n_hidden: int = 3,
    ):
        super().__init__()
        self.n_observables = n_observables
        self.n_params = n_params
        self.latent_dim = latent_dim

        # ── Encoder: observables → latent ──
        enc_layers: list[nn.Module] = [nn.Linear(n_observables, hidden_dim), nn.SiLU()]
        for _ in range(n_hidden - 1):
            enc_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # ── Decoder: latent → parameters ──
        dec_layers: list[nn.Module] = [nn.Linear(latent_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_hidden - 1):
            dec_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        dec_layers.append(nn.Linear(hidden_dim, n_params))
        self.decoder = nn.Sequential(*dec_layers)

        # Output activation: sigmoid to constrain parameters to valid ranges
        # Δ ∈ [0, 0.5], δ_g ∈ [0, 0.25], φ ∈ [1, 2]
        self.param_min = nn.Parameter(
            torch.tensor([0.01, 0.001, 1.0]), requires_grad=False
        )
        self.param_max = nn.Parameter(
            torch.tensor([0.45, 0.20, 2.0]), requires_grad=False
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode observables to latent distribution parameters.

        Parameters
        ----------
        x : (B, n_observables)

        Returns
        -------
        mu, logvar : (B, latent_dim)
        """
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z ~ N(μ, σ²) using the reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to parameter predictions.

        Parameters
        ----------
        z : (B, latent_dim)

        Returns
        -------
        params : (B, n_params) — in physical ranges
        """
        raw = self.decoder(z)
        # Sigmoid activation scaled to valid ranges
        scaled = torch.sigmoid(raw)
        return self.param_min + scaled * (self.param_max - self.param_min)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Parameters
        ----------
        x : (B, n_observables)

        Returns
        -------
        params_pred, mu, logvar
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        params_pred = self.decode(z)
        return params_pred, mu, logvar

    def loss(
        self,
        params_pred: torch.Tensor,
        params_true: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ELBO loss = reconstruction + β · KL divergence.

        Parameters
        ----------
        params_pred : (B, n_params) predicted parameters
        params_true : (B, n_params) true parameters
        mu, logvar : (B, latent_dim) encoder outputs
        beta : float
            KL weight (β-VAE).

        Returns
        -------
        total_loss, recon_loss, kl_loss
        """
        recon = F.mse_loss(params_pred, params_true, reduction="mean")
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = recon + beta * kl
        return total, recon, kl

    @torch.no_grad()
    def invert(
        self,
        observables: torch.Tensor,
        n_samples: int = 100,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Invert: observables → parameter estimates with uncertainty.

        Parameters
        ----------
        observables : (n_observables,) or (B, n_observables)
        n_samples : int
            Number of latent samples for uncertainty estimation.

        Returns
        -------
        mean_params, std_params : (n_params,) or (B, n_params)
        """
        self.eval()
        if observables.dim() == 1:
            observables = observables.unsqueeze(0)

        mu, logvar = self.encode(observables)
        std = torch.exp(0.5 * logvar)

        samples = []
        for _ in range(n_samples):
            eps = torch.randn_like(std)
            z = mu + eps * std
            params = self.decode(z)
            samples.append(params)

        samples = torch.stack(samples, dim=0)  # (n_samples, B, n_params)
        mean = samples.mean(dim=0).squeeze(0)
        std_out = samples.std(dim=0).squeeze(0)
        return mean, std_out


class InverterEnsemble(nn.Module):
    """Ensemble of CVAEs for better uncertainty calibration.

    Trains N independent CVAEs and aggregates their predictions.
    """

    def __init__(
        self,
        n_models: int = 5,
        **cvae_kwargs,
    ):
        super().__init__()
        self.models = nn.ModuleList(
            [InverterCVAE(**cvae_kwargs) for _ in range(n_models)]
        )

    def forward(self, x: torch.Tensor):
        """Forward through all ensemble members."""
        results = [m(x) for m in self.models]
        # Average predictions
        params = torch.stack([r[0] for r in results], dim=0).mean(dim=0)
        return params

    @torch.no_grad()
    def invert(
        self,
        observables: torch.Tensor,
        n_samples_per_model: int = 20,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Invert with full ensemble uncertainty."""
        all_samples = []
        for model in self.models:
            mean, _ = model.invert(observables, n_samples=n_samples_per_model)
            all_samples.append(mean)
        stacked = torch.stack(all_samples, dim=0)
        return stacked.mean(dim=0), stacked.std(dim=0)
