"""Linear Variational AutoEncoder (VAE)."""

from typing import Sequence

import torch
import torch.nn as nn

from .encoder import LinearEncoder
from .decoder import LinearDecoder
from ..common import reparameterize


class LinearVAE(nn.Module):
    """Fully-connected Variational AutoEncoder.

    Uses the same :class:`LinearEncoder` backbone as the deterministic
    autoencoder, but adds separate linear heads that predict the mean (mu)
    and log-variance of the approximate posterior.  Sampling is performed via
    the reparameterization trick so that gradients can flow through the
    stochastic node.

    Args:
        input_dim: Total number of input features (``C * H * W``).
        hidden_dims: Hidden layer sizes for the encoder.  The decoder mirrors
            them in reverse order.
        bottleneck_dim: Output dimension of the shared encoder backbone, before
            the mu / logvar projection.
        latent_dim: Dimensionality of the latent space (mu and logvar size).
        image_shape: Original image shape ``(C, H, W)`` used to reshape the
            decoder output.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dims: Sequence[int] = (128, 64),
        bottleneck_dim: int = 32,
        latent_dim: int = 2,
        image_shape: Sequence[int] = (1, 32, 32),
    ) -> None:
        super().__init__()
        self.image_shape = tuple(image_shape)

        # Shared encoder backbone
        self.encoder = LinearEncoder(input_dim, hidden_dims, output_dim=bottleneck_dim)

        # Separate heads for mu and log-variance
        self.fc_mu = nn.Linear(bottleneck_dim, latent_dim)
        self.fc_logvar = nn.Linear(bottleneck_dim, latent_dim)

        # Decoder: from latent back to image space
        self.decoder = LinearDecoder(latent_dim, hidden_dims=list(reversed(hidden_dims)), output_dim=input_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Full VAE forward pass: encode -> sample -> decode.

        Args:
            x: Input images of shape ``(B, C, H, W)``.

        Returns:
            Dict with keys:
            - ``"z"``: sampled latent codes ``(B, latent_dim)``
            - ``"recon"``: reconstructed images ``(B, C, H, W)``
            - ``"mu"``: posterior mean ``(B, latent_dim)``
            - ``"logvar"``: posterior log-variance ``(B, latent_dim)``
        """
        batch_size = x.shape[0]
        flat = x.flatten(1)  # (B, input_dim)

        # Encode to bottleneck
        h = self.encoder(flat)  # (B, bottleneck_dim)

        # Posterior parameters
        mu = self.fc_mu(h)  # (B, latent_dim)
        logvar = self.fc_logvar(h)  # (B, latent_dim)

        # Sample via reparameterization trick
        z = reparameterize(mu, logvar)  # (B, latent_dim)

        # Decode
        recon_flat = self.decoder(z)  # (B, input_dim)
        recon = recon_flat.reshape(batch_size, *self.image_shape)

        return {"z": z, "recon": recon, "mu": mu, "logvar": logvar}
