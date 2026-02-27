"""Convolutional Variational AutoEncoder (VAE) for 2D image data."""

import torch
import torch.nn as nn

from .encoder import ConvEncoder
from .decoder import ConvDecoder
from ..common import reparameterize


class ConvVAE(nn.Module):
    """Convolutional Variational AutoEncoder.

    Uses a :class:`ConvEncoder` backbone and adds two parallel convolutional
    heads that predict the spatial mean (mu) and log-variance maps of the
    approximate posterior.  The latent is sampled via the reparameterization
    trick and then decoded with a :class:`ConvDecoder`.

    For a 32x32 single-channel input with ``channels_bottleneck=4``, the
    encoder produces a ``(B, 4, 4, 4)`` feature map.  The mu and logvar
    convolutions operate on this spatial grid, preserving its shape.

    Args:
        in_channels: Number of input image channels.
        channels_bottleneck: Feature channels at the spatial bottleneck.
    """

    def __init__(self, in_channels: int = 1, channels_bottleneck: int = 4) -> None:
        super().__init__()
        self.channels_bottleneck = channels_bottleneck

        self.encoder = ConvEncoder(in_channels, channels_bottleneck)

        # Convolutional heads for mu and log-variance (same spatial resolution)
        self.conv_mu = nn.Conv2d(
            channels_bottleneck, channels_bottleneck,
            kernel_size=3, stride=1, padding="same",
        )
        self.conv_logvar = nn.Conv2d(
            channels_bottleneck, channels_bottleneck,
            kernel_size=3, stride=1, padding="same",
        )

        self.decoder = ConvDecoder(in_channels, channels_bottleneck)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Full VAE forward pass: encode -> sample -> decode.

        Args:
            x: Input images of shape ``(B, C, H, W)``.

        Returns:
            Dict with keys:
            - ``"z"``: sampled latent (flattened) ``(B, bottleneck * H' * W')``
            - ``"recon"``: reconstructed images ``(B, C, H, W)``
            - ``"mu"``: posterior mean (flattened) ``(B, bottleneck * H' * W')``
            - ``"logvar"``: posterior log-variance (flattened)
        """
        # Encode
        h = self.encoder(x)  # (B, bottleneck, 4, 4)

        # Posterior parameters
        mu = self.conv_mu(h)  # (B, bottleneck, 4, 4)
        logvar = self.conv_logvar(h)  # (B, bottleneck, 4, 4)

        # Sample via reparameterization trick (operates element-wise)
        z_map = reparameterize(mu, logvar)  # (B, bottleneck, 4, 4)

        # Decode
        recon = self.decoder(z_map)  # (B, C, H, W)

        # Flatten spatial latents for loss computation and visualization
        z = z_map.flatten(1)
        mu_flat = mu.flatten(1)
        logvar_flat = logvar.flatten(1)

        return {"z": z, "recon": recon, "mu": mu_flat, "logvar": logvar_flat}
