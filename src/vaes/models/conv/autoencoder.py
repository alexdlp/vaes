"""Convolutional autoencoder for 2D image data."""

import torch
import torch.nn as nn

from .encoder import ConvEncoder
from .decoder import ConvDecoder


class ConvAutoEncoder(nn.Module):
    """Deterministic convolutional autoencoder.

    Composes a :class:`ConvEncoder` and :class:`ConvDecoder`.  The encoder
    output feature map is flattened and used directly as the latent
    representation — there is no stochastic sampling.

    For a 32x32 single-channel input with ``channels_bottleneck=4``, the
    latent is a flattened vector of size ``4 * 4 * 4 = 64``.

    Args:
        in_channels: Number of input image channels.
        channels_bottleneck: Feature channels at the spatial bottleneck.
    """

    def __init__(self, in_channels: int = 1, channels_bottleneck: int = 4) -> None:
        super().__init__()
        self.encoder = ConvEncoder(in_channels, channels_bottleneck)
        self.decoder = ConvDecoder(in_channels, channels_bottleneck)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Full autoencoder forward pass.

        Args:
            x: Input images of shape ``(B, C, H, W)``.

        Returns:
            Dict with keys ``"z"`` (latent feature map, flattened to
            ``(B, bottleneck * H' * W')``) and ``"recon"`` (reconstructed
            images in original shape).
        """
        z_map = self.encoder(x)  # (B, bottleneck, 4, 4)
        recon = self.decoder(z_map)  # (B, C, H, W)

        # Flatten the spatial latent for downstream analysis / visualization
        z = z_map.flatten(1)

        return {"z": z, "recon": recon}
