"""Standard linear (fully-connected) autoencoder."""

from typing import Sequence

import torch
import torch.nn as nn

from .encoder import LinearEncoder
from .decoder import LinearDecoder


class LinearAutoEncoder(nn.Module):
    """Deterministic linear autoencoder.

    Composes a :class:`LinearEncoder` and :class:`LinearDecoder`.  The encoder
    output is used directly as the latent representation — there is no
    stochastic sampling.

    Args:
        input_dim: Total number of input features (``C * H * W``).
        hidden_dims: Hidden layer sizes for the encoder.  The decoder uses
            the reversed order.
        latent_dim: Dimensionality of the latent space.
        image_shape: Original image shape ``(C, H, W)`` used to reshape the
            decoder output.  Default matches single-channel 32x32 images.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dims: Sequence[int] = (128, 64),
        latent_dim: int = 2,
        image_shape: Sequence[int] = (1, 32, 32),
    ) -> None:
        super().__init__()
        self.image_shape = tuple(image_shape)

        self.encoder = LinearEncoder(input_dim, hidden_dims, output_dim=latent_dim)
        self.decoder = LinearDecoder(latent_dim, hidden_dims=list(reversed(hidden_dims)), output_dim=input_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Full autoencoder forward pass.

        Args:
            x: Input images of shape ``(B, C, H, W)``.

        Returns:
            Dict with keys ``"z"`` (latent codes) and ``"recon"``
            (reconstructed images in original shape).
        """
        batch_size = x.shape[0]
        flat = x.flatten(1)  # (B, input_dim)

        z = self.encoder(flat)  # (B, latent_dim)
        recon_flat = self.decoder(z)  # (B, input_dim)
        recon = recon_flat.reshape(batch_size, *self.image_shape)

        return {"z": z, "recon": recon}
