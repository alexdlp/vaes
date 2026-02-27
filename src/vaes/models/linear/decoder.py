"""Fully-connected decoder for image data."""

from typing import Sequence

import torch
import torch.nn as nn


class LinearDecoder(nn.Module):
    """Fully-connected decoder that maps latent vectors back to image space.

    The network is a stack of ``Linear -> ReLU`` blocks followed by a final
    ``Linear -> Sigmoid`` layer that produces pixel values in [0, 1].

    Args:
        latent_dim: Dimension of the input latent vector.
        hidden_dims: Sizes of the hidden layers, applied in order.  Typically
            the reverse of the encoder hidden dims.
        output_dim: Total number of output features (must match encoder
            ``input_dim``).
    """

    def __init__(
        self,
        latent_dim: int = 2,
        hidden_dims: Sequence[int] = (64, 128),
        output_dim: int = 1024,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        # Final projection with sigmoid to bound output to [0, 1]
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a batch of latent vectors.

        Args:
            z: Tensor of shape ``(B, latent_dim)``.

        Returns:
            Reconstruction of shape ``(B, output_dim)``.
        """
        return self.net(z)
