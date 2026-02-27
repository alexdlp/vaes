"""Fully-connected encoder for image data."""

from typing import Sequence

import torch
import torch.nn as nn


class LinearEncoder(nn.Module):
    """Fully-connected encoder that maps flattened images to a latent representation.

    The network is a stack of ``Linear -> ReLU`` blocks.  The final layer has no
    activation so the output is unbounded and can be used directly as a latent
    vector (for a vanilla autoencoder) or fed into separate mu / logvar heads
    (for a VAE).

    Args:
        input_dim: Total number of input features (e.g. ``1 * 32 * 32 = 1024``
            for single-channel 32x32 images).
        hidden_dims: Sizes of the hidden layers, applied in order.
        output_dim: Dimension of the encoder output.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dims: Sequence[int] = (128, 64),
        output_dim: int = 32,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        # Final projection (no activation — downstream decides)
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of flattened inputs.

        Args:
            x: Tensor of shape ``(B, input_dim)``.

        Returns:
            Encoded representation of shape ``(B, output_dim)``.
        """
        return self.net(x)
