"""Convolutional decoder for 2D image data."""

import torch
import torch.nn as nn


class ConvDecoder(nn.Module):
    """2D transposed-convolutional decoder that mirrors :class:`ConvEncoder`.

    Architecture: three transposed convolution blocks that progressively
    upsample spatial dimensions.  With the default settings and a 4x4 input,
    the spatial dimensions grow as 4 -> 8 -> 16 -> 32.  The final activation
    is ``Sigmoid`` to produce pixel values in [0, 1].

    Args:
        out_channels: Number of output image channels (1 for grayscale).
        channels_bottleneck: Number of feature channels at the bottleneck
            input (must match encoder output).
    """

    def __init__(self, out_channels: int = 1, channels_bottleneck: int = 4) -> None:
        super().__init__()

        self.channels_bottleneck = channels_bottleneck

        self.net = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(channels_bottleneck, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(8, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a batch of feature maps (or reshaped latent vectors).

        Args:
            z: Tensor of shape ``(B, channels_bottleneck, H', W')``.

        Returns:
            Reconstructed images of shape ``(B, out_channels, H, W)``.
        """
        return self.net(z)
