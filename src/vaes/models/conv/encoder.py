"""Convolutional encoder for 2D image data."""

import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    """2D convolutional encoder that progressively downsamples spatial dimensions.

    Architecture: three strided convolution blocks, each consisting of
    ``Conv2d -> BatchNorm2d -> ReLU``.  With the default settings and a 32x32
    input, the spatial dimensions reduce as 32 -> 16 -> 8 -> 4, yielding a
    feature map of shape ``(B, channels_bottleneck, 4, 4)``.

    Args:
        in_channels: Number of input image channels (1 for grayscale, 3 for
            RGB).
        channels_bottleneck: Number of feature channels at the bottleneck
            (smallest spatial resolution).
    """

    def __init__(self, in_channels: int = 1, channels_bottleneck: int = 4) -> None:
        super().__init__()

        self.net = nn.Sequential(
            # 32x32 -> 16x16
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # 16x16 -> 8x8
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # 8x8 -> 4x4
            nn.Conv2d(16, channels_bottleneck, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels_bottleneck),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images.

        Args:
            x: Input tensor of shape ``(B, in_channels, H, W)``.

        Returns:
            Feature map of shape ``(B, channels_bottleneck, H', W')`` where
            ``H'`` and ``W'`` are the downsampled spatial dimensions.
        """
        return self.net(x)
