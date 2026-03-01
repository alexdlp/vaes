from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

def plot_reconstruction_comparison(
    images: np.ndarray,
    reconstructions: np.ndarray,
    max_samples: int = 8,
) -> Figure:
    """
    Build a two-row figure comparing original images against reconstructions.

    Args:
        images: Batch of original images as [N, H, W] or [N, 1, H, W].
        reconstructions: Batch of reconstructed images with a compatible shape.
        max_samples: Maximum number of columns to draw.

    Returns:
        Matplotlib figure with a 2 x N comparison grid.
    """
    n = min(max_samples, images.shape[0], reconstructions.shape[0])
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for idx in range(n):
        axes[0, idx].imshow(np.squeeze(images[idx]), cmap="gray")
        axes[0, idx].grid(True, alpha=0.5)
        axes[0, idx].tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )
        axes[0, idx].set_title("Original")

        axes[1, idx].imshow(np.squeeze(reconstructions[idx]), cmap="gray")
        axes[1, idx].grid(True, alpha=0.5)
        axes[1, idx].tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )
        axes[1, idx].set_title("Recon")

    fig.suptitle("Reconstruction Comparison", fontsize=12)
    fig.tight_layout()
    return fig


def plot_latent_scatter(
    coords_2d: np.ndarray,
    labels: np.ndarray,
    title: str,
) -> Figure:
    """
    Build a class-colored 2D scatter plot of latent representations.

    Args:
        coords_2d: Two-dimensional latent coordinates as [N, 2].
        labels: Class labels as [N].
        title: Figure title.

    Returns:
        Matplotlib figure with the latent scatter.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for cls in np.unique(labels):
        mask = labels == cls
        ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1], s=5, alpha=0.8, label=str(int(cls)))

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.5)
    fig.tight_layout()
    return fig


def plot_latent_interpolation_grid(images: np.ndarray, grid_size: int) -> Figure:
    """
    Build a square image grid for latent-space interpolation outputs.

    Args:
        images: Generated images ordered by grid traversal, shape [K, H, W] or [K, 1, H, W].
        grid_size: Number of rows and columns in the interpolation grid.

    Returns:
        Matplotlib figure containing the interpolation panel.
    """
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            axes[i, j].imshow(np.squeeze(images[idx]), cmap="gray")
            axes[i, j].grid(True, alpha=0.5)
            axes[i, j].tick_params(
                axis="both",
                which="both",
                bottom=False,
                left=False,
                labelbottom=False,
                labelleft=False,
            )
            idx += 1

    fig.suptitle("Latent Space Interpolation", fontsize=12)
    fig.tight_layout()
    return fig


def plot_generated_samples(images: np.ndarray, max_samples: int = 16) -> Figure:
    """
    Build a compact grid of randomly generated samples.

    Args:
        images: Generated images as [N, H, W] or [N, 1, H, W].
        max_samples: Maximum number of samples to display.

    Returns:
        Matplotlib figure containing the generated sample grid.
    """
    n = min(max_samples, images.shape[0])
    cols = max(1, int(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(rows, cols)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            axes[r, c].grid(True, alpha=0.5)
            axes[r, c].tick_params(
                axis="both",
                which="both",
                bottom=False,
                left=False,
                labelbottom=False,
                labelleft=False,
            )
            if idx < n:
                axes[r, c].imshow(np.squeeze(images[idx]), cmap="gray")
            idx += 1

    fig.suptitle("Samples from Random Gaussian Noise", fontsize=12)
    fig.tight_layout()
    return fig
