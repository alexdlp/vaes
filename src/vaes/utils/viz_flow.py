import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def plot_exact_density(
    density: np.ndarray,
    lims: np.ndarray,
    cmap: str = "coolwarm",
    title: str = "Density",
) -> Figure:
    """
    Plot a target density on a regular grid using imshow.

    Uses imshow so the density is rendered on a uniform pixel grid, matching
    the convention in Rezende & Mohamed (2015), Figure 3.

    Args:
        density: Density values, shape (nb_points, nb_points).
        lims:    Plot boundaries [[x_min, x_max], [y_min, y_max]].
        cmap:    Matplotlib colormap.
        title:   Figure title.
    """
    fig, ax = plt.subplots()
    ax.imshow(
        density,
        extent=[lims[0][0], lims[0][1], lims[1][0], lims[1][1]],
        cmap=cmap,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    return fig


def plot_flow_density(
    x: np.ndarray,
    y: np.ndarray,
    density: np.ndarray,
    lims: np.ndarray,
    cmap: str = "coolwarm",
    title: str = "Density",
) -> Figure:
    """
    Plot a flow-transformed density on a deformed grid using pcolormesh.

    The y-axis is flipped (* -1) so the orientation matches the imshow
    convention used for the target density (origin at top-left, y increasing
    downward).  This follows the scheme in flows_orig.py.

    Args:
        x:       Flow-transformed x-coordinates, shape (nb_points, nb_points).
        y:       Flow-transformed y-coordinates, shape (nb_points, nb_points).
        density: Density values on the deformed grid, shape (nb_points, nb_points).
        lims:    Plot boundaries [[x_min, x_max], [y_min, y_max]].
        cmap:    Matplotlib colormap.
        title:   Figure title.
    """
    fig, ax = plt.subplots()
    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[1])
    ax.pcolormesh(x, y * -1, density, cmap=cmap, rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    return fig
