import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_by_color(df: pd.DataFrame, color_by: str = "vector_id", num_groups: int = None, figsize: tuple = (12, 8)):
    """
    Plots x_position vs y_position points colored by a specified column.
    
    Args:
        df: DataFrame containing x_position and y_position columns.
        color_by: Column name to use for coloring ("vector_id", "polygon_id", etc.)
        num_groups: Number of groups to display (None = all)
        figsize: Figure size as (width, height) tuple
    """
    df_plot = df.copy()
    
    # Filter by number of groups if specified
    if num_groups is not None:
        unique_groups = df_plot[color_by].unique()[:num_groups]
        df_plot = df_plot[df_plot[color_by].isin(unique_groups)]
    
    # Create colormap
    unique_vals = df_plot[color_by].nunique()
    colors = plt.cm.tab20(np.linspace(0, 1, max(unique_vals, 20)))
    color_map = {val: colors[i % 20] for i, val in enumerate(df_plot[color_by].unique())}
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    for val in df_plot[color_by].unique():
        mask = df_plot[color_by] == val
        ax.scatter(
            df_plot.loc[mask, "x_position"],
            df_plot.loc[mask, "y_position"],
            c=[color_map[val]],
            s=0.5,
            alpha=0.7
        )
    
    ax.set_xlabel("X Position (mm)")
    ax.set_ylabel("Y Position (mm)")
    ax.set_title(f"Colored by: {color_by} ({unique_vals} groups)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()