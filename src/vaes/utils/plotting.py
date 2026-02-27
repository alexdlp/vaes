import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Union, Optional
import pandas as pd

# def cube_validation_plots(
#     ref1_series: np.ndarray,
#     ref2_series: np.ndarray,
#     predicted_ref1: np.ndarray,
#     predicted_ref2: np.ndarray,
#     predicted_mean: np.ndarray,
#     cube: Union[int, str],
#     layer: Union[int, str],
#     vector_key: str,
# ) -> Figure:
#     """
#     Generate a 2x3 comparison plot for original and predicted meltpool time series
#     from an autoencoder.

#     The upper row visualizes:
#         - Original vs predicted Ref1
#         - Original vs predicted Ref2
#         - Original vs predicted mean (Ref1+Ref2)/2

#     The lower row visualizes:
#         - Predicted difference (Pred1 - Pred2)
#         - Residuals for each reference
#         - Residuals of the mean series

#     Args:
#         ref1_series (np.ndarray): Original meltpool sequence for reference 1.
#         ref2_series (np.ndarray): Original meltpool sequence for reference 2.
#         predicted_ref1 (np.ndarray): Autoencoder reconstruction for ref1.
#         predicted_ref2 (np.ndarray): Autoencoder reconstruction for ref2.
#         predicted_mean (np.ndarray): Autoencoder reconstruction of the mean.
#         cube (int | str): Cube identifier for the plot title.
#         layer (int | str): Layer identifier for the plot title.
#         vector_key (str): Vector identifier used in titles.

#     Returns:
#         matplotlib.figure.Figure: The generated matplotlib figure.
#     """

#     # ----- Upper y-limits -----
#     y_min_upper = int(min(ref1_series.min(), ref2_series.min(), predicted_ref1.min(), predicted_ref2.min()) * 1.2)
#     y_max_upper = int(max(ref1_series.max(), ref2_series.max(), predicted_ref1.max(), predicted_ref2.max()) * 1.1)  
#     y_min_upper = -50 if y_min_upper == 0 else y_min_upper  
                    
#     # ----- Create figure -----
#     fig, axs = plt.subplots(2, 3, figsize=(20, 12), sharey=False, sharex = False)
#     fig.suptitle(f"Cube {cube} | Layer {layer} | {vector_key} - Interpolated Series",fontsize=16, y = 0.95)

#     # -------------------------------------------------------------------------
#     # Row 1: Originals vs Predictions
#     # -------------------------------------------------------------------------

#     # Ref1
#     axs[0, 0].plot(ref1_series, "-", color="blue", alpha=0.7, label="Original Ref1")
#     axs[0, 0].plot(predicted_ref1, "-", color="#66B2FF", alpha=1, label="Predicted Ref1")
#     axs[0, 0].set_title("Reference 1", fontsize=14)
#     axs[0, 0].set_xlabel("Time Step", fontsize=12)
#     axs[0, 0].set_ylabel("Meltpool Value", fontsize=12)
#     axs[0, 0].grid(alpha=0.3)
#     axs[0, 0].legend()

#     # Ref2
#     axs[0, 1].plot(ref2_series, "-", color="red", alpha=0.7, label="Original Ref2")
#     axs[0, 1].plot(predicted_ref2, "-", color="#CC4C4C", alpha=1, label="Predicted Ref2")
#     axs[0, 1].set_title("Reference 2", fontsize=14)
#     axs[0, 1].set_xlabel("Time Step", fontsize=12)
#     axs[0, 1].set_ylabel("Meltpool Value", fontsize=12)
#     axs[0, 1].grid(alpha=0.3)
#     axs[0, 1].legend()
    
#     # Mean
#     axs[0,2].plot((ref1_series+ref2_series)/2, linestyle="-", color="#2E8B57", alpha=0.7, label="Original Ref1&Ref2 Mean")
#     axs[0,2].plot(predicted_mean, linestyle="-", color="#66CDAA", alpha=1, label="Autoencoder Predicted Mean")
#     axs[0,2].set_title("Mean", fontsize=14)
#     axs[0,2].set_xlabel("Time Step", fontsize=12)
#     axs[0,2].set_ylabel("Meltpool Value", fontsize=12)
#     axs[0,2].grid(alpha=0.3)
#     axs[0,2].legend()

#     # Assign limits to the upper plots (original vs. predicted)
#     axs[0, 0].set_ylim(y_min_upper, y_max_upper)
#     axs[0, 1].set_ylim(y_min_upper, y_max_upper)
#     axs[0, 2].set_ylim(y_min_upper, y_max_upper)

#     # -------------------------------------------------------------------------
#     # Row 2: Differences & Residuals
#     # -------------------------------------------------------------------------
#     predicted_diff = predicted_ref1 - predicted_ref2
#     residuals_ref1 = ref1_series - predicted_ref1
#     residuals_ref2 = ref2_series - predicted_ref2

#     mean_residuals = ((ref1_series+ref2_series)/2) - predicted_mean

#     # Compute axis limits for lower plots (difference & residuals)
#     y_min_lower = int(min(predicted_diff.min(), residuals_ref1.min(), residuals_ref2.min()) * 1.1)
#     y_max_lower = int(max(predicted_diff.max(), residuals_ref1.max(), residuals_ref2.max()) * 1.1)

#     # Difference plot
#     axs[1, 0].plot(predicted_diff, linestyle="-", color="purple", alpha=0.8, label="Difference (Pred1 - Pred2)")
#     axs[1, 0].axhline(0, color="black", linestyle="--", alpha=0.5)
#     axs[1, 0].set_title("Difference between Predicted Series", fontsize=14)
#     axs[1, 0].set_xlabel("Time Step", fontsize=12)
#     axs[1, 0].set_ylabel("Difference Value", fontsize=12)
#     axs[1, 0].grid(alpha=0.3)
#     axs[1, 0].legend()

#     # Residuals
#     axs[1, 1].plot(residuals_ref1, linestyle="-", color="blue", alpha=0.7, label="Residuals Ref1 (Orig1 - Pred1)")
#     axs[1, 1].plot(residuals_ref2, linestyle="-", color="red", alpha=0.7, label="Residuals Ref2 (Orig2 - Pred2)")
#     axs[1, 1].axhline(0, color="black", linestyle="--", alpha=0.5)
#     axs[1, 1].set_title("Residuals (Original - Predicted)", fontsize=14)
#     axs[1, 1].set_xlabel("Time Step", fontsize=12)
#     axs[1, 1].set_ylabel("Residual Value", fontsize=12)
#     axs[1, 1].grid(alpha=0.3)
#     axs[1, 1].legend()

#     # Residuals mean
#     axs[1, 2].plot(mean_residuals, linestyle="-", color="#2E8B57", alpha=0.7, label="Residuals Mean (Orig - Predicted)")
#     axs[1, 2].axhline(0, color="black", linestyle="--", alpha=0.5)
#     axs[1, 2].set_title("Residuals (Orig Mean - Predicted Mean)", fontsize=14)
#     axs[1, 2].set_xlabel("Time Step", fontsize=12)
#     axs[1, 2].set_ylabel("Residual Value", fontsize=12)
#     axs[1, 2].grid(alpha=0.3)
#     axs[1, 2].legend()

#     # Assign limits to the lower plots (difference & residuals)
#     axs[1, 0].set_ylim(y_min_lower, y_max_lower)
#     axs[1, 1].set_ylim(y_min_lower, y_max_lower)

#     fig.tight_layout(rect=[0, 0, 1, 0.96])

#     return fig

def cube_validation_plots(
    ref1_series: np.ndarray,
    ref2_series: np.ndarray,
    predicted_ref1: np.ndarray,
    predicted_ref2: np.ndarray,
    cube: Union[int, str],
    layer: Union[int, str],
    vector_key: str,
    predicted_mean: Optional[np.ndarray] = None,
) -> Figure:
    """
    Validation plot for cube vectors (two references).

    If predicted_mean is None, produce a 2x2 layout.
    If predicted_mean is provided, produce a 2x3 layout.
    """

    # ------------------------------------------------------------
    # Detectar si hay medias
    # ------------------------------------------------------------
    has_mean = predicted_mean is not None

    # ----- Upper y-limits -----
    y_min_upper = int(min(ref1_series.min(), ref2_series.min(),
                          predicted_ref1.min(), predicted_ref2.min()) * 1.2)
    y_max_upper = int(max(ref1_series.max(), ref2_series.max(),
                          predicted_ref1.max(), predicted_ref2.max()) * 1.1)
    y_min_upper = -50 if y_min_upper == 0 else y_min_upper

    # ------------------------------------------------------------
    # Layout dinámico
    # ------------------------------------------------------------
    if has_mean:
        fig, axs = plt.subplots(2, 3, figsize=(20, 12))
    else:
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))

    fig.suptitle(f"Cube {cube} | Layer {layer} | {vector_key} - Interpolated Series",
                 fontsize=16, y=0.95)

    # ------------------------------------------------------------
    # Row 1 — Originals vs Predictions
    # ------------------------------------------------------------
    # Ref1
    axs[0, 0].plot(ref1_series, "-", color="blue", alpha=0.7, label="Original Ref1")
    axs[0, 0].plot(predicted_ref1, "-", color="#66B2FF", alpha=1, label="Predicted Ref1")
    axs[0, 0].set_title("Reference 1")
    axs[0, 0].grid(alpha=0.3)
    axs[0, 0].legend()

    # Ref2
    axs[0, 1].plot(ref2_series, "-", color="red", alpha=0.7, label="Original Ref2")
    axs[0, 1].plot(predicted_ref2, "-", color="#CC4C4C", alpha=1, label="Predicted Ref2")
    axs[0, 1].set_title("Reference 2")
    axs[0, 1].grid(alpha=0.3)
    axs[0, 1].legend()

    # Mean (solo si existe)
    if has_mean:
        axs[0, 2].plot((ref1_series + ref2_series) / 2, "-", color="#2E8B57", alpha=0.7)
        axs[0, 2].plot(predicted_mean, "-", color="#66CDAA", alpha=1)
        axs[0, 2].set_title("Mean")
        axs[0, 2].grid(alpha=0.3)
        axs[0, 2].legend()

    # Set limits (solo a los que existan)
    axs[0, 0].set_ylim(y_min_upper, y_max_upper)
    axs[0, 1].set_ylim(y_min_upper, y_max_upper)
    if has_mean:
        axs[0, 2].set_ylim(y_min_upper, y_max_upper)

    # ------------------------------------------------------------
    # Row 2 — Differences & Residuals
    # ------------------------------------------------------------
    predicted_diff = predicted_ref1 - predicted_ref2
    residuals_ref1 = ref1_series - predicted_ref1
    residuals_ref2 = ref2_series - predicted_ref2

    y_min_lower = int(min(predicted_diff.min(),
                          residuals_ref1.min(),
                          residuals_ref2.min()) * 1.1)
    y_max_lower = int(max(predicted_diff.max(),
                          residuals_ref1.max(),
                          residuals_ref2.max()) * 1.1)

    # Difference plot
    axs[1, 0].plot(predicted_diff, "-", color="purple", alpha=0.8)
    axs[1, 0].axhline(0, color="black", linestyle="--", alpha=0.5)
    axs[1, 0].set_title("Difference (Pred1 - Pred2)")
    axs[1, 0].grid(alpha=0.3)

    # Residuals
    axs[1, 1].plot(residuals_ref1, "-", color="blue", alpha=0.7)
    axs[1, 1].plot(residuals_ref2, "-", color="red", alpha=0.7)
    axs[1, 1].axhline(0, color="black", linestyle="--", alpha=0.5)
    axs[1, 1].set_title("Residuals")
    axs[1, 1].grid(alpha=0.3)

    # Residual mean only if available
    if has_mean:
        axs[1, 2].plot(((ref1_series + ref2_series) / 2) - predicted_mean,
                       "-", color="#2E8B57", alpha=0.7)
        axs[1, 2].axhline(0, color="black", linestyle="--", alpha=0.5)
        axs[1, 2].set_title("Residuals Mean")
        axs[1, 2].grid(alpha=0.3)

    # Limits
    axs[1, 0].set_ylim(y_min_lower, y_max_lower)
    axs[1, 1].set_ylim(y_min_lower, y_max_lower)
    if has_mean:
        axs[1, 2].set_ylim(y_min_lower, y_max_lower)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def comb_validation_plots(
    target_series: np.ndarray,
    predicted_series: np.ndarray,
    file_name: str,
    segment: str,
    vector_key: str,
    title_prefix: str = "Comb Validation",
) -> Figure:
    """
    Validation plot for comb vectors (single target series).
    """
    target = np.asarray(target_series, dtype=np.float32).reshape(-1)
    predicted = np.asarray(predicted_series, dtype=np.float32).reshape(-1)
    common_len = min(target.shape[0], predicted.shape[0])

    target = target[:common_len]
    predicted = predicted[:common_len]
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    fig.suptitle(f"{title_prefix} | {file_name} | {segment} | vector_{vector_key}", fontsize=16, y=0.98)

    ax.plot(target, "-", color="#1f77b4", alpha=0.85, label="Target")
    ax.plot(predicted, "-", color="#d62728", alpha=0.75, label="Prediction")
    ax.set_title("Target vs Prediction", fontsize=14)
    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Meltpool Value", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def plot_curves_from_mlflow(
    df: pd.DataFrame,
    title: str,
    metric_col: str = "metric",
    value_col: str = "value",
    step_col: str = "step",
    train_metric: str = "train_loss",
    val_metric: str = "val_loss",
) -> None:
    """Plot train/val metrics vs step; include minima (3 decimals) in legend."""
    df_train = df[df[metric_col] == train_metric]
    df_val = df[df[metric_col] == val_metric]

    train_min = df_train[value_col].min()
    val_min = df_val[value_col].min()

    plt.figure(figsize=(12, 6))

    plt.plot(
        df_train[step_col],
        df_train[value_col],
        label=f"{train_metric} (min={train_min:.3f})",
    )
    plt.plot(
        df_val[step_col],
        df_val[value_col],
        label=f"{val_metric} (min={val_min:.3f})",
    )

    plt.xscale("linear")
    plt.yscale("log")

    plt.xlabel("Epochs")
    plt.ylabel("Loss (log)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
