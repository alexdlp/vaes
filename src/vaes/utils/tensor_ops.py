
import torch
import numpy as np

def pad_to_shape(

    x: torch.Tensor,
    max_L: int = 600,
    max_T: int | None = None,
) -> torch.Tensor:
    """
    Pad 1D or 2D tensor to a target shape using zeros.

    Supported shapes:
        (L,)           → padded to (max_L,)
        (L, T)         → padded to (max_L, max_T)

    Args:
        x:      1D or 2D tensor.
        max_L:  target first dimension.
        max_T:  target second dimension (only for 2D).

    Returns:
        Padded tensor with the same dtype, on same device.
    """
    if x.ndim == 1:
        L = x.shape[0]
        pad_L = max_L - L
        if pad_L < 0:
            return x[:max_L]
        return torch.nn.functional.pad(x, (0, pad_L), value=0.0)

    elif x.ndim == 2:
        if max_T is None:
            raise ValueError("max_T is required for 2D padding.")

        L, T = x.shape
        pad_L = max_L - L
        pad_T = max_T - T

        if pad_L < 0 or pad_T < 0:
            return x[:max_L, :max_T]

        return torch.nn.functional.pad(x, (0, pad_T, 0, pad_L), value=0.0)

    else:
        raise ValueError(f"pad_to_shape only supports 1D or 2D tensors, got shape {x.shape}")
    


def remove_padding(series: np.ndarray, pad_value: float = 0.0) -> np.ndarray:
    """
    Remove only trailing padding values from a 1D array (exact comparison).

    Preconditions for correctness:
      - Padding uses exactly `pad_value`.
      - The valid region does not end with `pad_value`.

    Args:
        series: 1D input array, possibly padded.
        pad_value: Value used for padding.

    Returns:
        Array without trailing padding. If the entire series is padding, returns empty.
    """
    series = np.asarray(series)
    if series.ndim != 1:
        raise ValueError("series must be 1D")

    if series.size == 0:
        return series.copy()

    is_pad = series == pad_value
    if np.all(is_pad):
        return np.array([], dtype=series.dtype)

    last_valid = np.max(np.nonzero(~is_pad))
    return series[: last_valid + 1].copy()