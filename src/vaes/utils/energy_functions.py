import numpy as np
import torch


# ---------------------------------------------------------------------------
# Auxiliary warp functions (Table 1, Rezende & Mohamed, 2015)
# ---------------------------------------------------------------------------
# These functions define curved manifolds in the z_0 dimension that shape
# the energy landscape of U2, U3 and U4 into non-Gaussian geometries.

def w1(z: torch.Tensor) -> torch.Tensor:
    """Sinusoidal warp: produces curved ridges in the energy landscape."""
    return torch.sin((2 * np.pi * z[:, 0]) / 4)


def w2(z: torch.Tensor) -> torch.Tensor:
    """Gaussian bump warp: introduces a localised mode offset near z_0 = 1."""
    return 3 * torch.exp(-0.5 * ((z[:, 0] - 1) / 0.6) ** 2)


def w3(z: torch.Tensor) -> torch.Tensor:
    """Sigmoid ramp warp: introduces a smooth step-shaped mode offset near z_0 = 1."""
    return 3 * torch.sigmoid((z[:, 0] - 1) / 0.3)


# ---------------------------------------------------------------------------
# Energy functions (Table 1, Rezende & Mohamed, 2015)
# ---------------------------------------------------------------------------
# Each function U_i defines an unnormalised target density via:
#
#       p(z) ∝ exp(-U(z))
#
# These are used to benchmark the representative power of normalizing flows
# against distributions that exhibit characteristics such as multimodality,
# periodicity and curved manifolds — none of which can be captured by a
# standard diagonal Gaussian posterior.
#
# All functions accept a batch of 2D points z of shape (N, 2), where
# z[:, 0] = z_1 and z[:, 1] = z_2 in the paper's notation.

def U1(z: torch.Tensor) -> torch.Tensor:
    """
    Ring-shaped distribution with two modes along the z_1 axis.

    The first term enforces a ring of radius 2. The second term introduces
    two modes at z_1 = ±2 via a log-sum-exp of two Gaussians.
    """
    r = torch.sqrt(z[:, 0] ** 2 + z[:, 1] ** 2)
    ring = 0.5 * ((r - 2) / 0.4) ** 2

    log_mix = torch.stack([
        -0.5 * ((z[:, 0] - 2) / 0.6) ** 2,
        -0.5 * ((z[:, 0] + 2) / 0.6) ** 2,
    ], dim=1).logsumexp(dim=1)

    return ring - log_mix


def U2(z: torch.Tensor) -> torch.Tensor:
    """
    Thin curved ridge following z_2 = w1(z).

    Produces a sinusoidal banana-shaped distribution that is impossible to
    capture with an axis-aligned Gaussian approximation.
    """
    return 0.5 * ((z[:, 1] - w1(z)) / 0.4) ** 2


def U3(z: torch.Tensor) -> torch.Tensor:
    """
    Bimodal distribution along a sinusoidal manifold.

    Two modes separated by w2(z) along the z_2 axis, both following the
    curved ridge defined by w1(z).
    """
    return -torch.stack([
        -0.5 * ((z[:, 1] - w1(z)) / 0.35) ** 2,
        -0.5 * ((z[:, 1] - w1(z) + w2(z)) / 0.35) ** 2,
    ], dim=1).logsumexp(dim=1)


def U4(z: torch.Tensor) -> torch.Tensor:
    """
    Bimodal distribution with a sigmoid-shaped mode offset.

    Similar to U3 but the second mode offset is defined by the smoother
    sigmoid ramp w3(z) instead of the localised Gaussian bump w2(z).
    """
    return -torch.stack([
        -0.5 * ((z[:, 1] - w1(z)) / 0.4) ** 2,
        -0.5 * ((z[:, 1] - w1(z) + w3(z)) / 0.35) ** 2,
    ], dim=1).logsumexp(dim=1)


# Convenience mapping from index to function, matching Table 1 of the paper
ENERGY_FUNCTIONS = {1: U1, 2: U2, 3: U3, 4: U4}