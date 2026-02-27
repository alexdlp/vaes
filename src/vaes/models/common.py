"""Shared utilities for autoencoder and VAE models."""

import torch


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Sample from a Gaussian using the reparameterization trick.

    Instead of sampling directly from N(mu, sigma^2), we sample epsilon ~ N(0, 1)
    and compute z = mu + sigma * epsilon. This keeps the computation graph
    differentiable through the sampling operation.

    Args:
        mu: Mean of the approximate posterior, shape ``(B, *)``.
        logvar: Log-variance of the approximate posterior, same shape as *mu*.

    Returns:
        A sample ``z`` from the distribution ``N(mu, exp(logvar))``.
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps
