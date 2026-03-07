import torch 

def reparameterization_trick(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Sample a latent tensor while preserving gradient flow.

    The encoder does not output a single deterministic latent tensor in a VAE.
    Instead, it predicts the parameters of a Gaussian distribution for each
    latent location. This method converts those parameters into a sample
    ``z = mu + sigma * epsilon`` where ``epsilon`` is standard Gaussian noise.
    Writing the sample in this form keeps the stochasticity in ``epsilon`` and
    allows gradients to flow through ``mu`` and ``logvar`` during backpropagation.

    Args:
        mu: Mean of the approximate posterior distribution.
        logvar: Log-variance of the approximate posterior distribution.

    Returns:
        A sampled latent tensor with the same shape as ``mu`` and ``logvar``.
    """
    # Convert log-variance into standard deviation for sampling.
    sigma = torch.exp(0.5 * logvar)

    # Draw unit Gaussian noise with the same shape as the latent distribution.
    noise = torch.randn_like(sigma, device=sigma.device)

    # Shift and scale the noise to obtain a sample from N(mu, sigma^2).
    return mu + sigma * noise