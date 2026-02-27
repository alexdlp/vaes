"""VAE loss combining reconstruction error and KL divergence."""

import torch
import torch.nn as nn


class VAELoss(nn.Module):
    """Combined reconstruction + KL divergence loss for Variational AutoEncoders.

    The reconstruction term is the sum of per-pixel MSE within each image,
    averaged across the batch.  The KL divergence term measures how far the
    learned posterior ``q(z|x) = N(mu, sigma^2)`` deviates from the standard
    normal prior ``p(z) = N(0, I)``, and is computed in closed form:

        KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    Both terms are computed per image and then averaged across the batch.
    The total loss is:

        L = reconstruction_weight * recon_loss + kl_weight * kl_loss

    Args:
        kl_weight: Scalar weight for the KL divergence term.  Higher values
            enforce a more regular latent space at the cost of reconstruction
            quality.
        reconstruction_weight: Scalar weight for the reconstruction term.
    """

    def __init__(self, kl_weight: float = 1.0, reconstruction_weight: float = 1.0) -> None:
        super().__init__()
        self.kl_weight = kl_weight
        self.reconstruction_weight = reconstruction_weight

    def forward(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the VAE loss.

        Args:
            x: Original images ``(B, C, H, W)``.
            x_hat: Reconstructed images ``(B, C, H, W)``.
            mu: Posterior mean ``(B, latent_dim)``.
            logvar: Posterior log-variance ``(B, latent_dim)``.

        Returns:
            A tuple ``(total_loss, components)`` where *components* is a dict
            with ``"reconstruction_loss"`` and ``"kl_loss"`` for logging.
        """
        # Reconstruction: per-pixel MSE, summed per image, averaged over batch
        pixel_mse = (x - x_hat) ** 2
        reconstruction_loss = pixel_mse.flatten(1).sum(dim=-1).mean()

        # KL divergence: closed-form for diagonal Gaussian vs N(0, I)
        kl_per_dim = 1 + logvar - mu.pow(2) - logvar.exp()
        kl_loss = -0.5 * kl_per_dim.sum(dim=-1).mean()

        total = self.reconstruction_weight * reconstruction_loss + self.kl_weight * kl_loss

        components = {
            "reconstruction_loss": reconstruction_loss.detach(),
            "kl_loss": kl_loss.detach(),
        }

        return total, components
