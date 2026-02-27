import torch
import torch.nn as nn


class VAELoss(nn.Module):
    """
    VAE loss: reconstruction (MSE, summed per image, averaged over batch)
    + KL divergence (summed per image, averaged over batch).
    """

    def __init__(self, kl_weight=1.0, reconstruction_weight=1.0):
        super().__init__()
        self.kl_weight = kl_weight
        self.reconstruction_weight = reconstruction_weight

    def forward(self, x, x_hat, mu, logvar):
        # MSE per pixel, sum per image, mean over batch
        pixel_mse = ((x - x_hat) ** 2).flatten(1)
        reconstruction_loss = pixel_mse.sum(dim=-1).mean()

        # KL divergence per image, sum across latent dims, mean over batch
        kl = (1 + logvar - mu ** 2 - torch.exp(logvar)).flatten(1)
        kl_per_image = -0.5 * torch.sum(kl, dim=-1)
        kl_loss = torch.mean(kl_per_image)

        return self.reconstruction_weight * reconstruction_loss + self.kl_weight * kl_loss
