import torch
import torch.nn as nn
from vaes.models.variational import reparameterization_trick


class LinearEncoder(nn.Module):
    """Encode images directly into the latent space of the linear autoencoder.

    Args:
        image_channels: Number of channels in the input image.
        image_size: Spatial size of the square input image.
        latent_dim: Size of the latent vector produced by the encoder.
    """

    def __init__(
        self,
        image_channels: int = 1,
        image_size: int = 32,
        latent_dim: int = 2
    ) -> None:
        """Build the multilayer perceptron used as the autoencoder encoder.

        Args:
            image_channels: Number of channels in the input image.
            image_size: Spatial size of the square input image.
            latent_dim: Size of the latent vector produced by the encoder.
        """
        super().__init__()
        input_dim = image_channels * image_size * image_size

        # Flatten the image and compress it into shared encoder features.
        self.network = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images into latent vectors.

        Args:
            x: Input tensor with shape ``[batch, channels, height, width]``.

        Returns:
            Latent vector for each input sample.
        """
        return self.network(x)
    



class LinearDecoder(nn.Module):
    """Decode a latent vector back into image space.

    Args:
        latent_dim: Size of the latent input vector.
        image_channels: Number of channels in the reconstructed image.
        image_size: Spatial size of the square reconstructed image.
    """

    def __init__(
        self,
        image_channels: int = 1,
        image_size: int = 32,
        latent_dim: int = 2
    ) -> None:
        """Build the multilayer perceptron used as the linear decoder.

        Args:
            latent_dim: Size of the latent input vector.
            image_channels: Number of channels in the reconstructed image.
            image_size: Spatial size of the square reconstructed image.
        """
        super().__init__()
        output_dim = image_channels * image_size * image_size

        # Expand the latent vector and restore the original image layout.
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid(),
            nn.Unflatten(dim=1, unflattened_size=(image_channels, image_size, image_size)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors into reconstructed images.

        Args:
            x: Latent tensor with shape ``[batch, latent_dim]``.

        Returns:
            Reconstructed image batch in image layout.
        """
        return self.network(x)


class LinearAutoencoder(nn.Module):
    """Combine a linear encoder and decoder into a deterministic autoencoder.

    Args:
        image_channels: Number of channels in the input and reconstructed image.
        latent_dim: Size of the latent vector used for reconstruction.
        image_size: Spatial size of the square input image.
    """

    def __init__(
        self,
        image_channels: int = 1,
        image_size: int = 32,
        latent_dim: int = 2
    ) -> None:
        """Create the linear autoencoder used for image reconstruction.

        Args:
            image_channels: Number of channels in the input and reconstructed image.
            latent_dim: Size of the latent vector used for reconstruction.
            image_size: Spatial size of the square input image.
        """
        super().__init__()

        # Encode the image
        self.encoder = LinearEncoder(image_size=image_size,
                                     latent_dim=latent_dim, 
                                     image_channels=image_channels)

        # Decode latent vectors back into image space.
        self.decoder = LinearDecoder(image_size=image_size,
                                     latent_dim=latent_dim, 
                                     image_channels=image_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of images and reconstruct them from the latent vectors.

        Args:
            x: Input image batch with shape ``[batch, channels, height, width]``.

        Returns:
            A tuple containing the latent vectors and the reconstructed images.
        """

        z = self.encoder(x)
        decoded = self.decoder(z)
        return z, decoded

class LinearVEncoder(nn.Module):
    """Encode images into the parameters of the latent Gaussian distribution.

    Args:
        image_channels: Number of channels in the input image.
        image_size: Spatial size of the square input image.
        latent_dim: Size of the latent distribution parameter vectors.
    """

    def __init__(
        self,
        image_channels: int = 1,
        image_size: int = 32,
        latent_dim: int = 2
    ) -> None:
        """Build the variational encoder that predicts ``mu`` and ``logvar``.

        Args:
            image_channels: Number of channels in the input image.
            image_size: Spatial size of the square input image.
            latent_dim: Size of the latent distribution parameter vectors.
        """
        super().__init__()
        input_dim = image_channels * image_size * image_size

        # Flatten the image and compress it into shared encoder features.
        self.network = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        # Predict the Gaussian parameters of the approximate posterior q(z|x).
        self.mu_head = nn.Linear(32, latent_dim)
        self.logvar_head = nn.Linear(32, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images into Gaussian distribution parameters.

        Args:
            x: Input tensor with shape ``[batch, channels, height, width]``.

        Returns:
            A tuple ``(mu, logvar)`` containing the mean and log-variance for
            each input sample.
        """
        x = self.network(x)

        mu = self.mu_head(x)
        logvar = self.logvar_head(x)

        return mu, logvar
    
class LinearVAE(nn.Module):
    """Combine the shared linear encoder and decoder into a variational autoencoder.

    Args:
        image_channels: Number of channels in the input and reconstructed image.
        latent_dim: Size of the sampled latent vector.
        image_size: Spatial size of the square input image.
    """

    def __init__(
        self,
        image_channels: int = 1,
        image_size: int = 32,
        latent_dim: int = 2
    ) -> None:
        """Create the linear VAE used for probabilistic latent modeling.

        Args:
            image_channels: Number of channels in the input and reconstructed image.
            latent_dim: Size of the sampled latent vector.
            image_size: Spatial size of the square input image.
        """
        super().__init__()

        # Extract deterministic features before predicting the latent distribution.
        self.vencoder = LinearVEncoder(image_size=image_size,
                                       latent_dim=latent_dim, 
                                       image_channels=image_channels)

        # Decode samples from the latent space back into images.
        self.decoder = LinearDecoder(image_size=image_size,
                                     latent_dim=latent_dim, 
                                     image_channels=image_channels)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode images, sample the latent space, and reconstruct the inputs.

        Args:
            x: Input image batch with shape ``[batch, channels, height, width]``.

        Returns:
            A tuple containing the sampled latent vectors, reconstruction, latent
            mean, and latent log-variance.
        """
        # Encode the image into deterministic intermediate features.
        mu, logvar = self.vencoder(x)        

        # Sample a latent vector and decode it back into pixel space.
        z = reparameterization_trick(mu, logvar)

        # Decode latent vectors back into image space.
        decoded = self.decoder(z)
        return z, decoded, mu, logvar
