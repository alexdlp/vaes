import torch
import torch.nn as nn
from vaes.models.variational import reparameterization_trick


class ConvEncoder(nn.Module):
    """Encode input images into a compact convolutional latent tensor.

    Args:
        in_channels: Number of channels in the input image.
        latent_channels: Number of channels produced in the latent tensor.
    """

    def __init__(self, in_channels: int = 1, latent_channels: int = 4) -> None:
        """Build the convolutional feature extractor used by the latent encoder.

        Args:
            in_channels: Number of channels in the input image.
            latent_channels: Number of channels in the encoded latent tensor.
        """
        super().__init__()

        # Downsample the image progressively until reaching the latent tensor.
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, latent_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images into a latent feature map.

        Args:
            x: Input image batch with shape ``[batch, channels, height, width]``.

        Returns:
            Encoded latent tensor produced by the convolutional stack.
        """
        return self.encoder_conv(x)


class ConvDecoder(nn.Module):
    """Decode a latent convolutional tensor back into image space.

    Args:
        out_channels: Number of channels expected in the reconstructed image.
        latent_channels: Number of channels present in the latent tensor.
    """

    def __init__(self, out_channels: int = 1, latent_channels: int = 4) -> None:
        """Build the transposed-convolution decoder used for reconstruction.

        Args:
            out_channels: Number of channels in the reconstructed image.
            latent_channels: Number of channels in the latent tensor.
        """
        super().__init__()

        # Mirror the encoder with transposed convolutions to recover image resolution.
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.ConvTranspose2d(8, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode a latent tensor into a batch of reconstructed images.

        Args:
            x: Latent tensor with convolutional shape
                ``[batch, latent_channels, latent_height, latent_width]``.

        Returns:
            Reconstructed image batch in the original image space.
        """
        return self.decoder_conv(x)


class ConvAutoencoder(nn.Module):
    """Combine a convolutional encoder and decoder into an autoencoder.

    Args:
        in_channels: Number of channels in the input images.
        latent_channels: Number of channels used in the latent tensor.
    """

    def __init__(self, in_channels: int = 1, latent_channels: int = 4) -> None:
        """Create the encoder-decoder pair used for deterministic reconstruction.

        Args:
            in_channels: Number of channels in the input images.
            latent_channels: Number of channels used in the latent tensor.
        """
        super().__init__()

        # Keep the submodules explicit so they can be reused by other models.
        self.encoder = ConvEncoder(in_channels=in_channels, latent_channels=latent_channels)
        self.decoder = ConvDecoder(out_channels=in_channels, latent_channels=latent_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode an input batch and reconstruct it from the latent tensor.

        Args:
            x: Input image batch with shape ``[batch, channels, height, width]``.

        Returns:
            A tuple containing the latent tensor and the reconstructed images.
        """
        # First compress the image into the latent representation.
        z = self.encoder(x)

        # Then reconstruct the image from that representation.
        dec = self.decoder(z)

        return z, dec


class ConvVAE(nn.Module):
    """Combine the reusable encoder/decoder pair into a variational autoencoder.

    Args:
        in_channels: Number of channels in the input images.
        latent_channels: Number of channels used in the latent tensor.
    """

    def __init__(self, in_channels: int = 1, latent_channels: int = 4) -> None:
        """Create the VAE using the shared convolutional encoder and decoder.

        Args:
            in_channels: Number of channels in the input images.
            latent_channels: Number of channels used in the latent tensor.
        """
        super().__init__()

        # Reuse the same feature extractor and image decoder as the plain autoencoder.
        self.encoder = ConvEncoder(in_channels=in_channels, latent_channels=latent_channels)
        self.decoder = ConvDecoder(out_channels=in_channels, latent_channels=latent_channels)

        # Predict the parameters of the latent Gaussian distribution from the encoder features.
        self.conv_mu = nn.Conv2d(latent_channels, latent_channels, kernel_size=3, stride=1, padding="same")
        self.conv_logvar = nn.Conv2d(latent_channels, latent_channels, kernel_size=3, stride=1, padding="same")



    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode the input, sample the latent space, and decode a reconstruction.

        Args:
            x: Input image batch with shape ``[batch, channels, height, width]``.

        Returns:
            A tuple containing the sampled latent tensor, reconstruction, latent
            mean, and latent log-variance.
        """
        # Extract deterministic convolutional features from the input image.
        features = self.encoder(x)

        # Turn those features into the parameters of the approximate posterior.
        mu = self.conv_mu(features)
        logvar = self.conv_logvar(features)

        # Sample a latent tensor and decode it into image space.
        z = reparameterization_trick(mu, logvar)
        dec = self.decoder(z)

        return z, dec, mu, logvar
