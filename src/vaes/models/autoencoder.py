from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F



class ConvBlock(nn.Module):
    """
    Reusable convolutional block: Conv1d -> Activation -> BatchNorm
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of convolutional kernel
        activation (str): Activation function ('relu', 'leaky_relu', 'elu')
        padding (str or int): Padding mode
    """

    def __init__(self, in_channels: int, out_channels : int, kernel_size: int = 5,
                 activation = 'relu', padding = 'same'):
        super().__init__()
         
        self.conv = nn.Conv1d(
             in_channels=in_channels,
             out_channels=out_channels,
             kernel_size=kernel_size,
             padding=padding)
        
        self.bn = nn.BatchNorm1d(out_channels * 2, eps=1e-3, momentum=0.01)

        # Select activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.conv(x)
        x = self.activation(x)
        x = self.bn(x)
        
        return x


class Encoder(nn.Module):
    """
    Convolutional encoder matching the original TensorFlow structure.

    Args:
        input_dim (Tuple[int, int]): (C, L) input shape.
        conv_filters (int): Base number of convolutional filters.
        latent_dim (int): Dimension of latent vector.
    """

    def __init__(self, input_dim: Tuple[int, int], conv_filters: int, latent_dim: int) -> None:
        super().__init__()

        CHANNELS, LENGTH = input_dim
        self.input_length = LENGTH
        self.conv_filters = conv_filters

        # 1. Primera Convolución (encoder_conv1 + batch_normalization)
        self.enc_conv1 = nn.Conv1d(in_channels=CHANNELS, out_channels=conv_filters, 
                                   kernel_size=5, padding="same")
        self.enc_bn1 = nn.BatchNorm1d(conv_filters, eps=1e-3, momentum=0.01)
        
        # 2. Segunda Convolución (encoder_conv2 + batch_normalization_1)
        self.enc_conv2 = nn.Conv1d(in_channels=conv_filters, out_channels=conv_filters * 2, 
                                   kernel_size=5, padding="same")
        self.enc_bn2 = nn.BatchNorm1d(conv_filters * 2, eps=1e-3, momentum=0.01)
        
        # 3. Pooling (TF: max_pooling1d)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # 4. Dense Latente (flatten + encoder_dense)
        self.fc_latent = nn.LazyLinear(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Bloque 1
        x = self.enc_conv1(x)
        x = F.relu(x)
        x = self.enc_bn1(x)

        # Bloque 2
        x = self.enc_conv2(x)
        x = F.relu(x)
        x = self.enc_bn2(x)

        # Max Pooling (600 -> 300)
        x = self.pool(x)

        # Flatten
        x = torch.flatten(x, start_dim=1)

        # sigmoid
        latent = torch.sigmoid(self.fc_latent(x))

        return latent


class Decoder(nn.Module):
    """
    Convolutional decoder (mirror of Encoder).

    Args:
        output_dim (Tuple[int, int]): (C, L) output shape.
        conv_filters (int): Base number of convolutional filters.
        latent_dim (int): Dimension of latent vector.
    """

    def __init__(self, output_dim: Tuple[int, int], conv_filters: int, latent_dim: int) -> None:
        super().__init__()

        CHANNELS, LENGTH = output_dim
        self.output_length = LENGTH
        self.conv_filters = conv_filters
        self.pooled_length = LENGTH // 2

        self.decoder_filters = conv_filters * 2

        # 1. expansion de las latentes -> (B, 256, 300)
        self.fc_decode = nn.LazyLinear(conv_filters * 2 * self.pooled_length)

        # 2. Conv -> BN -> RELU
        self.dec_conv1 = nn.Conv1d(in_channels=self.decoder_filters, 
                                   out_channels=self.decoder_filters, 
                                   kernel_size=5, 
                                   padding="same")
        self.dec_bn1 = nn.BatchNorm1d(self.decoder_filters, eps=1e-3, momentum=0.01)

        # 3. Upsample; linear en lugar de nearest para mayor suavizado -> (B, 256, 600)
        self.upsample = nn.Upsample(scale_factor=2)
        
        # 4. Conv -> BN -> RELU
        self.dec_conv2 = nn.Conv1d(in_channels=self.decoder_filters, 
                                   out_channels=conv_filters, 
                                   kernel_size=5, 
                                   padding="same")
        self.dec_bn2 = nn.BatchNorm1d(conv_filters, eps=1e-3, momentum=0.01)

        # Capa final para obtener el número correcto de canales de salida
        self.conv_out = nn.Conv1d(in_channels=conv_filters, 
                                  out_channels=CHANNELS, 
                                  kernel_size=5, 
                                  padding="same")

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # 1. Dense y Reshape
        x = F.relu(self.fc_decode(latent))
        x = x.view(-1, self.decoder_filters, self.pooled_length) # (Batch, 256, 300)
        
        # 2. Primera Convolución (en 300)
        x = self.dec_conv1(x)
        x = F.relu(x)
        x = self.dec_bn1(x)
        
        # 3. Upsample (300 -> 600)
        x = self.upsample(x) # (Batch, 256, 600)
        
        # 4. Segunda Convolución (en 600)
        x = self.dec_conv2(x)
        x = F.relu(x)
        x = self.dec_bn2(x)
        
        # 5. Salida
        x = self.conv_out(x) 
        
        return x


class Autoencoder(nn.Module):
    """
    Complete convolutional autoencoder.

    Args:
        input_dim (Tuple[int, int]): Input shape (L, C).
        conv_filters (int): Base num. of filters.
        latent_dim (int): Latent vector dimension.
    """

    def __init__(
        self,
        input_dim: Tuple[int, int],
        conv_filters: int,
        latent_dim: int
    ) -> None:
        super().__init__()
        self.encoder = Encoder(input_dim, conv_filters, latent_dim)
        self.decoder = Decoder(input_dim, conv_filters, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full AE forward pass.

        Args:
            x (torch.Tensor): shape (B, L, C)

        Returns:
            Tuple[latent, reconstruction]
        """
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon
    

