import torch
import torch.nn as nn


class ConvAutoEncoder(nn.Module):

    def __init__(self, in_channels=1, channels_bottleneck=4):
        super().__init__()

        self.bottleneck = channels_bottleneck
        self.in_channels = in_channels

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, self.bottleneck, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.bottleneck),
            nn.ReLU(),
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(self.bottleneck, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward_enc(self, x):
        return self.encoder_conv(x)

    def forward_dec(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.bottleneck, 4, 4)
        return self.decoder_conv(x)

    def forward(self, x):
        z = self.forward_enc(x)
        dec = self.forward_dec(z)
        return z, dec
