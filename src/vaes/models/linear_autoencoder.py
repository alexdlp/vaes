import torch
import torch.nn as nn


class LinearAutoEncoder(nn.Module):

    def __init__(self, latent_dim=2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32 * 32),
            nn.Sigmoid(),
        )

    def forward_enc(self, x):
        return self.encoder(x)

    def forward_dec(self, x):
        return self.decoder(x)

    def forward(self, x):
        batch, channels, height, width = x.shape
        x = x.flatten(1)
        z = self.forward_enc(x)
        dec = self.forward_dec(z)
        dec = dec.reshape(batch, channels, height, width)
        return z, dec
