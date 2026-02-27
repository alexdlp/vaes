import torch
import torch.nn as nn


class LinearVAE(nn.Module):

    def __init__(self, latent_dim=2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        self.fn_mu = nn.Linear(32, latent_dim)
        self.fn_logvar = nn.Linear(32, latent_dim)

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
        x = self.encoder(x)
        mu = self.fn_mu(x)
        logvar = self.fn_logvar(x)
        sigma = torch.exp(0.5 * logvar)
        noise = torch.randn_like(sigma, device=sigma.device)
        z = mu + sigma * noise
        return z, mu, logvar

    def forward_dec(self, x):
        return self.decoder(x)

    def forward(self, x):
        batch, channels, height, width = x.shape
        x = x.flatten(1)
        z, mu, logvar = self.forward_enc(x)
        dec = self.forward_dec(z)
        dec = dec.reshape(batch, channels, height, width)
        return z, dec, mu, logvar
