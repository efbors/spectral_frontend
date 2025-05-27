import torch.nn as nn
import torch.nn.functional as F


class SpectralConvAutoencoder_3L12(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()

        # Input: [B, 1, 4, 180]
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=(5, 5), padding=(2, 2)),  # [B, 16, time, freq]
            nn.GroupNorm(4, 12),  # 4 groups for 16 channels
            nn.ReLU(),

            nn.Conv2d(12, 24, kernel_size=3, padding=1),  # [B, 32, 5, 180]
            nn.GroupNorm(4, 24),  # 4 groups for 16 channels
            nn.ReLU(),

            nn.Dropout(p=0.1),

            nn.Conv2d(24, latent_dim, kernel_size=3, stride=2, padding=1),  # [B, 128, 3, 90]
            nn.GroupNorm(4, latent_dim),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),  # [B, 32, 1, 1]
            nn.Flatten(),  # [B, 32]
            nn.Linear(latent_dim, latent_dim),  # [B, latent]
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, 180)
        )

    def forward(self, x):
        latents = self.encoder(x)  # [B, latent_dim]
        recon = self.decoder(latents)  # [B, 180]
        return recon, latents

    def loss(self, recon, target):
        return F.mse_loss(recon, target)
        # return F.smooth_l1_loss(recon, target, beta=0.01)
