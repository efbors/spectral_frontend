import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConvAutoencoder_3L12DW(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()

        # Input: [B, 1, 4, 180]
        self.encoder = nn.Sequential(
            # Depthwise + pointwise conv1: (1 → 12)
            nn.Conv2d(1, 1, kernel_size=5, padding=2, groups=1),  # depthwise
            nn.Conv2d(1, 12, kernel_size=1),  # pointwise
            nn.GroupNorm(4, 12), # 4 groups for 12 channels
            nn.ReLU(),

            # Depthwise + pointwise conv2: (12 → 20)
            nn.Conv2d(12, 12, kernel_size=3, padding=1, groups=12),  # depthwise
            nn.Conv2d(12, 20, kernel_size=1),  # pointwise
            nn.GroupNorm(4, 20),  # 4 groups for 20 channels
            nn.ReLU(),

            nn.Dropout(p=0.1),

            nn.Conv2d(20, 20, kernel_size=3, stride=2, padding=1, groups=20),  # depthwise
            nn.Conv2d(20, latent_dim, kernel_size=1),  # pointwise
            #nn.Conv2d(20, latent_dim, kernel_size=3, stride=2, padding=1),  # [B, x,3,latent]
            nn.GroupNorm(4, latent_dim),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),  # [B, 32, 1, 1]
            nn.Flatten(),  # [B, latent]
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
        latents = self.encoder(x).to(torch.float16)  # cast here to float16
        recon = self.decoder(latents.to(torch.float32))  # decoder in float32
        return recon, latents

    def loss(self, recon, target):
        # use mse here since we're measuring IB and R2; ties better into variance-based metrics
        return F.mse_loss(recon, target)
        # return F.smooth_l1_loss(recon, target, beta=0.01)
