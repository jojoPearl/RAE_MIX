# src/utils/latent_mixer.py
import torch
import torch.nn as nn

class LatentMixer(nn.Module):
    """
    Latent-space ControlNet:
    z_cond = z + F(canny_latent)
    """
    def __init__(self, channels=768, init_std=0.02):
        super().__init__()
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.normal_(self.proj.weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.proj.bias)

    def forward(self, z, c, control_scale: float = 1.0):
        return z + control_scale * self.proj(c)
