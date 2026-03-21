import math

import torch
from torch import nn
import torch.nn.functional as F


class DopplerGuidedLocalAttention(nn.Module):
    """Local BEV attention with Doppler similarity bias."""

    def __init__(self,
                 channels,
                 attn_channels=64,
                 kernel_size=3,
                 lambda_doppler=2.0,
                 sigma_doppler=0.5,
                 sigma_reliability=1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.lambda_doppler = lambda_doppler
        self.sigma_doppler = sigma_doppler
        self.sigma_reliability = sigma_reliability

        self.query = nn.Conv2d(channels, attn_channels, kernel_size=1, bias=False)
        self.key = nn.Conv2d(channels, attn_channels, kernel_size=1, bias=False)
        self.value = nn.Conv2d(channels, attn_channels, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(attn_channels, channels, kernel_size=1, bias=False)

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=self.padding)

    def forward(self, x, doppler_mean, doppler_std=None):
        """Apply local attention guided by Doppler consistency.

        Args:
            x (Tensor): Shape [B, C, H, W].
            doppler_mean (Tensor): Shape [B, 1, H, W].
            doppler_std (Tensor, optional): Shape [B, 1, H, W].
        """
        b, _, h, w = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn_channels = q.shape[1]
        q_flat = q.view(b, attn_channels, h * w).transpose(1, 2).unsqueeze(2)
        k_local = self.unfold(k).view(
            b, attn_channels, self.kernel_size * self.kernel_size, h * w
        ).permute(0, 3, 2, 1)
        v_local = self.unfold(v).view(
            b, attn_channels, self.kernel_size * self.kernel_size, h * w
        ).permute(0, 3, 2, 1)

        feat_score = (q_flat * k_local).sum(-1) / math.sqrt(attn_channels)

        center_d = doppler_mean.view(b, 1, h * w).transpose(1, 2).unsqueeze(-1)
        neigh_d = self.unfold(doppler_mean).view(
            b, 1, self.kernel_size * self.kernel_size, h * w
        ).permute(0, 3, 2, 1)
        doppler_score = -((center_d - neigh_d) ** 2).squeeze(-1)
        doppler_score = doppler_score / (2 * self.sigma_doppler * self.sigma_doppler)

        if doppler_std is not None:
            center_std = doppler_std.view(b, 1, h * w).transpose(1, 2).unsqueeze(-1)
            neigh_std = self.unfold(doppler_std).view(
                b, 1, self.kernel_size * self.kernel_size, h * w
            ).permute(0, 3, 2, 1)
            reliability = torch.exp(
                -((center_std + neigh_std).squeeze(-1) ** 2)
                / (2 * self.sigma_reliability * self.sigma_reliability)
            )
            doppler_score = doppler_score * reliability

        attn = F.softmax(feat_score + self.lambda_doppler * doppler_score, dim=-1)
        out = (attn.unsqueeze(-1) * v_local).sum(dim=2)
        out = out.transpose(1, 2).view(b, attn_channels, h, w)
        return x + self.out_proj(out)
