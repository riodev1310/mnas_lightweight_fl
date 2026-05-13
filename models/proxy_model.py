from __future__ import annotations

import torch
import torch.nn as nn

from .lcsmc_net import AttentionBlock, DepthwiseSeparableConv


class ProxyLCSMC(nn.Module):
    """Unified proxy model with fixed architecture for federated aggregation."""

    def __init__(self, input_dim: int, num_labels: int, channels: int, reduction: int) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.SiLU(),
        )
        self.block1 = DepthwiseSeparableConv(channels, kernel_size=5)
        self.block2 = DepthwiseSeparableConv(channels, kernel_size=5)
        self.block3 = DepthwiseSeparableConv(channels, kernel_size=5)
        self.attn = AttentionBlock(channels, reduction=reduction)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        y = self.input_proj(x)
        y = self.block1(y)
        y = self.block2(y)
        y = self.block3(y)
        y = self.attn(y)
        return self.head(y)
