from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable Conv1D block used by LCSMC-style modules."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.dw = nn.Conv1d(channels, channels, kernel_size, padding=pad, groups=channels, bias=False)
        self.pw = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return F.silu(x)


class AttentionBlock(nn.Module):
    """Lightweight channel-temporal attention block."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.temporal = nn.Conv1d(channels, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.mean(dim=-1)
        c = F.silu(self.fc1(c))
        c = torch.sigmoid(self.fc2(c)).unsqueeze(-1)
        t = torch.sigmoid(self.temporal(x))
        return x * c * t


class MultiScaleConvBlock(nn.Module):
    """Separable multiscale block in LCSMC style."""

    def __init__(self, channels: int, kernels: Sequence[int] = (3, 5, 7), reduction: int = 4) -> None:
        super().__init__()
        self.branches = nn.ModuleList([DepthwiseSeparableConv(channels, k) for k in kernels])
        self.merge = nn.Conv1d(channels * len(kernels), channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.attn = AttentionBlock(channels, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [b(x) for b in self.branches]
        y = torch.cat(outs, dim=1)
        y = self.merge(y)
        y = self.bn(y)
        y = F.silu(y)
        y = self.attn(y)
        return y + x


class LCSMCNetClassifier(nn.Module):
    """Baseline LCSMC classifier for multi-label outputs."""

    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        hidden_channels: int = 64,
        num_blocks: int = 3,
        reduction: int = 4,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [MultiScaleConvBlock(hidden_channels, kernels=(3, 5, 7), reduction=reduction) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        y = self.input_proj(x)
        for block in self.blocks:
            y = block(y)
        return self.head(y)
