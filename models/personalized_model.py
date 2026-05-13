from __future__ import annotations

import torch
import torch.nn as nn

from .lcsmc_net import AttentionBlock
from .search_space import SearchOpBank


class PersonalizedSearchedLCSMC(nn.Module):
    """Personalized model constructed from searched top-k ops per node."""

    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        channels: int,
        reduction: int,
        selected_ops: list[list[str]],
    ) -> None:
        super().__init__()
        self.selected_ops = selected_ops
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.SiLU(),
        )
        self.banks = nn.ModuleList([SearchOpBank(channels) for _ in selected_ops])
        self.post_attn = AttentionBlock(channels, reduction=reduction)
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
        y = self.stem(x)
        for bank, ops in zip(self.banks, self.selected_ops):
            outs = [bank(y, op) for op in ops]
            y = y + torch.stack(outs, dim=0).mean(dim=0)
        y = self.post_attn(y)
        return self.head(y)
