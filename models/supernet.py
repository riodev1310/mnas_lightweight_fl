from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lcsmc_net import AttentionBlock
from .search_space import OPS, SearchNode


class SearchableLCSMCSupernet(nn.Module):
    """LCSMC-based supernet for MNAS search."""

    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        channels: int,
        num_nodes: int,
        reduction: int,
        op_cost: dict[str, float],
    ) -> None:
        super().__init__()
        self.channels = channels
        self.num_nodes = num_nodes
        self.op_cost = dict(op_cost)
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.SiLU(),
        )
        self.nodes = nn.ModuleList([SearchNode(channels, OPS) for _ in range(num_nodes)])
        self.post_attn = AttentionBlock(channels, reduction=reduction)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, num_labels),
        )

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        y = self.stem(x)
        gates = []
        for node in self.nodes:
            out, gate = node(y, temperature=temperature)
            y = y + out
            gates.append(gate)
        y = self.post_attn(y)
        return self.head(y), gates

    def architecture_parameters(self) -> list[nn.Parameter]:
        return [n.alpha for n in self.nodes]

    def weight_parameters(self) -> list[nn.Parameter]:
        arch_ids = {id(p) for p in self.architecture_parameters()}
        return [p for p in self.parameters() if id(p) not in arch_ids]

    def expected_latency(self) -> torch.Tensor:
        lat = torch.tensor(0.0, device=next(self.parameters()).device)
        for node in self.nodes:
            probs = F.softmax(node.alpha, dim=0)
            op_lat = torch.tensor([self.op_cost[o] for o in node.op_names], dtype=torch.float32, device=probs.device)
            lat = lat + (probs * op_lat).sum()
        return lat

    def export_topk_architecture(self, k: int) -> list[list[str]]:
        return [node.topk_ops(k) for node in self.nodes]
