from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


OPS = ["conv_7", "conv_5", "conv_3", "conv_1", "max_pool", "avg_pool", "skip", "identity"]


class SharedKernelConv1d(nn.Module):
    """Compressed search space: one max-kernel tensor reused by smaller kernels."""

    def __init__(self, channels: int, max_kernel: int = 7) -> None:
        super().__init__()
        if max_kernel % 2 != 1:
            raise ValueError("max_kernel must be odd")
        self.max_kernel = max_kernel
        self.dw = nn.Conv1d(channels, channels, kernel_size=max_kernel, padding=max_kernel // 2, groups=channels, bias=False)
        self.pw = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        if kernel_size == self.max_kernel:
            y = self.dw(x)
        else:
            w = self.dw.weight
            center = self.max_kernel // 2
            half = kernel_size // 2
            ws = w[:, :, center - half:center + half + 1]
            y = F.conv1d(x, ws, bias=None, stride=1, padding=half, groups=x.shape[1])
        y = self.pw(y)
        y = self.bn(y)
        return F.silu(y)


class SearchOpBank(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.shared_conv = SharedKernelConv1d(channels, max_kernel=7)
        self.proj_pool = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, op_name: str) -> torch.Tensor:
        if op_name == "conv_7":
            return self.shared_conv(x, 7)
        if op_name == "conv_5":
            return self.shared_conv(x, 5)
        if op_name == "conv_3":
            return self.shared_conv(x, 3)
        if op_name == "conv_1":
            return self.shared_conv(x, 1)
        if op_name == "max_pool":
            return self.proj_pool(F.max_pool1d(x, kernel_size=3, stride=1, padding=1))
        if op_name == "avg_pool":
            return self.proj_pool(F.avg_pool1d(x, kernel_size=3, stride=1, padding=1))
        if op_name in {"skip", "identity"}:
            return x
        raise KeyError(op_name)


class SearchNode(nn.Module):
    """Single-path searchable node with hard gumbel routing."""

    def __init__(self, channels: int, op_names: Sequence[str]) -> None:
        super().__init__()
        self.op_names = list(op_names)
        self.bank = SearchOpBank(channels)
        self.alpha = nn.Parameter(torch.zeros(len(self.op_names), dtype=torch.float32))

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        gate = F.gumbel_softmax(self.alpha, tau=temperature, hard=True)
        idx = int(torch.argmax(gate).item())
        y = self.bank(x, self.op_names[idx])
        return y * gate[idx], gate

    def topk_ops(self, k: int) -> list[str]:
        probs = F.softmax(self.alpha.detach(), dim=0)
        top_idx = torch.topk(probs, k=min(k, len(self.op_names))).indices.tolist()
        return [self.op_names[i] for i in top_idx]
