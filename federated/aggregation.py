from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class AggregationStrategy(ABC):
    @abstractmethod
    def aggregate(self, client_states: list[tuple[dict[str, torch.Tensor], int]]) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class FedAvgAggregation(AggregationStrategy):
    def aggregate(self, client_states: list[tuple[dict[str, torch.Tensor], int]]) -> dict[str, torch.Tensor]:
        return fedavg_proxy_states(client_states)


def fedavg_proxy_states(states: list[tuple[dict[str, torch.Tensor], int]]) -> dict[str, torch.Tensor]:
    """Aggregate proxy model states by sample-weighted FedAvg."""

    total = float(sum(n for _, n in states))
    if total <= 0:
        raise ValueError("Invalid total sample count for FedAvg")

    keys = list(states[0][0].keys())
    out: dict[str, torch.Tensor] = {}
    for key in keys:
        first = states[0][0][key]
        if torch.is_floating_point(first):
            acc = torch.zeros_like(first, dtype=torch.float32)
            for state_dict, n in states:
                acc += state_dict[key].float() * (float(n) / total)
            out[key] = acc.to(dtype=first.dtype, device=first.device)
        else:
            out[key] = first.clone()
    return out
