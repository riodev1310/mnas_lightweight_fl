from __future__ import annotations

import copy

import torch
from models.proxy_model import ProxyLCSMC

from .aggregation import AggregationStrategy


class MNASServer:
    def __init__(
        self,
        *,
        input_dim: int,
        num_labels: int,
        cfg,
        device: torch.device,
        aggregation_strategy: AggregationStrategy,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.aggregation_strategy = aggregation_strategy
        self.global_model = ProxyLCSMC(
            input_dim=input_dim,
            num_labels=num_labels,
            channels=cfg.model.hidden_channels,
            reduction=cfg.model.attention_reduction,
        ).to(device)
        self.optimizer_state_dict = None
        self.last_aggregation_result: dict[str, object] = {}

    def distribute_to_clients(self, clients: list) -> None:
        state = copy.deepcopy(self.global_model.state_dict())
        for client in clients:
            client.load_proxy_state(state)

    def aggregate(self, clients: list, round_idx: int) -> dict[str, object]:
        uploaded = [client.upload_proxy_state() for client in clients]
        new_state = self.aggregation_strategy.aggregate(uploaded)
        self.global_model.load_state_dict(new_state, strict=True)
        self.last_aggregation_result = {
            "round_idx": int(round_idx),
            "active_clients": len(clients),
            "aggregated_samples": int(sum(n for _, n in uploaded)),
        }
        return self.last_aggregation_result

    def architecture_config(self) -> dict[str, object]:
        return {
            "model": "ProxyLCSMC",
            "hidden_channels": int(self.cfg.model.hidden_channels),
            "attention_reduction": int(self.cfg.model.attention_reduction),
            "aggregation": str(self.cfg.federated.aggregation),
        }
