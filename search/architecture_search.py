from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.default_config import MNASConfig
from models.supernet import SearchableLCSMCSupernet


def default_selected_ops(num_nodes: int, topk: int) -> list[list[str]]:
    base = ["conv_3", "skip"][: max(1, topk)]
    return [list(base) for _ in range(num_nodes)]


def search_personalized_architecture(
    train_loader: DataLoader,
    input_dim: int,
    num_labels: int,
    cfg: MNASConfig,
    op_cost: dict[str, float],
    device: torch.device,
) -> list[list[str]]:
    """Run MNAS search and return top-k operations per node."""

    if not cfg.model.use_mnas_search:
        return default_selected_ops(cfg.model.mnas_num_nodes, cfg.model.mnas_topk_ops)

    model = SearchableLCSMCSupernet(
        input_dim=input_dim,
        num_labels=num_labels,
        channels=cfg.model.hidden_channels,
        num_nodes=cfg.model.mnas_num_nodes,
        reduction=cfg.model.attention_reduction,
        op_cost=op_cost,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    opt_w = torch.optim.Adam(model.weight_parameters(), lr=cfg.model.search_lr_w)
    opt_a = torch.optim.Adam(model.architecture_parameters(), lr=cfg.model.search_lr_alpha)

    model.train()
    for _ in range(cfg.model.search_epochs):
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt_w.zero_grad(set_to_none=True)
            opt_a.zero_grad(set_to_none=True)
            logits, _ = model(xb, temperature=cfg.model.search_temperature)
            label_loss = criterion(logits, yb)
            latency = model.expected_latency()
            latency_penalty = (latency / max(cfg.model.latency_budget_t0, 1e-8)) ** cfg.model.search_latency_lambda
            loss = label_loss + latency_penalty
            loss.backward()
            opt_w.step()
            opt_a.step()

    return model.export_topk_architecture(k=cfg.model.mnas_topk_ops)
