from __future__ import annotations

import torch
import torch.nn as nn


def build_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    name = cfg.training.optimizer.lower()
    kwargs = {"lr": cfg.training.lr, "weight_decay": cfg.training.weight_decay}
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), momentum=0.9, **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    return torch.optim.Adam(model.parameters(), **kwargs)


def fine_tune_personalized_model(client, cfg, device: torch.device) -> None:
    criterion = nn.BCEWithLogitsLoss()
    client.personalized_model.train()
    for _ in range(cfg.training.finetune_epochs):
        for xb, yb in client.train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            client.opt_personalized.zero_grad(set_to_none=True)
            logits = client.personalized_model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            client.opt_personalized.step()
