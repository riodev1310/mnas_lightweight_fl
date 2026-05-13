from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def adaptive_distillation_loss(
    logits_personal: torch.Tensor,
    logits_proxy: torch.Tensor,
    label_loss_personal: torch.Tensor,
    label_loss_proxy: torch.Tensor,
    temperature: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    ps = torch.sigmoid(logits_personal / temperature)
    pr = torch.sigmoid(logits_proxy / temperature)
    dist = F.mse_loss(ps, pr)
    return dist / (label_loss_personal.detach() + label_loss_proxy.detach() + eps)


def train_one_local_epoch_mutual(client, cfg, device: torch.device) -> dict[str, float]:
    criterion = nn.BCEWithLogitsLoss()
    client.personalized_model.train()
    client.proxy_model.train()
    total_personal = 0.0
    total_proxy = 0.0
    total_n = 0

    for xb, yb in client.train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        client.opt_personalized.zero_grad(set_to_none=True)
        client.opt_proxy.zero_grad(set_to_none=True)

        logits_p = client.personalized_model(xb)
        logits_r = client.proxy_model(xb)
        loss_p = criterion(logits_p, yb)
        loss_r = criterion(logits_r, yb)
        dist_w = adaptive_distillation_loss(
            logits_personal=logits_p,
            logits_proxy=logits_r,
            label_loss_personal=loss_p,
            label_loss_proxy=loss_r,
            temperature=cfg.training.distill_temperature,
        )
        total_loss_p = loss_p + cfg.training.distill_weight * dist_w
        total_loss_r = loss_r + cfg.training.distill_weight * dist_w

        total_loss_p.backward(retain_graph=True)
        total_loss_r.backward()
        client.opt_personalized.step()
        client.opt_proxy.step()

        bs = xb.size(0)
        total_personal += float(total_loss_p.item()) * bs
        total_proxy += float(total_loss_r.item()) * bs
        total_n += bs

    return {
        "loss_personal": total_personal / max(total_n, 1),
        "loss_proxy": total_proxy / max(total_n, 1),
    }
