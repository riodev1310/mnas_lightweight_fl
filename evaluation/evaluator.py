from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import EvaluationResult, compute_classification_metrics


def sanity_check_shapes(x: np.ndarray, y: np.ndarray, num_labels: int, feature_dim: int) -> None:
    if x.ndim != 2:
        raise ValueError(f"Expected X rank-2, got {x.ndim}")
    if y.ndim != 2:
        raise ValueError(f"Expected Y rank-2, got {y.ndim}")
    if x.shape[0] != y.shape[0]:
        raise ValueError("X/Y sample mismatch")
    if x.shape[1] != feature_dim:
        raise ValueError(f"Feature dim mismatch: {x.shape[1]} != {feature_dim}")
    if y.shape[1] != num_labels:
        raise ValueError(f"Label dim mismatch: {y.shape[1]} != {num_labels}")


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    eval_loader: DataLoader,
    threshold: float,
    device: torch.device,
    label_names: Sequence[str],
) -> EvaluationResult:
    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    all_probs = []
    all_targets = []
    running_loss = 0.0
    n_total = 0

    for xb, yb in eval_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        probs = torch.sigmoid(logits)
        all_probs.append(probs.detach().cpu().numpy())
        all_targets.append(yb.detach().cpu().numpy())
        bs = xb.size(0)
        running_loss += float(loss.item()) * bs
        n_total += bs

    probs_all = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    y_pred = (probs_all >= threshold).astype(np.int64)
    loss = running_loss / max(n_total, 1)
    return compute_classification_metrics(y_true, y_pred, loss=loss, label_names=label_names)


class PaperFaithfulClientEvaluation:
    """Evaluate personalized client models after server proxy aggregation."""

    def evaluate_after_round(self, clients, round_idx: int) -> dict[int, EvaluationResult]:
        return {client.client_id: client.evaluate_local(round_idx) for client in clients}
