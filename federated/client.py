from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.evaluator import evaluate_model

from .mutual_learning import train_one_local_epoch_mutual


@dataclass
class MNASClient:
    client_id: int
    train_loader: DataLoader
    eval_loader: DataLoader
    num_samples: int
    selected_ops: list[list[str]]
    personalized_model: nn.Module
    proxy_model: nn.Module
    opt_personalized: torch.optim.Optimizer
    opt_proxy: torch.optim.Optimizer
    label_coverage: list[int]
    label_names: list[str]
    cfg: object
    device: torch.device

    def load_proxy_state(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.proxy_model.load_state_dict(state_dict, strict=True)

    def train_one_round(self, round_idx: int) -> dict[str, float]:
        losses = []
        for _ in range(self.cfg.federated.local_epochs_per_round):
            losses.append(train_one_local_epoch_mutual(self, cfg=self.cfg, device=self.device))
        return {
            "loss_personal": float(sum(x["loss_personal"] for x in losses) / max(len(losses), 1)),
            "loss_proxy": float(sum(x["loss_proxy"] for x in losses) / max(len(losses), 1)),
        }

    def evaluate_local(self, round_idx: int):
        return evaluate_model(
            self.personalized_model,
            self.eval_loader,
            threshold=self.cfg.evaluation.threshold,
            device=self.device,
            label_names=self.label_names,
        )

    def upload_proxy_state(self) -> tuple[dict[str, torch.Tensor], int]:
        return copy.deepcopy(self.proxy_model.state_dict()), int(self.num_samples)
