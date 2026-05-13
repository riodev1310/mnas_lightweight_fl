from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from config.default_config import MNASConfig

from .client_checkpoint import build_client_checkpoint


class CheckpointManager:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)

    def client_checkpoint_path(self, mode: str, num_clients: int, round_idx: int, client_id: int) -> Path:
        return (
            self.output_dir
            / "checkpoints"
            / mode
            / f"clients_{num_clients}"
            / f"round_{round_idx:03d}"
            / f"client_{client_id:03d}.pt"
        )

    def save_client_checkpoint(
        self,
        *,
        round_idx: int,
        client: Any,
        metrics: Any,
        config: MNASConfig,
        batch_size: int,
    ) -> Path:
        path = self.client_checkpoint_path(
            config.experiment.mode,
            config.experiment.num_clients,
            round_idx,
            client.client_id,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = build_client_checkpoint(
            round_idx=round_idx,
            num_clients=config.experiment.num_clients,
            batch_size=batch_size,
            client=client,
            metrics=metrics,
            config=config,
        )
        torch.save(payload, path)
        return path

    def load_client_checkpoint(
        self,
        mode: str,
        num_clients: int,
        round_idx: int,
        client_id: int,
        map_location: str = "cpu",
    ) -> dict[str, Any]:
        return torch.load(self.client_checkpoint_path(mode, num_clients, round_idx, client_id), map_location=map_location)
