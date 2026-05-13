from __future__ import annotations

import time
from typing import Any

from utils.serialization import to_jsonable


def build_server_checkpoint(
    *,
    round_idx: int,
    num_clients: int,
    batch_size: int,
    server: Any,
    metrics: Any,
    config: Any,
) -> dict[str, Any]:
    return {
        "round_idx": int(round_idx),
        "num_clients": int(num_clients),
        "batch_size": int(batch_size),
        "model_state_dict": server.global_model.state_dict(),
        "optimizer_state_dict": getattr(server, "optimizer_state_dict", None),
        "architecture_config": server.architecture_config(),
        "metrics": metrics.scalar_dict() if hasattr(metrics, "scalar_dict") else to_jsonable(metrics),
        "config_snapshot": to_jsonable(config),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "random_seed": int(config.experiment.seed),
    }
