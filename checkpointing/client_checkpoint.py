from __future__ import annotations

import time
from typing import Any

from utils.serialization import to_jsonable


def build_client_checkpoint(
    *,
    round_idx: int,
    num_clients: int,
    batch_size: int,
    client: Any,
    metrics: Any,
    config: Any,
) -> dict[str, Any]:
    return {
        "round_idx": int(round_idx),
        "client_id": int(client.client_id),
        "num_clients": int(num_clients),
        "batch_size": int(batch_size),
        "personalized_model_state_dict": client.personalized_model.state_dict(),
        "proxy_model_state_dict": client.proxy_model.state_dict(),
        "optimizer_personalized_state_dict": client.opt_personalized.state_dict(),
        "optimizer_proxy_state_dict": client.opt_proxy.state_dict(),
        "architecture": to_jsonable(client.selected_ops),
        "selected_ops": to_jsonable(client.selected_ops),
        "local_metrics": metrics.scalar_dict() if hasattr(metrics, "scalar_dict") else to_jsonable(metrics),
        "client_label_coverage": to_jsonable(client.label_coverage),
        "num_samples": int(client.num_samples),
        "config_snapshot": to_jsonable(config),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "random_seed": int(config.experiment.seed),
    }
