from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from config.default_config import PartitionConfig

from .constraints import compute_partition_integrity, enforce_client_constraints_indices
from .rebalance import rebalance_partition_indices
from .stats import partition_stats, score_partition


def dirichlet_partition_indices(
    labels: np.ndarray,
    num_clients: int,
    alpha: float = 0.5,
    random_state: int = 42,
) -> list[np.ndarray]:
    rng = np.random.default_rng(random_state)
    unique_labels = np.unique(labels)
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for label in unique_labels:
        idx = np.where(labels == label)[0]
        rng.shuffle(idx)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        splits = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        label_splits = np.split(idx, splits)
        for i in range(num_clients):
            client_indices[i].extend(label_splits[i].astype(int).tolist())

    out = [np.array(c, dtype=np.int64) for c in client_indices]
    for arr in out:
        rng.shuffle(arr)
    return out


def build_partition_with_retry(
    labels_for_partition: np.ndarray,
    y_multi: np.ndarray,
    label_names: Sequence[str],
    num_clients: int,
    partition_config: PartitionConfig,
    seed: int,
) -> dict[str, Any]:
    best_pack: dict[str, Any] | None = None
    best_score = -1e18
    n_total = len(labels_for_partition)

    for attempt in range(partition_config.partition_retry_attempts):
        attempt_seed = int(seed + num_clients * 1000 + attempt)
        raw = dirichlet_partition_indices(
            labels=labels_for_partition,
            num_clients=num_clients,
            alpha=partition_config.dirichlet_alpha,
            random_state=attempt_seed,
        )
        valid, repaired_clients = enforce_client_constraints_indices(
            raw,
            min_samples=partition_config.min_client_samples,
            random_state=attempt_seed,
        )
        repaired = rebalance_partition_indices(
            valid,
            labels=labels_for_partition,
            tolerance_ratio=partition_config.rebalance_tolerance_ratio,
            random_state=attempt_seed,
        )

        integrity = compute_partition_integrity(repaired, n_total)
        if not integrity["covers_all"] or integrity["duplicate_count"] != 0 or len(repaired) != num_clients:
            continue

        stat_df = partition_stats(repaired, y_multi=y_multi, label_names=label_names)
        score = score_partition(stat_df, repaired_clients_count=len(repaired_clients))
        if score > best_score:
            best_score = score
            best_pack = {
                "partition": repaired,
                "repaired_clients": repaired_clients,
                "removed_clients": [],
                "stats": stat_df,
                "integrity": integrity,
                "score": score,
                "seed": attempt_seed,
                "attempt": attempt,
            }

            mean_cov = float(stat_df["coverage_ratio"].mean())
            sizes = stat_df["num_samples"].to_numpy()
            spread = float((sizes.max() - sizes.min()) / max(sizes.mean(), 1.0))
            if mean_cov >= partition_config.required_min_coverage_ratio and spread <= partition_config.rebalance_tolerance_ratio:
                break

    if best_pack is None:
        raise RuntimeError(f"Cannot produce valid partition for {num_clients} clients")
    return best_pack
