from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def partition_stats(partition: list[np.ndarray], y_multi: np.ndarray, label_names: Sequence[str]) -> pd.DataFrame:
    rows = []
    n_labels = y_multi.shape[1]
    for cid, idx in enumerate(partition):
        yy = y_multi[idx]
        coverage = (yy.sum(axis=0) > 0).astype(np.int64)
        coverage_count = int(coverage.sum())
        rows.append(
            {
                "client_id": int(cid),
                "num_samples": int(len(idx)),
                "coverage_count": coverage_count,
                "coverage_ratio": float(coverage_count / max(n_labels, 1)),
                "missing_labels": ",".join([str(label_names[i]) for i in np.where(coverage == 0)[0].tolist()]),
            }
        )
    return pd.DataFrame(rows)


def score_partition(stat_df: pd.DataFrame, repaired_clients_count: int) -> float:
    if len(stat_df) == 0:
        return -1e9
    sizes = stat_df["num_samples"].to_numpy(dtype=np.float64)
    mean_sz = float(sizes.mean()) if len(sizes) else 1.0
    spread = float((sizes.max() - sizes.min()) / max(mean_sz, 1.0))
    balance_score = 1.0 - spread
    coverage_score = float(stat_df["coverage_ratio"].mean())
    penalty_repaired = 0.02 * float(repaired_clients_count)
    return 0.55 * coverage_score + 0.45 * balance_score - penalty_repaired


def get_label_coverage_for_client(indices: np.ndarray, y_multi: np.ndarray) -> list[int]:
    yy = y_multi[indices]
    return np.where(yy.sum(axis=0) > 0)[0].astype(int).tolist()
