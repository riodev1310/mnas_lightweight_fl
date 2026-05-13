from __future__ import annotations

import numpy as np


def rebalance_partition_indices(
    partition: list[np.ndarray],
    labels: np.ndarray,
    tolerance_ratio: float,
    random_state: int,
) -> list[np.ndarray]:
    """Repair step to make client sizes approximately equal without data loss."""

    rng = np.random.default_rng(random_state)
    labels = np.asarray(labels)
    clients = [arr.astype(np.int64).tolist() for arr in partition]
    if not clients:
        return []

    baseline = np.concatenate(partition) if partition else np.array([], dtype=np.int64)
    baseline_set = set(baseline.tolist())
    target = int(round(sum(len(c) for c in clients) / len(clients)))
    tolerance = max(1, int(target * tolerance_ratio))
    label_set_global = set(np.unique(labels).tolist())

    def label_set(lst: list[int]) -> set[int]:
        if not lst:
            return set()
        return set(labels[np.array(lst, dtype=np.int64)].tolist())

    for _ in range(5000):
        sizes = np.array([len(c) for c in clients], dtype=np.int64)
        donor = int(np.argmax(sizes))
        recv = int(np.argmin(sizes))
        if sizes[donor] - sizes[recv] <= tolerance:
            break

        need = max(1, (sizes[donor] - sizes[recv]) // 2)
        missing = label_set_global - label_set(clients[recv])
        donor_pool = clients[donor]
        priority = [ix for ix in donor_pool if labels[ix] in missing]
        candidate = priority if priority else donor_pool
        move_n = min(need, len(candidate))
        selected = rng.choice(candidate, size=move_n, replace=False)
        selected_set = set(int(x) for x in selected.tolist())

        clients[donor] = [ix for ix in donor_pool if ix not in selected_set]
        clients[recv].extend(list(selected_set))

    repaired = [np.array(c, dtype=np.int64) for c in clients]
    merged = np.concatenate(repaired) if repaired else np.array([], dtype=np.int64)
    if len(merged) != len(baseline) or set(merged.tolist()) != baseline_set:
        raise RuntimeError("Partition repair caused data loss/duplication.")

    for arr in repaired:
        rng.shuffle(arr)
    return repaired
