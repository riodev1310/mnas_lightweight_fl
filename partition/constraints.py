from __future__ import annotations

import numpy as np


def compute_partition_integrity(partition: list[np.ndarray], n_total: int) -> dict[str, int | bool]:
    flat = np.concatenate(partition) if partition else np.array([], dtype=np.int64)
    uniq = np.unique(flat)
    return {
        "total_indices": int(len(flat)),
        "unique_indices": int(len(uniq)),
        "covers_all": bool(len(uniq) == n_total and len(flat) == n_total),
        "duplicate_count": int(len(flat) - len(uniq)),
    }


def enforce_client_constraints_indices(
    client_indices: list[np.ndarray],
    min_samples: int = 1,
    random_state: int = 42,
) -> tuple[list[np.ndarray], list[int]]:
    """Ensure each client has at least min_samples while preserving exact client count."""

    rng = np.random.default_rng(random_state)
    clients = [arr.astype(np.int64).tolist() for arr in client_indices]
    repaired_clients: list[int] = []

    if not clients:
        raise ValueError("No client partitions provided.")

    for cid, bucket in enumerate(clients):
        while len(bucket) < min_samples:
            sizes = np.array([len(c) for c in clients], dtype=np.int64)
            eligible = np.where(sizes > min_samples)[0]
            if len(eligible) == 0:
                raise ValueError("Not enough samples to satisfy min_client_samples for all clients.")
            donor = int(eligible[np.argmax(sizes[eligible])])
            draw_pos = int(rng.integers(0, len(clients[donor])))
            bucket.append(int(clients[donor].pop(draw_pos)))
            if cid not in repaired_clients:
                repaired_clients.append(cid)

    out = [np.array(c, dtype=np.int64) for c in clients]
    for arr in out:
        rng.shuffle(arr)
    return out, repaired_clients
