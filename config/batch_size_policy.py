from __future__ import annotations

from typing import Mapping


DEFAULT_BATCH_SIZE_POLICY: dict[int, int] = {
    10: 2048,
    20: 1024,
    50: 512,
}


def get_batch_size_by_num_clients(
    num_clients: int,
    policy: Mapping[int | str, int] | None = None,
    *,
    strict: bool = True,
) -> int:
    """Return the configured batch size for a supported FL client scenario."""

    lookup = DEFAULT_BATCH_SIZE_POLICY if policy is None else {int(k): int(v) for k, v in policy.items()}
    if int(num_clients) in lookup:
        return int(lookup[int(num_clients)])
    if strict:
        supported = ", ".join(str(k) for k in sorted(lookup))
        raise ValueError(f"Unsupported num_clients={num_clients}. Supported values: {supported}")
    return int(min(lookup.values()))
