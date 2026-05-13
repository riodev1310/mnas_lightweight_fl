from __future__ import annotations


def build_operation_cost_table() -> dict[str, float]:
    """Hardware-aware surrogate op cost table."""

    return {
        "conv_7": 2.9,
        "conv_5": 2.2,
        "conv_3": 1.6,
        "conv_1": 1.0,
        "max_pool": 0.35,
        "avg_pool": 0.35,
        "skip": 0.08,
        "identity": 0.05,
    }
