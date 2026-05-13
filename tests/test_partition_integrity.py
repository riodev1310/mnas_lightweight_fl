import numpy as np

from config.default_config import PartitionConfig
from partition import build_partition_with_retry


def test_partition_integrity_keeps_every_sample_once():
    labels = np.array([0, 1] * 50)
    y_multi = labels.reshape(-1, 1).astype(np.float32)
    cfg = PartitionConfig(
        dirichlet_alpha=0.5,
        min_client_samples=1,
        partition_retry_attempts=5,
        rebalance_tolerance_ratio=0.5,
        required_min_coverage_ratio=0.0,
    )

    pack = build_partition_with_retry(
        labels_for_partition=labels,
        y_multi=y_multi,
        label_names=["target"],
        num_clients=10,
        partition_config=cfg,
        seed=42,
    )
    partition = pack["partition"]
    flat = np.concatenate(partition)

    assert pack["integrity"]["covers_all"] is True
    assert pack["integrity"]["duplicate_count"] == 0
    assert len(flat) == len(labels)
    assert len(np.unique(flat)) == len(labels)
    assert all(len(indices) > 0 for indices in partition)
