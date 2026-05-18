from pathlib import Path

import numpy as np
import pandas as pd

from data import prepare_tabular_data


def test_prepare_tabular_data_splits_before_training_and_keeps_test_held_out(tmp_path: Path):
    df = pd.DataFrame(
        {
            "timestamp": np.arange(100, dtype=np.float32),
            "ID": np.arange(100, dtype=np.int32),
            "DATA0": np.arange(100, dtype=np.int16),
            "DATA1": np.arange(100, dtype=np.int16) % 7,
            "label": [0, 1] * 50,
        }
    )
    path = tmp_path / "toy.csv"
    df.to_csv(path, index=False)

    bundle = prepare_tabular_data(str(path), test_ratio=0.2, seed=42)

    assert len(bundle.dataset) == 80
    assert len(bundle.test_dataset) == 20
    assert set(bundle.train_indices).isdisjoint(set(bundle.test_indices))
    assert sorted(np.concatenate([bundle.train_indices, bundle.test_indices]).tolist()) == list(range(100))
    assert bundle.primary_labels.shape[0] == 80
    assert bundle.test_primary_labels.shape[0] == 20
    np.testing.assert_allclose(bundle.x.mean(axis=0), np.zeros(bundle.x.shape[1]), atol=1e-6)


def test_prepare_tabular_data_can_reuse_saved_global_split(tmp_path: Path):
    df = pd.DataFrame(
        {
            "timestamp": np.arange(10, dtype=np.float32),
            "ID": np.arange(10, dtype=np.int32),
            "DATA0": np.arange(10, dtype=np.int16),
            "label": [0, 1] * 5,
        }
    )
    path = tmp_path / "toy.csv"
    df.to_csv(path, index=False)
    distribution_dir = tmp_path / "distribution"
    distribution_dir.mkdir()
    np.save(distribution_dir / "global_train_indices.npy", np.array([0, 1, 2, 3, 4, 5, 6, 7]))
    np.save(distribution_dir / "global_test_indices.npy", np.array([8, 9]))

    bundle = prepare_tabular_data(str(path), test_ratio=0.5, seed=7, distribution_dir=distribution_dir)

    assert bundle.train_indices.tolist() == list(range(8))
    assert bundle.test_indices.tolist() == [8, 9]
    assert bundle.summary["split"]["source"] == "loaded_distribution"
