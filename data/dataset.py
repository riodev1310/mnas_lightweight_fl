from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from config.paths import resolve_project_path

from .preprocessing import load_dataframe
from .schema import build_multilabel_targets, infer_label_columns, summarize_dataset


class FullTabularDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x.astype(np.float32)
        self.y = y.astype(np.uint8)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.x[idx]).float()
        y = torch.from_numpy(self.y[idx]).float()
        return x, y


@dataclass
class TabularDataBundle:
    dataframe: pd.DataFrame
    train_dataframe: pd.DataFrame
    test_dataframe: pd.DataFrame
    dataset: FullTabularDataset
    test_dataset: FullTabularDataset
    x: np.ndarray
    y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    primary_labels: np.ndarray
    test_primary_labels: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    label_names: list[str]
    feature_columns: list[str]
    label_columns: list[str]
    scaler: StandardScaler
    summary: dict[str, Any]


def _numeric_feature_columns(df: pd.DataFrame, label_cols: list[str]) -> list[str]:
    feature_cols = [c for c in df.columns if c not in set(label_cols)]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("No numeric feature columns available.")
    return numeric_cols


def _split_indices(primary_labels: np.ndarray, *, test_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(primary_labels), dtype=np.int64)
    counts = pd.Series(primary_labels).value_counts()
    n_classes = int(len(counts))
    n_test = int(np.ceil(len(indices) * float(test_ratio)))
    n_train = int(len(indices) - n_test)
    can_stratify = len(counts) > 0 and counts.min() >= 2 and n_test >= n_classes and n_train >= n_classes
    stratify = primary_labels if can_stratify else None
    train_idx, test_idx = train_test_split(
        indices,
        test_size=float(test_ratio),
        random_state=int(seed),
        shuffle=True,
        stratify=stratify,
    )
    return np.sort(train_idx.astype(np.int64)), np.sort(test_idx.astype(np.int64))


def _load_split_indices(split_dir: str | Path | None, n_samples: int) -> tuple[np.ndarray, np.ndarray] | None:
    if split_dir is None:
        return None
    root = resolve_project_path(split_dir)
    train_path = root / "global_train_indices.npy"
    test_path = root / "global_test_indices.npy"
    if not train_path.exists() or not test_path.exists():
        return None

    train_idx = np.load(train_path).astype(np.int64)
    test_idx = np.load(test_path).astype(np.int64)
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError(f"Distribution split at {root} has empty train/test indices.")
    if train_idx.min() < 0 or test_idx.min() < 0 or train_idx.max() >= n_samples or test_idx.max() >= n_samples:
        raise ValueError(f"Distribution split at {root} does not match the loaded dataset size {n_samples}.")
    train_set = set(train_idx.tolist())
    test_set = set(test_idx.tolist())
    if train_set & test_set:
        raise ValueError(f"Distribution split at {root} has train/test overlap.")
    if len(train_set) + len(test_set) != n_samples:
        raise ValueError(f"Distribution split at {root} does not cover the loaded dataset exactly once.")
    return np.sort(train_idx), np.sort(test_idx)


def prepare_tabular_data(
    data_path: str,
    max_samples: int | None = None,
    *,
    test_ratio: float = 0.2,
    seed: int = 42,
    distribution_dir: str | Path | None = None,
) -> TabularDataBundle:
    resolved_path = resolve_project_path(data_path)
    df = load_dataframe(resolved_path, max_samples=max_samples)
    label_columns = infer_label_columns(df)
    y, label_names, primary_labels = build_multilabel_targets(df, label_columns)
    loaded_split = _load_split_indices(distribution_dir, len(df))
    if loaded_split is None:
        train_indices, test_indices = _split_indices(primary_labels, test_ratio=test_ratio, seed=seed)
        split_source = "generated"
    else:
        train_indices, test_indices = loaded_split
        split_source = "loaded_distribution"

    feature_columns = _numeric_feature_columns(df, label_columns)
    raw_x = df[feature_columns].to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    train_x = scaler.fit_transform(raw_x[train_indices]).astype(np.float32)
    test_x = scaler.transform(raw_x[test_indices]).astype(np.float32)
    train_y = y[train_indices]
    test_y = y[test_indices]
    train_primary = primary_labels[train_indices]
    test_primary = primary_labels[test_indices]
    train_df = df.iloc[train_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)

    full_summary = summarize_dataset(df, y, label_names, feature_columns)
    train_summary = summarize_dataset(train_df, train_y, label_names, feature_columns)
    test_summary = summarize_dataset(test_df, test_y, label_names, feature_columns)
    summary = {
        **full_summary,
        "split": {
            "train_samples": int(len(train_indices)),
            "test_samples": int(len(test_indices)),
            "test_ratio": float(test_ratio),
            "seed": int(seed),
            "source": split_source,
            "train_label_distribution": train_summary["global_label_distribution"],
            "test_label_distribution": test_summary["global_label_distribution"],
        },
    }
    return TabularDataBundle(
        dataframe=df,
        train_dataframe=train_df,
        test_dataframe=test_df,
        dataset=FullTabularDataset(train_x, train_y),
        test_dataset=FullTabularDataset(test_x, test_y),
        x=train_x,
        y=train_y,
        test_x=test_x,
        test_y=test_y,
        primary_labels=train_primary,
        test_primary_labels=test_primary,
        train_indices=train_indices,
        test_indices=test_indices,
        label_names=label_names,
        feature_columns=feature_columns,
        label_columns=label_columns,
        scaler=scaler,
        summary=summary,
    )
