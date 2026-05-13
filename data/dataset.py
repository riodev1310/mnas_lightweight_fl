from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from config.paths import resolve_project_path

from .preprocessing import load_dataframe
from .schema import build_feature_matrix, build_multilabel_targets, infer_label_columns, summarize_dataset


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
    dataset: FullTabularDataset
    x: np.ndarray
    y: np.ndarray
    primary_labels: np.ndarray
    label_names: list[str]
    feature_columns: list[str]
    label_columns: list[str]
    scaler: StandardScaler
    summary: dict[str, Any]


def prepare_tabular_data(data_path: str, max_samples: int | None = None) -> TabularDataBundle:
    resolved_path = resolve_project_path(data_path)
    df = load_dataframe(resolved_path, max_samples=max_samples)
    label_columns = infer_label_columns(df)
    x, feature_columns, scaler = build_feature_matrix(df, label_columns)
    y, label_names, primary_labels = build_multilabel_targets(df, label_columns)
    summary = summarize_dataset(df, y, label_names, feature_columns)
    return TabularDataBundle(
        dataframe=df,
        dataset=FullTabularDataset(x, y),
        x=x,
        y=y,
        primary_labels=primary_labels,
        label_names=label_names,
        feature_columns=feature_columns,
        label_columns=label_columns,
        scaler=scaler,
        summary=summary,
    )
