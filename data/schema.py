from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def infer_label_columns(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    for c in ["label", "target", "y", "labels"]:
        if c in cols:
            return [c]

    prefix_cols = [c for c in cols if c.lower().startswith(("label_", "target_"))]
    if len(prefix_cols) >= 2:
        return prefix_cols

    candidates = []
    skip = {"timestamp", "id", "can_id"}
    for c in cols:
        if c.lower() in skip:
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            uniq = set(pd.Series(s).dropna().unique().tolist())
            if len(uniq) <= 2 and uniq.issubset({0, 1}):
                candidates.append(c)
    if len(candidates) >= 2:
        return candidates
    raise ValueError("Cannot infer label columns robustly from dataframe schema.")


def build_feature_matrix(
    df: pd.DataFrame,
    label_cols: Sequence[str],
) -> tuple[np.ndarray, list[str], StandardScaler]:
    feature_cols = [c for c in df.columns if c not in set(label_cols)]
    numeric_feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_feature_cols:
        raise ValueError("No numeric feature columns available.")

    x = df[numeric_feature_cols].to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    x = scaler.fit_transform(x).astype(np.float32)
    return x, numeric_feature_cols, scaler


def build_multilabel_targets(
    df: pd.DataFrame,
    label_cols: Sequence[str],
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return multi-hot targets and primary labels used for Dirichlet partitioning."""

    if len(label_cols) >= 2:
        y = (df[list(label_cols)].fillna(0).to_numpy() > 0).astype(np.uint8)
        label_names = list(label_cols)
        row_sum = y.sum(axis=1)
        primary = y.argmax(axis=1).astype(np.int64)
        primary[row_sum == 0] = 0
        return y, label_names, primary

    col = label_cols[0]
    s = df[col]
    if s.dtype == object:
        tokens_per_row: list[list[str]] = []
        all_tokens: set[str] = set()
        for val in s.fillna("").astype(str).tolist():
            cleaned = val.strip().replace("[", "").replace("]", "").replace(";", ",").replace("|", ",")
            toks = [t.strip() for t in cleaned.split(",") if t.strip()]
            tokens_per_row.append(toks)
            all_tokens.update(toks)

        sorted_tokens = sorted(all_tokens)
        token_to_idx = {t: i for i, t in enumerate(sorted_tokens)}
        y = np.zeros((len(tokens_per_row), len(sorted_tokens)), dtype=np.uint8)
        primary = np.zeros(len(tokens_per_row), dtype=np.int64)
        for i, toks in enumerate(tokens_per_row):
            if not toks:
                continue
            for t in toks:
                y[i, token_to_idx[t]] = 1
            primary[i] = token_to_idx[toks[0]]
        return y, [str(t) for t in sorted_tokens], primary

    le = LabelEncoder()
    encoded = le.fit_transform(s.astype(str))
    n_cls = len(le.classes_)
    y = np.zeros((len(encoded), n_cls), dtype=np.uint8)
    y[np.arange(len(encoded)), encoded] = 1
    label_names = [str(x) for x in le.classes_.tolist()]
    return y, label_names, encoded.astype(np.int64)


def summarize_dataset(
    df: pd.DataFrame,
    y_multi: np.ndarray,
    label_names: Sequence[str],
    feature_cols: Sequence[str],
) -> dict[str, Any]:
    label_freq = y_multi.sum(axis=0).astype(np.int64)
    label_dist = {str(label_names[i]): int(label_freq[i]) for i in range(len(label_names))}
    return {
        "num_samples": int(len(df)),
        "num_features": int(len(feature_cols)),
        "feature_shape": (int(len(df)), int(len(feature_cols))),
        "num_labels": int(len(label_names)),
        "label_names": list(map(str, label_names)),
        "global_label_distribution": label_dist,
    }
