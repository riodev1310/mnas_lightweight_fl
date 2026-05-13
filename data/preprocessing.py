from __future__ import annotations

from pathlib import Path

import pandas as pd


ROAD_DTYPE_MAP = {
    "timestamp": "float32",
    "ID": "int32",
    "DATA0": "int16",
    "DATA1": "int16",
    "DATA2": "int16",
    "DATA3": "int16",
    "DATA4": "int16",
    "DATA5": "int16",
    "DATA6": "int16",
    "DATA7": "int16",
    "label": "int16",
}


def load_dataframe(data_path: str | Path, max_samples: int | None = None) -> pd.DataFrame:
    path = Path(data_path)
    header = pd.read_csv(path, nrows=0)
    dtype = {k: v for k, v in ROAD_DTYPE_MAP.items() if k in header.columns}
    df = pd.read_csv(path, dtype=dtype)
    df = df.dropna().reset_index(drop=True)
    if max_samples is not None:
        df = df.head(int(max_samples)).reset_index(drop=True)
    return df
