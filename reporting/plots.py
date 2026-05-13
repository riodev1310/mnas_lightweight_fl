from __future__ import annotations

from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mnas_matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def plot_round_metrics(metrics_csv: str | Path, out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(metrics_csv)
    saved: dict[str, str] = {}
    for metric in ["loss", "accuracy", "macro_f1", "micro_f1", "weighted_f1"]:
        if metric not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.plot(df["round_idx"], df[metric], marker="o", linewidth=1.4)
        ax.set_xlabel("round")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by round")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = out / f"{metric}_curve.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved[metric] = str(path)
    return saved
