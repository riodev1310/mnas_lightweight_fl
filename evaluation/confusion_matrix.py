from __future__ import annotations

from pathlib import Path
import os
import tempfile
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mnas_matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_confusion_matrix_artifacts(
    confusion_matrix: np.ndarray,
    label_names: Sequence[str],
    base_path: Path,
) -> dict[str, str]:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    npy_path = base_path.with_suffix(".npy")
    csv_path = base_path.with_suffix(".csv")
    png_path = base_path.with_suffix(".png")

    np.save(npy_path, confusion_matrix)
    rows = []
    for i, label in enumerate(label_names):
        mat = confusion_matrix[i]
        rows.append(
            {
                "label": str(label),
                "tn": int(mat[0, 0]),
                "fp": int(mat[0, 1]),
                "fn": int(mat[1, 0]),
                "tp": int(mat[1, 1]),
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    fig_width = max(7, min(16, 0.35 * len(label_names) + 5))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    heat = np.array([[r["tn"], r["fp"], r["fn"], r["tp"]] for r in rows], dtype=float)
    im = ax.imshow(heat, aspect="auto", cmap="Blues")
    ax.set_yticks(np.arange(len(label_names)))
    ax.set_yticklabels([str(x) for x in label_names], fontsize=8)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(["tn", "fp", "fn", "tp"])
    ax.set_title("Multi-label confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return {"npy": str(npy_path), "csv": str(csv_path), "png": str(png_path)}
