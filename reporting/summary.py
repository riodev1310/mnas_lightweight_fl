from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from utils.serialization import to_jsonable


def save_results_summary(summary: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir) / "summaries"
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "experiment_summary.json"
    csv_path = out / "experiment_summary.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, indent=2, ensure_ascii=False)
    pd.DataFrame([to_jsonable(summary)]).to_csv(csv_path, index=False)
    return {"summary_json": str(json_path), "summary_csv": str(csv_path)}
