from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from evaluation.classification_report import save_classification_report
from evaluation.confusion_matrix import save_confusion_matrix_artifacts
from evaluation.metrics import EvaluationResult
from utils.serialization import to_jsonable


class MetricsRecorder:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        mode: str,
        num_clients: int,
        label_names: Sequence[str],
        overwrite: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.mode = mode
        self.num_clients = int(num_clients)
        self.label_names = [str(x) for x in label_names]
        self.root = self.output_dir / "metrics" / mode / f"clients_{num_clients}"
        self.report_dir = self.root / "classification_reports"
        self.cm_dir = self.root / "confusion_matrices"
        self.root.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.cm_dir.mkdir(parents=True, exist_ok=True)
        self.round_csv = self.root / "round_metrics.csv"
        self.round_jsonl = self.root / "round_metrics.jsonl"
        self.client_csv = self.root / "client_round_metrics.csv"
        self.client_jsonl = self.root / "client_round_metrics.jsonl"
        if overwrite:
            for path in [self.round_csv, self.round_jsonl, self.client_csv, self.client_jsonl]:
                if path.exists():
                    path.unlink()

    def prune_after_round(self, round_idx: int) -> None:
        """Keep only metrics/artifacts up to a completed resume round."""

        self._prune_csv(self.round_csv, round_idx)
        self._prune_csv(self.client_csv, round_idx)
        self._prune_jsonl(self.round_jsonl, round_idx)
        self._prune_jsonl(self.client_jsonl, round_idx)
        self._delete_round_artifacts_after(self.report_dir, round_idx)
        self._delete_round_artifacts_after(self.cm_dir, round_idx)

    def log_round_metrics(self, round_idx: int, metrics: EvaluationResult, extra: dict[str, Any] | None = None) -> None:
        row = {"round_idx": int(round_idx), "num_clients": self.num_clients}
        row.update(metrics.scalar_dict())
        if extra:
            row.update(to_jsonable(extra))
        self._append_row(self.round_csv, row)
        self._append_jsonl(self.round_jsonl, row)

    def log_client_metrics(
        self,
        round_idx: int,
        client_id: int,
        metrics: EvaluationResult,
        extra: dict[str, Any] | None = None,
    ) -> None:
        row = {"round_idx": int(round_idx), "client_id": int(client_id), "num_clients": self.num_clients}
        row.update(metrics.scalar_dict())
        if extra:
            row.update(to_jsonable(extra))
        self._append_row(self.client_csv, row)
        self._append_jsonl(self.client_jsonl, row)

    def save_classification_report(self, round_idx: int, metrics: EvaluationResult) -> str:
        return save_classification_report(metrics.classification_report, self.report_dir / f"round_{round_idx:03d}.json")

    def save_client_classification_report(self, round_idx: int, client_id: int, metrics: EvaluationResult) -> str:
        return save_classification_report(
            metrics.classification_report,
            self.report_dir / f"round_{round_idx:03d}_client_{client_id:03d}.json",
        )

    def save_confusion_matrix(self, round_idx: int, metrics: EvaluationResult) -> dict[str, str]:
        return save_confusion_matrix_artifacts(metrics.confusion_matrix, self.label_names, self.cm_dir / f"round_{round_idx:03d}")

    def save_client_confusion_matrix(self, round_idx: int, client_id: int, metrics: EvaluationResult) -> dict[str, str]:
        return save_confusion_matrix_artifacts(
            metrics.confusion_matrix,
            self.label_names,
            self.cm_dir / f"round_{round_idx:03d}_client_{client_id:03d}",
        )

    @staticmethod
    def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")

    @staticmethod
    def _append_row(path: Path, row: dict[str, Any]) -> None:
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _prune_csv(path: Path, round_idx: int) -> None:
        if not path.exists():
            return
        df = pd.read_csv(path)
        if "round_idx" not in df.columns:
            return
        df = df[df["round_idx"].astype(int) <= int(round_idx)]
        df.to_csv(path, index=False)

    @staticmethod
    def _prune_jsonl(path: Path, round_idx: int) -> None:
        if not path.exists():
            return
        kept = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if int(row.get("round_idx", 0)) <= int(round_idx):
                    kept.append(row)
        with path.open("w", encoding="utf-8") as f:
            for row in kept:
                f.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")

    @staticmethod
    def _delete_round_artifacts_after(directory: Path, round_idx: int) -> None:
        if not directory.exists():
            return
        pattern = re.compile(r"round_(\d{3})")
        for path in directory.glob("round_*"):
            match = pattern.search(path.name)
            if match and int(match.group(1)) > int(round_idx):
                path.unlink()
