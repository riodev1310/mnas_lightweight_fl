from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.metrics import EvaluationResult
from reporting import MetricsRecorder


def _fake_metrics() -> EvaluationResult:
    return EvaluationResult(
        loss=0.5,
        accuracy=0.25,
        label_accuracy=0.75,
        precision=0.7,
        recall=0.6,
        f1=0.65,
        macro_precision=0.7,
        macro_recall=0.6,
        macro_f1=0.65,
        micro_precision=0.8,
        micro_recall=0.7,
        micro_f1=0.74,
        weighted_precision=0.76,
        weighted_recall=0.71,
        weighted_f1=0.73,
        classification_report={"label_0": {"precision": 1.0}},
        confusion_matrix=np.array([[[8, 1], [2, 9]], [[7, 3], [1, 9]]]),
        per_label_metrics=[],
    )


def test_client_metrics_recorder_writes_csv_jsonl_reports_and_confusion_matrix(tmp_path: Path):
    recorder = MetricsRecorder(tmp_path, mode="heterogeneous", num_clients=10, label_names=["a", "b"])
    metrics = _fake_metrics()

    recorder.log_client_metrics(1, 0, metrics)
    recorder.save_client_classification_report(1, 0, metrics)
    paths = recorder.save_client_confusion_matrix(1, 0, metrics)

    root = tmp_path / "metrics" / "heterogeneous" / "clients_10"
    assert (root / "client_round_metrics.csv").exists()
    assert (root / "client_round_metrics.jsonl").exists()
    assert (root / "classification_reports" / "round_001_client_000.json").exists()
    assert Path(paths["npy"]).exists()
    assert Path(paths["csv"]).exists()
    assert Path(paths["png"]).exists()
    assert len(pd.read_csv(root / "client_round_metrics.csv")) == 1
