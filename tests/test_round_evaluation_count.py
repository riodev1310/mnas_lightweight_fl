from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.metrics import EvaluationResult
from reporting import MetricsRecorder


def _metrics() -> EvaluationResult:
    return EvaluationResult(
        loss=0.1,
        accuracy=1.0,
        label_accuracy=1.0,
        precision=1.0,
        recall=1.0,
        f1=1.0,
        macro_precision=1.0,
        macro_recall=1.0,
        macro_f1=1.0,
        micro_precision=1.0,
        micro_recall=1.0,
        micro_f1=1.0,
        weighted_precision=1.0,
        weighted_recall=1.0,
        weighted_f1=1.0,
        classification_report={},
        confusion_matrix=np.array([[[1, 0], [0, 1]]]),
        per_label_metrics=[],
    )


def test_10_clients_100_rounds_have_1000_client_evaluation_records(tmp_path: Path):
    recorder = MetricsRecorder(tmp_path, mode="heterogeneous", num_clients=10, label_names=["a"])
    for rnd in range(1, 101):
        for cid in range(10):
            recorder.log_client_metrics(rnd, cid, _metrics())

    df = pd.read_csv(tmp_path / "metrics" / "heterogeneous" / "clients_10" / "client_round_metrics.csv")
    assert len(df) == 1000
    assert df.groupby("round_idx")["client_id"].nunique().eq(10).all()
