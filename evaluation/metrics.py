from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import classification_report, multilabel_confusion_matrix, precision_recall_fscore_support


@dataclass
class EvaluationResult:
    loss: float
    accuracy: float
    label_accuracy: float
    precision: float
    recall: float
    f1: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    classification_report: dict[str, Any]
    confusion_matrix: np.ndarray
    per_label_metrics: list[dict[str, Any]]

    def scalar_dict(self) -> dict[str, float]:
        return {
            "loss": float(self.loss),
            "accuracy": float(self.accuracy),
            "label_accuracy": float(self.label_accuracy),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1": float(self.f1),
            "macro_precision": float(self.macro_precision),
            "macro_recall": float(self.macro_recall),
            "macro_f1": float(self.macro_f1),
            "micro_precision": float(self.micro_precision),
            "micro_recall": float(self.micro_recall),
            "micro_f1": float(self.micro_f1),
            "weighted_precision": float(self.weighted_precision),
            "weighted_recall": float(self.weighted_recall),
            "weighted_f1": float(self.weighted_f1),
        }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    loss: float,
    label_names: Sequence[str],
) -> EvaluationResult:
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    p_label, r_label, f_label, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    names = [str(x) for x in label_names]
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(y_true.shape[1])),
        target_names=names if len(names) == y_true.shape[1] else None,
        output_dict=True,
        zero_division=0,
    )
    per_label = []
    for i, name in enumerate(names):
        per_label.append(
            {
                "label": name,
                "precision": float(p_label[i]),
                "recall": float(r_label[i]),
                "f1": float(f_label[i]),
                "support": int(support[i]),
            }
        )

    exact_match = float(np.mean(np.all(y_true == y_pred, axis=1))) if len(y_true) else 0.0
    label_acc = float(np.mean(y_true == y_pred)) if y_true.size else 0.0
    cm = multilabel_confusion_matrix(y_true, y_pred)

    return EvaluationResult(
        loss=float(loss),
        accuracy=exact_match,
        label_accuracy=label_acc,
        precision=float(p_macro),
        recall=float(r_macro),
        f1=float(f_macro),
        macro_precision=float(p_macro),
        macro_recall=float(r_macro),
        macro_f1=float(f_macro),
        micro_precision=float(p_micro),
        micro_recall=float(r_micro),
        micro_f1=float(f_micro),
        weighted_precision=float(p_weighted),
        weighted_recall=float(r_weighted),
        weighted_f1=float(f_weighted),
        classification_report=report,
        confusion_matrix=cm,
        per_label_metrics=per_label,
    )
