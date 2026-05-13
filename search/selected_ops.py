from __future__ import annotations


def normalize_selected_ops(selected_ops: list[list[str]]) -> list[list[str]]:
    return [[str(op) for op in node_ops] for node_ops in selected_ops]
