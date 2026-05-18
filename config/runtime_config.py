from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from .default_config import MNASConfig
from .paths import resolve_project_path


def _coerce_policy_keys(data: dict[str, Any]) -> dict[str, Any]:
    fed = data.get("federated", {})
    policy = fed.get("batch_size_policy")
    if isinstance(policy, dict):
        fed["batch_size_policy"] = {int(k): int(v) for k, v in policy.items()}
    return data


def _merge_dataclass(instance: Any, updates: dict[str, Any]) -> Any:
    for f in fields(instance):
        if f.name not in updates:
            continue
        current = getattr(instance, f.name)
        value = updates[f.name]
        if is_dataclass(current) and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(instance, f.name, value)
    return instance


def load_config(config_path: str | Path | None = None) -> MNASConfig:
    path = Path(config_path) if config_path else Path("configs") / "mnas_default.yaml"
    path = resolve_project_path(path)
    cfg = MNASConfig()
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        _merge_dataclass(cfg, _coerce_policy_keys(raw))
    cfg.validate()
    return cfg


def str_to_bool(value: str | bool | None) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


def build_config_from_args(args: argparse.Namespace) -> MNASConfig:
    cfg = load_config(args.config)
    if getattr(args, "num_clients", None) is not None:
        cfg.experiment.num_clients = int(args.num_clients)
    if getattr(args, "rounds", None) is not None:
        cfg.experiment.rounds = int(args.rounds)
    if getattr(args, "resume_from_round", None) is not None:
        cfg.experiment.resume_from_round = int(args.resume_from_round)
    if getattr(args, "seed", None) is not None:
        cfg.experiment.seed = int(args.seed)
    if getattr(args, "device", None) is not None:
        cfg.experiment.device = str(args.device)
    if getattr(args, "data_path", None) is not None:
        cfg.data.data_path = str(args.data_path)
    if getattr(args, "output_dir", None) is not None:
        cfg.outputs.output_dir = str(args.output_dir)
    if getattr(args, "max_samples", None) is not None:
        cfg.data.max_samples = int(args.max_samples)
    if getattr(args, "num_workers", None) is not None:
        cfg.data.num_workers = int(args.num_workers)
    if getattr(args, "save_every_round", None) is not None:
        cfg.federated.checkpoint_every_round = bool(args.save_every_round)
    if getattr(args, "eval_every_round", None) is not None:
        cfg.federated.eval_every_round = bool(args.eval_every_round)
    if getattr(args, "use_mnas_search", None) is not None:
        cfg.model.use_mnas_search = bool(args.use_mnas_search)
    cfg.validate()
    return cfg
