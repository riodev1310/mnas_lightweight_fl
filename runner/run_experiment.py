from __future__ import annotations

import argparse
import json

from ._bootstrap import ensure_project_parent_on_path

ensure_project_parent_on_path()

from config.runtime_config import build_config_from_args, str_to_bool
from federated import MNASExperiment
from utils.serialization import to_jsonable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one MNAS federated learning scenario.")
    parser.add_argument("--config", default=None, help="Path to YAML config.")
    parser.add_argument("--num-clients", type=int, choices=[10, 20, 50], default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--resume-from-round", type=int, default=None, help="Resume from completed heterogeneous client checkpoints at this round.")
    parser.add_argument("--save-every-round", type=str_to_bool, default=None)
    parser.add_argument("--eval-every-round", type=str_to_bool, default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional smoke-test cap for the dataset.")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--use-mnas-search", type=str_to_bool, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = build_config_from_args(args)
    summary = MNASExperiment(cfg).run()
    print(json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
