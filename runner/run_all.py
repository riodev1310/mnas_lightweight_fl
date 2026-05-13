from __future__ import annotations

import argparse
import copy
import json

from ._bootstrap import ensure_project_parent_on_path

ensure_project_parent_on_path()

from config.runtime_config import load_config, str_to_bool
from federated import MNASExperiment
from utils.serialization import to_jsonable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MNAS scenarios for 10, 20, and 50 clients.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--use-mnas-search", type=str_to_bool, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base = load_config(args.config)
    summaries = {}
    for n in [10, 20, 50]:
        cfg = copy.deepcopy(base)
        cfg.experiment.num_clients = n
        if args.rounds is not None:
            cfg.experiment.rounds = args.rounds
        if args.data_path is not None:
            cfg.data.data_path = args.data_path
        if args.output_dir is not None:
            cfg.outputs.output_dir = args.output_dir
        if args.device is not None:
            cfg.experiment.device = args.device
        if args.max_samples is not None:
            cfg.data.max_samples = args.max_samples
        if args.num_workers is not None:
            cfg.data.num_workers = args.num_workers
        if args.use_mnas_search is not None:
            cfg.model.use_mnas_search = args.use_mnas_search
        cfg.validate()
        summaries[n] = MNASExperiment(cfg).run()
    print(json.dumps(to_jsonable(summaries), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
