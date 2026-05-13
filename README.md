# MNAS Project

This folder is intentionally runnable as the project root.

## Run From This Folder

```bash
cd mnas_project
python -m runner.run_experiment --num-clients 10 --mode homogeneous --rounds 100
```

Quick smoke test:

```bash
python -m runner.run_experiment \
  --num-clients 10 \
  --mode homogeneous \
  --rounds 2 \
  --max-samples 100 \
  --num-workers 0 \
  --use-mnas-search false
```

Run all scenarios:

```bash
python -m runner.run_all --mode homogeneous --rounds 100
python -m runner.run_all --mode heterogeneous --rounds 100
```

## Layout

- `runner/`: command-line entrypoints
- `config/` and `configs/`: dataclass config and YAML config
- `data/`, `partition/`, `models/`, `search/`: dataset, client split, LCSMC, MNAS search
- `federated/`: client/server/FedAvg/orchestration
- `evaluation/`, `checkpointing/`, `reporting/`: metrics, checkpoints, plots and summaries

The default dataset path is `./datasets/road_multi_label.csv`, but path resolution also checks the parent project folder. In this repository the dataset is available at `../datasets/road_multi_label.csv` when you are inside `mnas_project/`.

Outputs are written to `./outputs/` by default when you run from this folder.
