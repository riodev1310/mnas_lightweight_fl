# MNAS Project

This folder is intentionally runnable as the project root.

## Run From This Folder

```bash
cd mnas_project
python -m runner.run_experiment --num-clients 10 --rounds 100
```

Quick smoke test:

```bash
python -m runner.run_experiment \
  --num-clients 10 \
  --rounds 2 \
  --max-samples 100 \
  --num-workers 0 \
  --use-mnas-search false
```

Run all scenarios:

```bash
python -m runner.run_all --rounds 100
```

## Paper-Faithful Architecture Mode

The MNAS paper in this repository uses heterogeneous personalized architectures on clients.
The server aggregates only the unified proxy model weights and distributes
the aggregated proxy weights back to clients. Personalized architectures and
weights stay on each client.

Because of that, this project does not expose a `--mode homogeneous` switch.
After each aggregation round, evaluation is performed per client on the
personalized model, and each client checkpoint is saved.

That means a 100-round run with 10 clients writes 1000 client evaluation
records, not 100 server-only records.

## Tests

```bash
python -m compileall -q .
python -m pytest tests -q
```

If `pytest` is not installed yet, run `python -m pip install -r requirements.txt`
first.

## Layout

- `runner/`: command-line entrypoints
- `config/` and `configs/`: dataclass config and YAML config
- `data/`, `partition/`, `models/`, `search/`: dataset, client split, LCSMC, MNAS search
- `federated/`: client/server/FedAvg/orchestration
- `evaluation/`, `checkpointing/`, `reporting/`: metrics, checkpoints, plots and summaries

The default dataset path is `./datasets/road_multi_label.csv`, but path resolution also checks the parent project folder. In this repository the dataset is available at `../datasets/road_multi_label.csv` when you are inside `mnas_project/`.

Outputs are written to `./outputs/` by default when you run from this folder.
