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

Resume from completed client checkpoints:

```bash
python -m runner.run_experiment \
  --num-clients 10 \
  --rounds 100 \
  --resume-from-round 55
```

Resume expects the same `--output-dir`, dataset, seed, partition config, and
number of clients as the crashed run. Metrics after the resume round are pruned
before new records are appended, so a partially written next round will not be
duplicated.

Run all scenarios:

```bash
python -m runner.run_all --rounds 100
```

Resume all scenarios from the same completed round:

```bash
python -m runner.run_all --rounds 100 --resume-from-round 55
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

## Data Split Protocol

MNAS now follows the benchmark protocol used for fair method comparison:

1. Load the raw dataset once.
2. Split a deterministic global train/test set first. Default: `test_ratio=0.2`.
3. Partition only the global train set into non-IID client train sets.
4. Train each client only on its `client_train` partition.
5. After every server proxy aggregation round, evaluate every personalized
   client model on the same global held-out test set.

The train/test split and client train partition are saved under:

```text
outputs/distribution/seed_<seed>/test_ratio_<ratio>/dirichlet_alpha_<alpha>/clients_<N>/
```

To force MNAS to reuse an existing distribution artifact instead of creating a
new one, pass the scenario directory:

```bash
python -m runner.run_experiment \
  --num-clients 10 \
  --rounds 100 \
  --distribution-dir outputs/distribution/seed_42/test_ratio_0.20/dirichlet_alpha_0.5/clients_10
```

The per-round metrics include `eval_scope=global_test` and `eval_samples`, so
it is clear the reported test metrics are no longer computed on the training
partition.

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
