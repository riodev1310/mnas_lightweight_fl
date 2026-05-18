from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from checkpointing import CheckpointManager
from config.default_config import MNASConfig
from data import build_client_dataloaders, prepare_tabular_data
from evaluation.evaluator import sanity_check_shapes
from federated.aggregation import FedAvgAggregation
from federated.client import MNASClient
from federated.server import MNASServer
from federated.trainer import build_optimizer, fine_tune_personalized_model
from models.personalized_model import PersonalizedSearchedLCSMC
from models.proxy_model import ProxyLCSMC
from partition import build_partition_with_retry, get_label_coverage_for_client
from reporting import MetricsRecorder, plot_round_metrics, save_results_summary
from search import build_operation_cost_table, search_personalized_architecture
from utils import configure_logging, resolve_device, set_seed
from utils.serialization import to_jsonable


class MNASFederatedOrchestrator:
    def __init__(self, config: MNASConfig) -> None:
        self.config = config
        self.output_dir = Path(config.outputs.output_dir)
        self.batch_size = config.resolved_batch_size()
        self.device = resolve_device(config.experiment.device)
        self.logger = configure_logging(self.output_dir / "logs")
        self.clients: list[MNASClient] = []
        self.server: MNASServer | None = None
        self.metrics_recorder: MetricsRecorder | None = None
        self.checkpoint_manager = CheckpointManager(self.output_dir)
        self.data_bundle = None
        self.partition_pack: dict[str, Any] | None = None
        self.round_history: list[dict[str, Any]] = []

    def setup(self) -> None:
        cfg = self.config
        set_seed(cfg.experiment.seed)
        self.logger.info("Loading dataset from %s", cfg.data.data_path)
        self.data_bundle = prepare_tabular_data(cfg.data.data_path, max_samples=cfg.data.max_samples)
        self.logger.info("Dataset summary: %s", self.data_bundle.summary)

        input_dim = self.data_bundle.x.shape[1]
        num_labels = self.data_bundle.y.shape[1]
        sanity_check_shapes(self.data_bundle.x, self.data_bundle.y, num_labels=num_labels, feature_dim=input_dim)

        self.partition_pack = build_partition_with_retry(
            labels_for_partition=self.data_bundle.primary_labels,
            y_multi=self.data_bundle.y,
            label_names=self.data_bundle.label_names,
            num_clients=cfg.experiment.num_clients,
            partition_config=cfg.partition,
            seed=cfg.experiment.seed,
        )
        partition = self.partition_pack["partition"]
        self.logger.info("Partition integrity: %s", self.partition_pack["integrity"])
        resume_round = cfg.experiment.resume_from_round
        resume_checkpoints = self._load_resume_checkpoints() if resume_round is not None else None

        train_loaders = build_client_dataloaders(
            self.data_bundle.dataset,
            partition,
            batch_size=self.batch_size,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory and self.device.type == "cuda",
            shuffle=True,
        )
        eval_loaders = build_client_dataloaders(
            self.data_bundle.dataset,
            partition,
            batch_size=self.batch_size,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory and self.device.type == "cuda",
            shuffle=False,
        )
        self.metrics_recorder = MetricsRecorder(
            self.output_dir,
            mode=cfg.experiment.mode,
            num_clients=cfg.experiment.num_clients,
            label_names=self.data_bundle.label_names,
            overwrite=cfg.outputs.overwrite_metrics and resume_round is None,
        )
        if resume_round is not None:
            self.metrics_recorder.prune_after_round(resume_round)
        self._save_split_summary()

        op_cost = build_operation_cost_table() if resume_checkpoints is None else None
        self.clients = []
        for cpos, cid in enumerate(sorted(train_loaders.keys()), 1):
            self.logger.info("Preparing client %s/%s", cpos, len(train_loaders))
            checkpoint = resume_checkpoints.get(cid) if resume_checkpoints is not None else None
            if checkpoint is not None:
                selected_ops = self._selected_ops_from_checkpoint(checkpoint)
            else:
                assert op_cost is not None
                selected_ops = search_personalized_architecture(
                    train_loader=train_loaders[cid],
                    input_dim=input_dim,
                    num_labels=num_labels,
                    cfg=cfg,
                    op_cost=op_cost,
                    device=self.device,
                )
            personalized = PersonalizedSearchedLCSMC(
                input_dim=input_dim,
                num_labels=num_labels,
                channels=cfg.model.hidden_channels,
                reduction=cfg.model.attention_reduction,
                selected_ops=selected_ops,
            ).to(self.device)
            proxy = ProxyLCSMC(
                input_dim=input_dim,
                num_labels=num_labels,
                channels=cfg.model.hidden_channels,
                reduction=cfg.model.attention_reduction,
            ).to(self.device)
            client = MNASClient(
                client_id=cid,
                train_loader=train_loaders[cid],
                eval_loader=eval_loaders[cid],
                num_samples=int(len(partition[cid])),
                selected_ops=selected_ops,
                personalized_model=personalized,
                proxy_model=proxy,
                opt_personalized=build_optimizer(personalized, cfg),
                opt_proxy=build_optimizer(proxy, cfg),
                label_coverage=get_label_coverage_for_client(partition[cid], self.data_bundle.y),
                label_names=self.data_bundle.label_names,
                cfg=cfg,
                device=self.device,
            )
            if checkpoint is not None:
                self._restore_client_from_checkpoint(client, checkpoint)
            self.clients.append(client)

        self.server = MNASServer(
            input_dim=input_dim,
            num_labels=num_labels,
            cfg=cfg,
            device=self.device,
            aggregation_strategy=FedAvgAggregation(),
        )
        if resume_round is not None:
            self._restore_server_proxy_from_clients(resume_round)
        else:
            first_state = self.clients[0].proxy_model.state_dict()
            self.server.global_model.load_state_dict(first_state, strict=True)

    def run(self) -> dict[str, Any]:
        if self.server is None:
            self.setup()
        assert self.server is not None
        assert self.metrics_recorder is not None

        start = time.time()
        rng = np.random.default_rng(self.config.experiment.seed + self.config.experiment.num_clients)
        all_clients = sorted(self.clients, key=lambda c: c.client_id)

        start_round = (self.config.experiment.resume_from_round or 0) + 1
        for round_idx in range(start_round, self.config.experiment.rounds + 1):
            self.logger.info("Starting round %03d/%03d", round_idx, self.config.experiment.rounds)
            self.server.distribute_to_clients(all_clients)
            active_clients = self._select_active_clients(all_clients, rng)
            local_losses = []
            for client in active_clients:
                local_losses.append(client.train_one_round(round_idx))

            aggregation_result = self.server.aggregate(active_clients, round_idx)
            extra = {
                "active_clients": len(active_clients),
                "mean_local_loss_personal": float(np.mean([x["loss_personal"] for x in local_losses])) if local_losses else 0.0,
                "mean_local_loss_proxy": float(np.mean([x["loss_proxy"] for x in local_losses])) if local_losses else 0.0,
            }
            extra.update(aggregation_result)

            if self.config.federated.eval_every_round:
                client_metrics = []
                for client in all_clients:
                    metrics = client.evaluate_local(round_idx)
                    client_extra = {
                        **extra,
                        "num_samples": client.num_samples,
                        "label_coverage": ",".join(map(str, client.label_coverage)),
                    }
                    self.metrics_recorder.log_client_metrics(round_idx, client.client_id, metrics, extra=client_extra)
                    self.metrics_recorder.save_client_classification_report(round_idx, client.client_id, metrics)
                    self.metrics_recorder.save_client_confusion_matrix(round_idx, client.client_id, metrics)
                    if self._should_checkpoint(round_idx):
                        self.checkpoint_manager.save_client_checkpoint(
                            round_idx=round_idx,
                            client=client,
                            metrics=metrics,
                            config=self.config,
                            batch_size=self.batch_size,
                        )
                    client_metrics.append(metrics.scalar_dict())
                avg_row = self._average_client_metrics(round_idx, client_metrics, extra)
                self.round_history.append(avg_row)
                self.logger.info("Finished round %03d | avg_macro_f1=%.6f", round_idx, avg_row.get("macro_f1", 0.0))
            else:
                self.logger.info("Finished round %03d | evaluation skipped by config", round_idx)

        self.finalize()
        elapsed = time.time() - start
        summary = {
            "experiment": self.config.experiment.name,
            "mode": self.config.experiment.mode,
            "num_clients": self.config.experiment.num_clients,
            "rounds": self.config.experiment.rounds,
            "batch_size": self.batch_size,
            "device": str(self.device),
            "elapsed_sec": elapsed,
            "evaluation_records": self._count_evaluation_records(),
            "output_dir": str(self.output_dir),
            "resumed_from_round": self.config.experiment.resume_from_round,
        }
        summary.update(self.round_history[-1] if self.round_history else {})
        save_results_summary(summary, self.output_dir)
        return summary

    def finalize(self) -> None:
        if self.config.training.finetune_epochs > 0:
            for client in self.clients:
                fine_tune_personalized_model(client, cfg=self.config, device=self.device)
        if self.config.outputs.save_plots and self.metrics_recorder is not None:
            metrics_csv = self.metrics_recorder.client_csv
            if metrics_csv.exists():
                plot_round_metrics(metrics_csv, self.output_dir / "plots" / self.config.experiment.mode / f"clients_{self.config.experiment.num_clients}")

    def _select_active_clients(self, all_clients: list[MNASClient], rng: np.random.Generator) -> list[MNASClient]:
        frac = self.config.federated.client_fraction
        if frac >= 1.0:
            return all_clients
        m = max(1, int(len(all_clients) * frac))
        ids = set(rng.choice([c.client_id for c in all_clients], size=m, replace=False).tolist())
        return [c for c in all_clients if c.client_id in ids]

    def _should_checkpoint(self, round_idx: int) -> bool:
        return bool(self.config.outputs.save_checkpoints and self.config.federated.checkpoint_every_round)

    def _load_resume_checkpoints(self) -> dict[int, dict[str, Any]]:
        resume_round = self.config.experiment.resume_from_round
        assert resume_round is not None
        if self.config.federated.client_fraction < 1.0:
            raise ValueError(
                "Resume from client checkpoints is exact only when federated.client_fraction=1.0. "
                "Existing checkpoints do not record active client ids for partial-client rounds."
            )
        checkpoints = self.checkpoint_manager.load_client_round_checkpoints(
            self.config.experiment.mode,
            self.config.experiment.num_clients,
            resume_round,
            map_location="cpu",
        )
        self.logger.info(
            "Loaded %d client checkpoints from round %03d; metrics after this round will be pruned before append.",
            len(checkpoints),
            resume_round,
        )
        return checkpoints

    def _selected_ops_from_checkpoint(self, checkpoint: dict[str, Any]) -> list[list[str]]:
        selected_ops = checkpoint.get("selected_ops") or checkpoint.get("architecture")
        if not selected_ops:
            raise ValueError(f"Checkpoint for client {checkpoint.get('client_id')} does not contain selected_ops")
        return [[str(op) for op in ops] for ops in selected_ops]

    def _restore_client_from_checkpoint(self, client: MNASClient, checkpoint: dict[str, Any]) -> None:
        resume_round = self.config.experiment.resume_from_round
        if int(checkpoint.get("round_idx", -1)) != int(resume_round):
            raise ValueError(f"Client {client.client_id} checkpoint round mismatch: {checkpoint.get('round_idx')} != {resume_round}")
        if int(checkpoint.get("client_id", -1)) != int(client.client_id):
            raise ValueError(f"Client checkpoint id mismatch: {checkpoint.get('client_id')} != {client.client_id}")
        if int(checkpoint.get("num_clients", -1)) != int(self.config.experiment.num_clients):
            raise ValueError("Checkpoint num_clients does not match current config.")
        if int(checkpoint.get("batch_size", -1)) != int(self.batch_size):
            raise ValueError("Checkpoint batch_size does not match current config.")
        if int(checkpoint.get("num_samples", -1)) != int(client.num_samples):
            raise ValueError(
                f"Client {client.client_id} sample count mismatch. "
                "Use the same dataset, max_samples, partition config, seed, and num_clients as the crashed run."
            )

        client.personalized_model.load_state_dict(checkpoint["personalized_model_state_dict"], strict=True)
        client.proxy_model.load_state_dict(checkpoint["proxy_model_state_dict"], strict=True)
        client.opt_personalized.load_state_dict(checkpoint["optimizer_personalized_state_dict"])
        client.opt_proxy.load_state_dict(checkpoint["optimizer_proxy_state_dict"])
        self._move_optimizer_state_to_device(client.opt_personalized)
        self._move_optimizer_state_to_device(client.opt_proxy)

    def _move_optimizer_state_to_device(self, optimizer: torch.optim.Optimizer) -> None:
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)

    def _restore_server_proxy_from_clients(self, round_idx: int) -> None:
        assert self.server is not None
        proxy_states = [(copy.deepcopy(client.proxy_model.state_dict()), int(client.num_samples)) for client in self.clients]
        aggregated = self.server.aggregation_strategy.aggregate(proxy_states)
        self.server.global_model.load_state_dict(aggregated, strict=True)
        self.server.last_aggregation_result = {
            "round_idx": int(round_idx),
            "active_clients": len(self.clients),
            "aggregated_samples": int(sum(client.num_samples for client in self.clients)),
            "restored_from_client_checkpoints": True,
        }
        self.logger.info("Reconstructed server proxy state from round %03d client checkpoints.", round_idx)

    def _save_split_summary(self) -> None:
        assert self.partition_pack is not None
        mode = self.config.experiment.mode
        n = self.config.experiment.num_clients
        out = self.output_dir / "metrics" / mode / f"clients_{n}"
        out.mkdir(parents=True, exist_ok=True)
        self.partition_pack["stats"].to_csv(out / "client_split_summary.csv", index=False)
        pd.DataFrame([to_jsonable(self.partition_pack["integrity"])]).to_csv(out / "partition_integrity.csv", index=False)

    def _average_client_metrics(
        self,
        round_idx: int,
        rows: list[dict[str, float]],
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        avg = {"round_idx": int(round_idx)}
        if rows:
            for key in rows[0]:
                avg[key] = float(np.mean([r[key] for r in rows]))
        avg.update(extra)
        return avg

    def _count_evaluation_records(self) -> int:
        if self.metrics_recorder is not None and self.metrics_recorder.client_csv.exists():
            return int(len(pd.read_csv(self.metrics_recorder.client_csv)))
        return len(self.round_history) * self.config.experiment.num_clients


class MNASExperiment:
    def __init__(self, config: MNASConfig) -> None:
        self.config = config

    def run(self) -> dict[str, Any]:
        return MNASFederatedOrchestrator(self.config).run()
