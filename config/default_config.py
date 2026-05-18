from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .batch_size_policy import DEFAULT_BATCH_SIZE_POLICY, get_batch_size_by_num_clients


@dataclass
class ExperimentConfig:
    name: str = "mnas_lcsmc_federated"
    mode: str = "heterogeneous"
    num_clients: int = 10
    rounds: int = 100
    seed: int = 42
    device: str = "auto"
    resume_from_round: int | None = None


@dataclass
class DataConfig:
    data_path: str = "./datasets/road_multi_label.csv"
    use_all_data_for_training: bool = False
    test_ratio: float = 0.2
    distribution_dir: str | None = None
    num_workers: int = 4
    pin_memory: bool = True
    max_samples: int | None = None


@dataclass
class FederatedConfig:
    batch_size_policy: dict[int, int] = field(default_factory=lambda: dict(DEFAULT_BATCH_SIZE_POLICY))
    batch_size: int | None = None
    local_epochs_per_round: int = 1
    aggregation: str = "fedavg"
    eval_every_round: bool = True
    checkpoint_every_round: bool = True
    client_fraction: float = 1.0


@dataclass
class PartitionConfig:
    strategy: str = "dirichlet"
    dirichlet_alpha: float = 0.5
    min_client_samples: int = 1
    partition_retry_attempts: int = 20
    rebalance_tolerance_ratio: float = 0.05
    required_min_coverage_ratio: float = 1.0


@dataclass
class ModelConfig:
    classifier: str = "LCSMCNet"
    use_mnas_search: bool = True
    search_epochs: int = 1
    search_lr_w: float = 0.001
    search_lr_alpha: float = 0.001
    search_temperature: float = 1.0
    search_latency_lambda: float = 2.0
    latency_budget_t0: float = 1.0
    mnas_num_nodes: int = 5
    mnas_topk_ops: int = 2
    hidden_channels: int = 64
    attention_reduction: int = 4


@dataclass
class TrainingConfig:
    optimizer: str = "adam"
    lr: float = 0.001
    weight_decay: float = 0.0001
    finetune_epochs: int = 1
    distill_temperature: float = 2.0
    distill_weight: float = 1.0


@dataclass
class EvaluationConfig:
    threshold: float = 0.5
    average_methods: list[str] = field(default_factory=lambda: ["micro", "macro", "weighted"])
    save_confusion_matrix: bool = True
    save_classification_report: bool = True


@dataclass
class OutputsConfig:
    output_dir: str = "./outputs"
    save_checkpoints: bool = True
    save_metrics: bool = True
    save_plots: bool = True
    overwrite_metrics: bool = True


@dataclass
class MNASConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    partition: PartitionConfig = field(default_factory=PartitionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    outputs: OutputsConfig = field(default_factory=OutputsConfig)

    def resolved_batch_size(self) -> int:
        if self.federated.batch_size is not None:
            return int(self.federated.batch_size)
        return get_batch_size_by_num_clients(
            self.experiment.num_clients,
            self.federated.batch_size_policy,
        )

    def validate(self) -> None:
        mode = self.experiment.mode.lower()
        if mode != "heterogeneous":
            raise ValueError(
                "MNAS paper uses heterogeneous personalized architectures with a unified proxy model. "
                "Server-only homogeneous evaluation is disabled; use experiment.mode='heterogeneous'."
            )
        self.experiment.mode = mode
        if self.experiment.rounds < 1:
            raise ValueError("experiment.rounds must be >= 1")
        if not (0.0 < float(self.data.test_ratio) < 1.0):
            raise ValueError("data.test_ratio must be in (0, 1)")
        if self.experiment.resume_from_round is not None:
            if self.experiment.resume_from_round < 1:
                raise ValueError("experiment.resume_from_round must be >= 1")
            if self.experiment.resume_from_round >= self.experiment.rounds:
                raise ValueError("experiment.resume_from_round must be smaller than experiment.rounds")
        if self.federated.local_epochs_per_round < 1:
            raise ValueError("federated.local_epochs_per_round must be >= 1")
        if not (0 < self.federated.client_fraction <= 1):
            raise ValueError("federated.client_fraction must be in (0, 1]")
        self.resolved_batch_size()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def output_root(self) -> Path:
        return Path(self.outputs.output_dir)
