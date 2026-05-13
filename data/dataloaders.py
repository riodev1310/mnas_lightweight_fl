from __future__ import annotations

import numpy as np
from torch.utils.data import DataLoader, Subset

from .dataset import FullTabularDataset


def build_client_dataloaders(
    dataset: FullTabularDataset,
    partition: list[np.ndarray],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    *,
    shuffle: bool = True,
) -> dict[int, DataLoader]:
    loaders: dict[int, DataLoader] = {}
    for cid, idx in enumerate(partition):
        loaders[cid] = DataLoader(
            Subset(dataset, idx.astype(int).tolist()),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    return loaders


def build_global_eval_dataloader(
    dataset: FullTabularDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    all_idx = np.arange(len(dataset), dtype=np.int64)
    return DataLoader(
        Subset(dataset, all_idx.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
