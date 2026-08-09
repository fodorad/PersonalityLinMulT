"""Lightning ``DataModule`` over the built FI HDF5 file."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import lightning as L
import torch
from torch.utils.data import DataLoader

from personalitylinmult.data.collate import collate_fi
from personalitylinmult.data.dataset import FiHdf5Dataset

if TYPE_CHECKING:
    from personalitylinmult.train.config import DataConfig

logger = logging.getLogger(__name__)
"""Module-level logger."""


def seed_worker(worker_id: int) -> None:  # noqa: ARG001 -- worker_init_fn's fixed signature
    """Seed a dataloader worker's RNGs from torch's per-worker seed.

    Without this, every worker inherits the parent's numpy seed and augmentation
    draws repeat across workers.

    Args:
        worker_id (int): Worker index, unused but required by the hook.
    """
    import random

    import numpy as np

    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


class FiDataModule(L.LightningDataModule):
    """Serves the FI splits from the single HDF5 artifact.

    Args:
        config (DataConfig): Dataloader configuration.
        target_names (list[str]): Label columns for the active task, in head
            order.
    """

    def __init__(self, config: DataConfig, target_names: list[str]):
        super().__init__()
        self.config = config
        self.target_names = list(target_names)
        self.train_dataset: FiHdf5Dataset | None = None
        self.valid_dataset: FiHdf5Dataset | None = None
        self.test_dataset: FiHdf5Dataset | None = None

    def _build(self, subset: str) -> FiHdf5Dataset:
        """Construct one split's dataset.

        Args:
            subset (str): Split name.

        Returns:
            FiHdf5Dataset: The dataset.
        """
        return FiHdf5Dataset(
            h5_path=self.config.h5_path,
            subset=subset,
            feature_names=self.config.features,
            target_names=self.target_names,
            window_seconds=self.config.window_seconds,
            crop_mode=self.config.crop_mode,
            seed=self.config.seed,
        )

    def setup(self, stage: str | None = None) -> None:
        """Construct the datasets needed for a stage.

        Args:
            stage (str | None): ``"fit"``, ``"validate"``, ``"test"``, or
                ``None`` for all.
        """
        if stage in (None, "fit"):
            self.train_dataset = self._build("train")
            self.valid_dataset = self._build("valid")
        if stage in (None, "validate") and self.valid_dataset is None:
            self.valid_dataset = self._build("valid")
        if stage in (None, "test"):
            self.test_dataset = self._build("test")

    def _loader(self, dataset: FiHdf5Dataset, shuffle: bool) -> DataLoader:
        """Wrap a dataset in a configured ``DataLoader``.

        Args:
            dataset (FiHdf5Dataset): The dataset.
            shuffle (bool): Whether to shuffle (training only).

        Returns:
            DataLoader: The loader.
        """
        generator = torch.Generator()
        generator.manual_seed(self.config.seed)

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            # Pinned (page-locked) host memory only speeds up CUDA transfers;
            # on MPS/CPU it is a silent no-op that PyTorch warns about on every
            # run. The config's pin_memory is the user's *intent* -- it is only
            # ever actually honoured when CUDA is the device that would benefit.
            pin_memory=self.config.pin_memory and torch.cuda.is_available(),
            # Workers reopen their HDF5 handle on start, so tearing them down
            # every epoch would pay that cost repeatedly.
            persistent_workers=self.config.persistent_workers and self.config.num_workers > 0,
            prefetch_factor=self.config.prefetch_factor if self.config.num_workers > 0 else None,
            drop_last=shuffle,
            collate_fn=collate_fi,
            worker_init_fn=seed_worker,
            generator=generator,
        )

    def train_dataloader(self) -> DataLoader:
        """Training loader.

        Returns:
            DataLoader: Shuffled, drop-last loader over the train split.

        Raises:
            RuntimeError: If called before ``setup``.
        """
        if self.train_dataset is None:
            raise RuntimeError("setup('fit') must run before train_dataloader().")
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Validation loader.

        Returns:
            DataLoader: Deterministic loader over the valid split.

        Raises:
            RuntimeError: If called before ``setup``.
        """
        if self.valid_dataset is None:
            raise RuntimeError("setup('fit') must run before val_dataloader().")
        return self._loader(self.valid_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Test loader.

        Returns:
            DataLoader: Deterministic loader over the test split.

        Raises:
            RuntimeError: If called before ``setup``.
        """
        if self.test_dataset is None:
            raise RuntimeError("setup('test') must run before test_dataloader().")
        return self._loader(self.test_dataset, shuffle=False)
