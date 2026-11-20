r"""Random-input baseline: train on noise unrelated to the targets.

The real masks, targets and ids are kept; only the feature tensors are replaced
by ``N(0, 1)`` noise of the stored window shape. Whatever the network scores
here is what it achieves without any information in its input.

Training noise is resampled on every access, so it cannot be memorised.
Validation and test noise is seeded by the clip's row, so evaluation is
deterministic.

Example:
    Train the baseline with the L1 loss::

        make train-fi-random-l1
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch

from personalitylinmult.data.datamodule import FiDataModule
from personalitylinmult.data.dataset import FiHdf5Dataset
from personalitylinmult.train.cli import run
from personalitylinmult.train.config import ExperimentConfig, parse_override

logger = logging.getLogger(__name__)
"""Module-level logger."""


class NoiseHdf5Dataset(FiHdf5Dataset):
    """:class:`FiHdf5Dataset` whose features are replaced by Gaussian noise.

    Accepts the same arguments as :class:`FiHdf5Dataset`. Masks, targets and
    video ids are the real ones.
    """

    def _noise_like(self, index: int, reference: torch.Tensor) -> torch.Tensor:
        """Draw noise shaped like a real feature tensor.

        Args:
            index (int): Row within this split.
            reference (torch.Tensor): The real tensor whose shape to copy.

        Returns:
            torch.Tensor: Standard normal noise. Fresh on every call for the
            train split; a pure function of the clip's row otherwise.
        """
        if self.subset == "train":
            return torch.randn_like(reference)
        generator = torch.Generator().manual_seed(self.seed + int(self.row_indices[index]))
        return torch.randn(reference.shape, generator=generator)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Read one clip with its features replaced by noise.

        Args:
            index (int): Row within this split.

        Returns:
            dict: As :meth:`FiHdf5Dataset.__getitem__`, with noise features.
        """
        item = super().__getitem__(index)
        for name in self.feature_names:
            item[name] = self._noise_like(index, item[name])
        return item


class NoiseDataModule(FiDataModule):
    """:class:`FiDataModule` serving :class:`NoiseHdf5Dataset` splits."""

    def _build(self, subset: str) -> NoiseHdf5Dataset:
        """Construct one split's noise dataset.

        Args:
            subset (str): Split name.

        Returns:
            NoiseHdf5Dataset: The dataset.
        """
        return NoiseHdf5Dataset(
            h5_path=self.config.h5_path,
            subset=subset,
            feature_names=self.config.features,
            target_names=self.target_names,
            window_seconds=self.config.window_seconds,
            crop_mode=self.config.crop_mode,
            seed=self.config.seed,
        )


def main() -> None:
    """CLI entry point: the standard training CLI on noise inputs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="Data config YAML.")
    parser.add_argument("--model", type=Path, required=True, help="Model config YAML.")
    parser.add_argument("--train", type=Path, required=True, help="Train config YAML.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, e.g. --set train.loss=l1.",
    )
    parser.add_argument("--fast-dev-run", action="store_true", help="Run one batch and exit.")
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    overrides = dict(parse_override(text) for text in args.overrides)
    config = ExperimentConfig.from_files(args.data, args.model, args.train, overrides)
    datamodule = NoiseDataModule(config.data, target_names=config.train.targets)
    run(config, fast_dev_run=args.fast_dev_run, datamodule=datamodule)


if __name__ == "__main__":
    main()
