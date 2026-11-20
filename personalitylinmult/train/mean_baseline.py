"""Train-mean predictor baseline: what "learn nothing" scores on test.

Predicts, for every test clip, the per-trait mean of the train labels, then
scores it with the same metric functions as a trained run so the numbers sit in
the same table. Pearson r, CCC and std_ratio are 0 by construction (the
prediction is constant), while ``1 - MAE`` is deceptively high -- the point of
the baseline.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import torch

from personalitylinmult.data.schema import BIG_FIVE_TRAITS, SUBSET_CODES
from personalitylinmult.train.metrics import (
    clamp_predictions,
    concordance_correlation,
    one_minus_mae,
    pearson_r,
    r2_score,
    std_ratio,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("results/FI/bigfive/fi_baseline_train-mean")
"""Where the metrics land, beside the trained runs' output directories."""

METRICS_FILE = "metrics_test.json"
"""Same file name a trained run writes."""


def read_split_labels(h5_path: Path, subset: str, traits: tuple[str, ...]) -> torch.Tensor:
    """Read the label matrix of one split.

    Args:
        h5_path (Path): Dataset HDF5 file.
        subset (str): ``"train"``, ``"valid"`` or ``"test"``.
        traits (tuple[str, ...]): Label columns to read, in output order.

    Returns:
        torch.Tensor: ``(N, len(traits))`` float32 labels of the split's rows.
    """
    with h5py.File(h5_path, "r") as fh:
        rows = np.flatnonzero(fh["subset"][:] == SUBSET_CODES[subset])
        columns = [fh[f"labels/{trait}"][:][rows] for trait in traits]
    return torch.from_numpy(np.stack(columns, axis=1).astype(np.float32))


def evaluate_mean_baseline(
    h5_path: Path, traits: tuple[str, ...] = BIG_FIVE_TRAITS
) -> dict[str, float]:
    """Score the train-mean predictor on the test split.

    Args:
        h5_path (Path): Dataset HDF5 file.
        traits (tuple[str, ...]): Traits to evaluate.

    Returns:
        dict[str, float]: Metrics keyed like a trained run's
        ``metrics_test.json`` (``test/ccc/<trait>``, ``test/mean_ccc``, ...,
        ``test/mse``).
    """
    train = read_split_labels(h5_path, "train", traits)
    target = read_split_labels(h5_path, "test", traits)
    preds = clamp_predictions(train.mean(dim=0, keepdim=True).expand_as(target))

    metrics: dict[str, float] = {}
    for name, values in (
        ("one_minus_mae", one_minus_mae(preds, target)),
        ("pearson_r", pearson_r(preds, target)),
        ("ccc", concordance_correlation(preds, target)),
        ("r2", r2_score(preds, target)),
        ("std_ratio", std_ratio(preds, target)),
    ):
        for trait, value in zip(traits, values, strict=True):
            metrics[f"test/{name}/{trait}"] = float(value)
        metrics[f"test/mean_{name}"] = float(values.mean())
    metrics["test/mse"] = float((preds - target).pow(2).mean())
    return metrics


def run_mean_baseline(h5_path: Path, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, float]:
    """Evaluate the baseline and write ``metrics_test.json``.

    Args:
        h5_path (Path): Dataset HDF5 file.
        output_dir (Path): Directory for the metrics file; created if missing.

    Returns:
        dict[str, float]: The metrics that were written.
    """
    metrics = evaluate_mean_baseline(h5_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / METRICS_FILE).write_text(json.dumps(metrics, indent=2))
    logger.info(f"train-mean test/mean_ccc = {metrics['test/mean_ccc']:.4f}")
    logger.info(f"train-mean test/mean_one_minus_mae = {metrics['test/mean_one_minus_mae']:.4f}")
    return metrics


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--h5", type=Path, default=Path("data/processed/FI/h5/fi.h5"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    run_mean_baseline(args.h5, args.output_dir)


if __name__ == "__main__":
    main()
