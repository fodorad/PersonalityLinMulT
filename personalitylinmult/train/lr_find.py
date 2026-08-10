"""Find a good peak learning rate for a run, via Lightning's LR range test.

Ramps the learning rate exponentially over a few hundred steps, records the
loss at each, and suggests the point of steepest descent (just before the loss
starts diverging) -- the standard LR-range-test heuristic. Prints the
suggestion and saves a loss-vs-lr plot so the choice can be checked by eye, not
just trusted blindly (the automatic suggestion can be fooled by a noisy loss
curve).

This only probes the LR; it does not train the model or write a checkpoint.

Example:
    Find a peak LR for ``fi_lint_wavlm``, saving the plot under its own
    ``results/FI/bigfive/fi_lint_wavlm/lr_find.png``::

        uv run python -m personalitylinmult.train.lr_find \
            --data config/FI/data/wavlm.yaml \
            --model config/FI/model/lint.yaml \
            --train config/FI/train/bigfive.yaml \
            --set train.mlflow.experiment_name=fi_lint_wavlm
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import lightning as L
from lightning.pytorch.tuner import Tuner

from personalitylinmult.data.datamodule import FiDataModule
from personalitylinmult.train.cli import read_input_dims
from personalitylinmult.train.config import ExperimentConfig, parse_override
from personalitylinmult.train.module import FiLightningModule

logger = logging.getLogger(__name__)
"""Module-level logger."""

DEFAULT_MIN_LR = 1e-6
"""Sweep start."""

DEFAULT_MAX_LR = 1.0
"""Sweep end -- high enough to see the loss diverge for any reasonable model."""

DEFAULT_NUM_TRAINING_STEPS = 200
"""Steps in the sweep. Lightning's default (100) is a bit coarse; 200 gives a
smoother curve without taking meaningfully longer on a single-feature model."""


def _output_path(config: ExperimentConfig) -> Path:
    """The plot path for this run.

    Lives in the same per-experiment results directory every other artifact
    (checkpoints, test predictions) uses.

    Args:
        config (ExperimentConfig): The run the LR is being chosen for.

    Returns:
        Path: ``{output_dir}/{task}/{experiment_name}/lr_find.png``.
    """
    return (
        Path(config.train.output_dir)
        / config.train.task
        / config.train.mlflow.experiment_name
        / "lr_find.png"
    )


def find_lr(
    config: ExperimentConfig,
    output_path: Path | None = None,
    min_lr: float = DEFAULT_MIN_LR,
    max_lr: float = DEFAULT_MAX_LR,
    num_training: int = DEFAULT_NUM_TRAINING_STEPS,
) -> float:
    """Run the LR range test and save its plot.

    Args:
        config (ExperimentConfig): The run this LR is being chosen for (same
            data/model/train configs a real run would use).
        output_path (Path | None): Where to save the loss-vs-lr plot (PNG).
            ``None`` (the default) saves it under the run's own
            ``results/FI/<task>/<experiment_name>/lr_find.png``, alongside its
            other artifacts -- see :func:`_output_path`.
        min_lr (float): Sweep start.
        max_lr (float): Sweep end.
        num_training (int): Steps in the sweep.

    Returns:
        float: The suggested peak learning rate.

    Raises:
        RuntimeError: If the finder could not produce a suggestion (e.g. the
            loss never decreased at any point in the sweep).
    """
    L.seed_everything(config.train.seed, workers=True)

    datamodule = FiDataModule(config.data, target_names=config.train.targets)
    datamodule.setup("fit")
    steps_per_epoch = max(1, len(datamodule.train_dataloader()))

    module = FiLightningModule(
        model_config=config.model,
        train_config=config.train,
        feature_names=config.data.features,
        input_feature_dim=read_input_dims(config.data),
        steps_per_epoch=steps_per_epoch,
    )

    trainer = L.Trainer(
        accelerator=config.train.accelerator,
        devices=config.train.devices,
        precision=config.train.precision,  # ty: ignore[invalid-argument-type]
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(
        module,
        datamodule=datamodule,
        min_lr=min_lr,
        max_lr=max_lr,
        num_training=num_training,
    )
    if lr_finder is None:
        raise RuntimeError(
            "LR finder produced no result -- the loss may never have "
            "decreased across the swept range. Check min_lr/max_lr."
        )
    raw_suggestion = lr_finder.suggestion()
    if raw_suggestion is None:
        raise RuntimeError(
            "LR finder produced no suggestion -- the loss may never have "
            "decreased across the swept range. Check min_lr/max_lr."
        )
    suggestion = float(raw_suggestion)

    if output_path is None:
        output_path = _output_path(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Lightning's own plot(suggest=True) only marks the suggestion with a red
    # dot -- no number on it, which is the whole point of eyeballing this
    # rather than trusting the suggestion blindly. Annotate the value directly.
    fig = lr_finder.plot(suggest=True)
    if fig is None:
        raise RuntimeError("LR finder could not produce a plot (matplotlib unavailable?).")
    ax = fig.axes[0]
    ax.set_title(f"LR range test -- suggested lr = {suggestion:.2e}")
    ax.annotate(
        f"{suggestion:.2e}",
        xy=(suggestion, lr_finder.results["loss"][lr_finder._optimal_idx]),
        xytext=(10, 10),
        textcoords="offset points",
        color="red",
        fontweight="bold",
    )
    # lr_finder.plot() always builds its own plt.subplots() Figure (never a
    # SubFigure); the stub's wider Figure | SubFigure return type is imprecise.
    fig.savefig(output_path)  # ty: ignore[unresolved-attribute]
    logger.info(f"Saved LR-vs-loss plot to {output_path}")
    logger.info(f"Suggested lr: {suggestion:.2e}")
    return suggestion


def main() -> None:
    """CLI entry point."""
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
        help="Override a config value, e.g. --set data.features=[wavlm].",
    )
    parser.add_argument("--min-lr", type=float, default=DEFAULT_MIN_LR)
    parser.add_argument("--max-lr", type=float, default=DEFAULT_MAX_LR)
    parser.add_argument("--num-training", type=int, default=DEFAULT_NUM_TRAINING_STEPS)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save the loss-vs-lr plot. Default: "
        "results/FI/<task>/<experiment_name>/lr_find.png.",
    )
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    overrides = dict(parse_override(text) for text in args.overrides)
    config = ExperimentConfig.from_files(args.data, args.model, args.train, overrides)
    find_lr(config, args.output, args.min_lr, args.max_lr, args.num_training)


if __name__ == "__main__":
    main()
