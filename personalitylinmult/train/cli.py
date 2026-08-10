r"""Entry point for FI training runs.

Example:
    Train big-five regression::

        make train-fi-bigfive

    Or directly, for control over the individual configs::

        uv run python -m personalitylinmult.train.cli \
            --data config/FI/data/avt.yaml \
            --model config/FI/model/linmult.yaml \
            --train config/FI/train/bigfive.yaml

    Run the 10-second analysis window. The window is applied by the dataloader,
    so this needs no rebuild of the dataset::

        make train-fi-bigfive ARGS="--set data.window_seconds=10"
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import h5py
import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from personalitylinmult.data.datamodule import FiDataModule
from personalitylinmult.train.callbacks import (
    PlotCallback,
    PredictionWriter,
    TimeTrackingCallback,
)
from personalitylinmult.train.config import ExperimentConfig, parse_override
from personalitylinmult.train.metrics import PRIMARY_METRIC
from personalitylinmult.train.mlflow_utils import (
    build_logger,
    environment_params,
    log_artifacts,
    truncate_params,
)
from personalitylinmult.train.module import FiLightningModule

if TYPE_CHECKING:
    from personalitylinmult.train.config import DataConfig

# Lightning 2.6.5's own internals call torch.utils._pytree with an API torch
# 2.13 deprecated (isinstance(treespec, LeafSpec) instead of
# TreeSpec.is_leaf()); nothing in this project touches pytree directly, so
# there is no local fix -- silence the specific upstream message rather than
# either patch a dependency or let every run print noise it cannot act on.
# torch raises this via typing_extensions.deprecated, which defaults to
# FutureWarning, not UserWarning.
warnings.filterwarnings(
    "ignore",
    message=r"`isinstance\(treespec, LeafSpec\)` is deprecated",
    category=FutureWarning,
)

logger = logging.getLogger(__name__)
"""Module-level logger."""

MONITOR = f"valid/{PRIMARY_METRIC}"
"""Metric that selects the best checkpoint: mean trait-wise 1-MAE."""


def read_input_dims(config: DataConfig) -> list[int]:
    """Read each configured feature's width from the dataset itself.

    Deriving the widths from the file rather than the config is what makes a
    shape mismatch impossible: the model is built from the data it will be fed.

    Args:
        config (DataConfig): Dataloader configuration.

    Returns:
        list[int]: Feature widths, in model input order.

    Raises:
        FileNotFoundError: If the dataset has not been built.
        KeyError: If a configured feature is absent from the dataset.
    """
    path = Path(config.h5_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}. Build it first: make hdf5-fi")

    dims = []
    with h5py.File(path, "r") as fh:
        for name in config.features:
            if name not in fh["features"]:
                raise KeyError(
                    f"{path} has no feature {name!r}. It contains {sorted(fh['features'].keys())}."
                )
            dims.append(int(fh[f"features/{name}"].attrs["dim"]))
    return dims


def build_callbacks(config: ExperimentConfig, output_dir: Path) -> list[L.Callback]:
    """Assemble a run's callbacks.

    Args:
        config (ExperimentConfig): The run's configuration.
        output_dir (Path): Directory for run outputs.

    Returns:
        list[L.Callback]: The callbacks.
    """
    checkpoint_dir = output_dir / "checkpoints"

    # Keyed by purpose, not position. The old code did
    # `callbacks[1].best_model_path`, which silently depends on list order.
    checkpoints = {
        "best": ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best",
            monitor=MONITOR,
            mode="max",
            save_top_k=1,
            # Full state, not just weights: a run must be resumable.
            save_weights_only=False,
            save_last=True,
        ),
        "best_loss": ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best_loss",
            monitor="valid/loss",
            mode="min",
            save_top_k=1,
            save_weights_only=False,
        ),
    }

    callbacks: list[L.Callback] = [
        *checkpoints.values(),
        LearningRateMonitor(logging_interval="step"),
        TimeTrackingCallback(output_dir),
        PredictionWriter(output_dir),
        PlotCallback(output_dir),
    ]

    if config.train.early_stopping.enabled:
        callbacks.append(
            EarlyStopping(
                monitor=config.train.early_stopping.monitor,
                mode=config.train.early_stopping.mode,
                patience=config.train.early_stopping.patience,
                min_delta=config.train.early_stopping.min_delta,
            )
        )

    return callbacks


RUN_ID_FILE = "mlflow_run_id.txt"
"""Name of the file in a run's output dir that persists its MLflow run id.

Written on the first launch and read back on ``--resume`` so a continued run
appends to the same MLflow run instead of minting a second, disjoint one.
"""

MAX_EPOCHS_FILE = "max_epochs.txt"
"""Name of the file persisting the run's planned ``max_epochs``.

The OneCycle LR schedule is a fixed-length curve set at first launch; resuming
with a different ``max_epochs`` cannot extend it, so this is stored to reject
such a mismatch with a clear message.
"""


def run(
    config: ExperimentConfig,
    fast_dev_run: bool = False,
    resume: bool = False,
    extra_callbacks: list[L.Callback] | None = None,
) -> dict[str, Any]:
    """Train, then test, one configured experiment.

    Args:
        config (ExperimentConfig): The run's configuration.
        fast_dev_run (bool): Run a single batch through every stage, for
            smoke-testing.
        resume (bool): Continue a stopped run from its ``last.ckpt``, appending
            to the same MLflow run. Fresh start (the default) begins at epoch 0.
        extra_callbacks (list[L.Callback] | None): Additional Lightning
            callbacks appended after the standard set, e.g. an Optuna pruning
            callback during a hyperparameter search. ``None`` adds nothing.

    Returns:
        dict[str, float]: The test metrics.
    """
    L.seed_everything(config.train.seed, workers=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    # Per-experiment subdir so the modality groups (A / AT / AVT) keep separate
    # checkpoints and can each be stopped and resumed independently; without it
    # they share `task` and overwrite each other's last.ckpt.
    output_dir = (
        Path(config.train.output_dir) / config.train.task / config.train.mlflow.experiment_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume plumbing: on --resume, continue from last.ckpt and reuse the stored
    # MLflow run id so the metric curves stay continuous.
    ckpt_path: str | None = None
    run_id: str | None = None
    max_epochs_file = output_dir / MAX_EPOCHS_FILE
    if resume:
        last_ckpt = output_dir / "checkpoints" / "last.ckpt"
        if not last_ckpt.is_file():
            raise FileNotFoundError(
                f"--resume was given but no checkpoint at {last_ckpt}. Start a "
                "fresh run (drop --resume) or point at the right experiment."
            )
        # OneCycle's LR curve is fixed-length: it is built for the original
        # max_epochs and cannot be stepped past that total. Resuming with a
        # different max_epochs would either crash mid-schedule or silently change
        # the LR plan, so it is rejected -- resume finishes the run as designed.
        if max_epochs_file.is_file():
            planned = int(max_epochs_file.read_text().strip())
            if planned != config.train.max_epochs:
                raise ValueError(
                    f"--resume must keep the original max_epochs={planned}, but the "
                    f"config now says {config.train.max_epochs}. The LR schedule is "
                    "fixed at first launch; set max_epochs high enough before the "
                    "first run, then resume finishes it."
                )
        ckpt_path = str(last_ckpt)
        run_id_file = output_dir / RUN_ID_FILE
        run_id = run_id_file.read_text().strip() if run_id_file.is_file() else None
        logger.info(f"Resuming from {ckpt_path} (mlflow run_id={run_id}).")

    datamodule = FiDataModule(config.data, target_names=config.train.targets)
    datamodule.setup("fit")

    # Lightning's Trainer counts accumulate_grad_batches *forward* passes as
    # one optimizer step, so the step-based schedules (warmup, OneCycle) must
    # divide by it too, or they will warm up/anneal N times slower than the
    # actual number of weight updates.
    batches_per_epoch = len(datamodule.train_dataloader())
    steps_per_epoch = max(1, batches_per_epoch // config.train.accumulate_grad_batches)
    module = FiLightningModule(
        model_config=config.model,
        train_config=config.train,
        feature_names=config.data.features,
        input_feature_dim=read_input_dims(config.data),
        steps_per_epoch=steps_per_epoch,
    )

    mlflow_logger = build_logger(config, run_id=run_id)
    # Persist the run id on the first launch so a later --resume appends to this
    # same run. Reading mlflow_logger.run_id here also materialises the run.
    (output_dir / RUN_ID_FILE).write_text(str(mlflow_logger.run_id))
    # Record the planned horizon so --resume can reject a changed max_epochs.
    max_epochs_file.write_text(str(config.train.max_epochs))
    # Logged from the config rather than the module's hyperparameters: the
    # module's copy exists to make checkpoints self-contained, and letting
    # Lightning log it too would duplicate every param under a second name.
    module._log_hyperparams = False
    # Only on a fresh run: MLflow forbids changing a param, so re-logging on
    # resume would crash the moment anything differs (e.g. a raised max_epochs).
    # The params were captured on the first launch and still describe the run.
    if not resume:
        mlflow_logger.log_hyperparams(
            truncate_params({**config.to_flat_dict(), **environment_params(config)})
        )

    trainer = L.Trainer(
        max_epochs=config.train.max_epochs,
        accelerator=config.train.accelerator,
        devices=config.train.devices,
        # Validated against PRECISIONS at config load; Lightning types this as a
        # Literal union that a plain str cannot satisfy statically.
        precision=cast("Any", config.train.precision),
        deterministic=config.train.deterministic,
        gradient_clip_val=config.train.optimizer.gradient_clip_val,
        accumulate_grad_batches=config.train.accumulate_grad_batches,
        default_root_dir=output_dir,
        logger=mlflow_logger,
        callbacks=[*build_callbacks(config, output_dir), *(extra_callbacks or [])],
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        fast_dev_run=fast_dev_run,
    )

    # ckpt_path restores model + optimizer + scheduler + epoch/step, so training
    # picks up exactly where the stopped run left off (None = fresh start).
    trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)

    if fast_dev_run:
        trainer.test(module, datamodule=datamodule)
        return {}

    best_path = _best_checkpoint(trainer)
    logger.info(f"Testing best checkpoint: {best_path}")
    results = trainer.test(module, datamodule=datamodule, ckpt_path=best_path or None)
    metrics: dict[str, Any] = dict(results[0]) if results else {}

    (output_dir / "metrics_test.json").write_text(json.dumps(metrics, indent=2))
    log_artifacts(mlflow_logger, output_dir)

    if config.train.mlflow.log_model and best_path:
        mlflow_logger.experiment.log_artifact(
            mlflow_logger.run_id, best_path, artifact_path="model"
        )

    logger.info(f"test/{PRIMARY_METRIC} = {metrics.get(f'test/{PRIMARY_METRIC}')}")
    return metrics


def _best_checkpoint(trainer: L.Trainer) -> str:
    """Return the path of the checkpoint selected by the primary metric.

    Looked up by the metric it monitors rather than by position in the callback
    list, so adding or reordering callbacks cannot silently load the wrong
    model — the failure mode of the old ``callbacks[1].best_model_path``.

    Args:
        trainer (L.Trainer): The finished trainer.

    Returns:
        str: The checkpoint path, or ``""`` if there is none.
    """
    for callback in trainer.checkpoint_callbacks:
        if isinstance(callback, ModelCheckpoint) and callback.monitor == MONITOR:
            return callback.best_model_path
    return ""


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
        help="Override a config value, e.g. --set data.window_seconds=10.",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run one batch through every stage and exit.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a stopped run from its last.ckpt, appending to the same "
        "MLflow run. Without this, a re-run starts fresh at epoch 0.",
    )
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    overrides = dict(parse_override(text) for text in args.overrides)
    config = ExperimentConfig.from_files(args.data, args.model, args.train, overrides)
    run(config, fast_dev_run=args.fast_dev_run, resume=args.resume)


if __name__ == "__main__":
    main()
