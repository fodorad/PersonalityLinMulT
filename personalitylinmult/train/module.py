"""The Lightning module wrapping LinMulT for the FI regression tasks."""

from __future__ import annotations

import logging
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Any

import lightning as L
from linmult import HeadConfig, LinMulT, LinMulTConfig, LinT, LinTConfig

from personalitylinmult.data.collate import unpack_batch
from personalitylinmult.train.config import ModelConfig, TrainConfig
from personalitylinmult.train.losses import build_loss
from personalitylinmult.train.metrics import (
    PRIMARY_METRIC,
    TraitRegressionMetrics,
    clamp_predictions,
)
from personalitylinmult.train.optim import build_optimizer, build_scheduler

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)
"""Module-level logger."""

HEAD_NAME = "app"
"""Name of the regression head.

"app" = automatic personality perception, kept from the original work so that
old and new runs remain comparable by name.
"""


def _as_model_config(config: ModelConfig | dict[str, Any]) -> ModelConfig:
    """Coerce a checkpoint's stored dict back into a typed config.

    Args:
        config (ModelConfig | dict): A config or its ``asdict`` form.

    Returns:
        ModelConfig: The typed config.
    """
    return config if isinstance(config, ModelConfig) else ModelConfig(**config)


def _as_train_config(config: TrainConfig | dict[str, Any]) -> TrainConfig:
    """Coerce a checkpoint's stored dict back into a typed config.

    Args:
        config (TrainConfig | dict): A config or its ``asdict`` form.

    Returns:
        TrainConfig: The typed config, with nested sections rebuilt.
    """
    if isinstance(config, TrainConfig):
        return config
    return TrainConfig.from_dict(config)


def _head(model_config: ModelConfig, output_dim: int) -> HeadConfig:
    """Build the single regression head shared by both families.

    Args:
        model_config (ModelConfig): Architecture settings.
        output_dim (int): Head width, derived from the task's targets.

    Returns:
        HeadConfig: The head.
    """
    return HeadConfig(
        type=model_config.head_type,
        output_dim=output_dim,
        name=HEAD_NAME,
        dropout=model_config.head_dropout,
        hidden_dim=model_config.head_hidden_dim,
        norm=model_config.head_norm,
    )


def _attention_kwargs(model_config: ModelConfig) -> dict[str, Any]:
    """Attention-variant fields common to both config classes.

    Args:
        model_config (ModelConfig): Architecture settings.

    Returns:
        dict: Keyword arguments accepted by both ``LinMulTConfig`` and
        ``LinTConfig``.
    """
    return {
        "attention_type": model_config.attention_type,
        "flash_query_key_dim": model_config.flash_query_key_dim,
        "performer_num_random_features": model_config.performer_num_random_features,
        "bigbird_block_size": model_config.bigbird_block_size,
        "bigbird_num_global_tokens": model_config.bigbird_num_global_tokens,
        "bigbird_num_random_tokens": model_config.bigbird_num_random_tokens,
    }


def build_model(
    model_config: ModelConfig,
    input_feature_dim: list[int],
    output_dim: int,
) -> LinMulT | LinT:
    """Construct the model for the configured family.

    ``family='lint'`` builds a single-stream :class:`LinT` (one feature only);
    ``family='linmult'`` builds the multimodal :class:`LinMulT`. Both accept
    only their typed config — the old code passed a merged YAML dict, which 2.x
    rejects outright.

    Args:
        model_config (ModelConfig): Architecture settings.
        input_feature_dim (list[int]): Per-modality feature widths, derived from
            the HDF5 rather than hand-written.
        output_dim (int): Head width, derived from the task's targets.

    Returns:
        LinMulT | LinT: The model.

    Raises:
        ValueError: If ``family='lint'`` is given more than one feature.
    """
    # Annotated dict[str, Any] so **-splatting into the typed configs is accepted:
    # a heterogeneous literal dict would make the checker try (and fail) to match
    # each value's union type to every keyword parameter.
    common: dict[str, Any] = {
        "name": model_config.name,
        "d_model": model_config.d_model,
        "num_heads": model_config.num_heads,
        "cmt_num_layers": model_config.cmt_num_layers,
        "dropout_input": model_config.dropout_input,
        "dropout_output": model_config.dropout_output,
        "dropout_pe": model_config.dropout_pe,
        "dropout_ffn": model_config.dropout_ffn,
        "dropout_attention": model_config.dropout_attention,
        "add_module_tcn": model_config.add_module_tcn,
        "tcn_num_layers": model_config.tcn_num_layers,
        "tcn_kernel_size": model_config.tcn_kernel_size,
        "tcn_dropout": model_config.tcn_dropout,
        "time_dim_reducer": model_config.time_dim_reducer,
        "add_module_ffn_fusion": model_config.add_module_ffn_fusion,
        "heads": [_head(model_config, output_dim)],
        **_attention_kwargs(model_config),
    }

    if model_config.family == "lint":
        if len(input_feature_dim) != 1:
            raise ValueError(
                f"LinT is single-stream but was given {len(input_feature_dim)} "
                "features. Use family='linmult' for more than one feature."
            )
        return LinT(LinTConfig(input_feature_dim=input_feature_dim[0], **common))

    return LinMulT(
        LinMulTConfig(
            input_feature_dim=input_feature_dim,
            branch_sat_num_layers=model_config.branch_sat_num_layers,
            # Cross-modal MM-signal module (LinMulT-only): LinT has no cross-modal
            # fusion, so these are not part of the shared `common` kwargs.
            add_module_multimodal_signal=model_config.add_module_multimodal_signal,
            mms_num_layers=model_config.mms_num_layers,
            **common,
        )
    )


class FiLightningModule(L.LightningModule):
    """Trains LinMulT on one FI regression task.

    The model is built here from its config rather than injected, so
    ``load_from_checkpoint(path)`` reconstructs a run without the caller having
    to rebuild and pass the architecture.

    Args:
        model_config (ModelConfig): Architecture settings.
        train_config (TrainConfig): Task, loss, and optimizer settings.
        feature_names (list[str]): Model input order; also passed to LinMulT so
            it can route per-feature special handling.
        input_feature_dim (list[int]): Per-modality widths, from the HDF5.
        steps_per_epoch (int): Optimizer steps per epoch, for step schedules.
    """

    def __init__(
        self,
        model_config: ModelConfig | dict[str, Any],
        train_config: TrainConfig | dict[str, Any],
        feature_names: list[str],
        input_feature_dim: list[int],
        steps_per_epoch: int = 1,
    ):
        super().__init__()

        # Configs are stored as plain dicts, not dataclass instances. Torch 2.6+
        # loads checkpoints with weights_only=True, which refuses to unpickle
        # arbitrary classes -- saving the dataclasses themselves would make
        # every checkpoint unloadable without an allowlist.
        model_config = _as_model_config(model_config)
        train_config = _as_train_config(train_config)
        self.save_hyperparameters(
            {
                "model_config": asdict(model_config),
                "train_config": asdict(train_config),
                "feature_names": list(feature_names),
                "input_feature_dim": list(input_feature_dim),
                "steps_per_epoch": steps_per_epoch,
            }
        )

        self.model_config = model_config
        self.train_config = train_config
        self.feature_names = list(feature_names)
        self.input_feature_dim = list(input_feature_dim)
        self.steps_per_epoch = steps_per_epoch

        self.model = build_model(model_config, self.input_feature_dim, train_config.output_dim)
        self.is_lint = model_config.family == "lint"
        self.loss_fn = build_loss(train_config.loss)

        traits = list(train_config.targets)
        self.train_metrics = TraitRegressionMetrics(traits)
        self.valid_metrics = TraitRegressionMetrics(traits)
        self.test_metrics = TraitRegressionMetrics(traits)

        self.test_video_ids: list[str] = []

    @property
    def lr(self) -> float:
        """The optimizer's peak learning rate.

        A thin proxy onto ``train_config.optimizer.lr`` under the flat name
        Lightning's LR finder requires (it looks for a top-level ``lr`` or
        ``learning_rate`` attribute to read and write back the LR it swept to;
        see :mod:`~personalitylinmult.train.lr_find`). Nothing in the normal
        training path reads this -- :meth:`configure_optimizers` reads
        ``train_config.optimizer.lr`` directly.
        """
        return self.train_config.optimizer.lr

    @lr.setter
    def lr(self, value: float) -> None:
        # OptimConfig/TrainConfig are frozen dataclasses (every other config in
        # this project is immutable once loaded); replace() rebuilds them with
        # just this field changed rather than mutating in place.
        new_optimizer = replace(self.train_config.optimizer, lr=value)
        self.train_config = replace(self.train_config, optimizer=new_optimizer)

    def forward(self, inputs: list[torch.Tensor], masks: list[torch.Tensor]) -> torch.Tensor:
        """Run the model and select the regression head.

        Args:
            inputs (list[torch.Tensor]): One tensor per modality.
            masks (list[torch.Tensor]): One mask per modality, ``True`` = valid.

        Returns:
            torch.Tensor: Predictions, ``(B, output_dim)``.
        """
        # Both families return a dict keyed by head name. The old code indexed
        # it with [0], which is a KeyError, not "the first head". LinT is
        # single-stream: it takes one tensor and a singular ``name``, not lists.
        if self.is_lint:
            outputs = self.model(inputs[0], masks[0], name=self.feature_names[0])
        else:
            outputs = self.model(inputs, masks, names=self.feature_names)
        return outputs[HEAD_NAME]

    def _shared_step(
        self, batch: dict[str, Any], metrics: TraitRegressionMetrics
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one step and accumulate its metrics.

        Args:
            batch (dict): A collated batch.
            metrics (TraitRegressionMetrics): Accumulator for this split.

        Returns:
            tuple: ``(loss, predictions)``.
        """
        inputs, masks, _ = unpack_batch(batch, self.feature_names)
        targets = batch["targets"]

        preds = self(inputs, masks)
        # The loss sees raw predictions: clamping here would zero the gradient
        # exactly where the model is most wrong.
        loss = self.loss_fn(preds, targets)
        metrics.update(preds, targets)
        return loss, preds

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Run one training step.

        Args:
            batch (dict): A collated batch.
            batch_idx (int): Batch index within the epoch.

        Returns:
            torch.Tensor: The loss to backpropagate.
        """
        loss, _ = self._shared_step(batch, self.train_metrics)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        # UncertaintyWeightedLoss's log_sigma is otherwise invisible outside a
        # checkpoint: logging it lets a real run confirm whether a term's
        # learned weight actually recedes once it is no longer needed, rather
        # than assuming the mechanism worked.
        log_sigma = getattr(self.loss_fn, "log_sigma", None)
        if log_sigma is not None:
            for i, value in enumerate(log_sigma):
                self.log(f"train/loss_log_sigma_{i}", value, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Run one validation step.

        Args:
            batch (dict): A collated batch.
            batch_idx (int): Batch index within the epoch.

        Returns:
            torch.Tensor: The loss.
        """
        loss, _ = self._shared_step(batch, self.valid_metrics)
        self.log("valid/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Run one test step.

        Args:
            batch (dict): A collated batch.
            batch_idx (int): Batch index within the epoch.

        Returns:
            torch.Tensor: The loss.
        """
        loss, _ = self._shared_step(batch, self.test_metrics)
        self.test_video_ids.extend(batch["video_id"])
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        return loss

    def predict_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        """Predict one batch, keeping clip ids alongside the predictions.

        Args:
            batch (dict): A collated batch.
            batch_idx (int): Batch index.

        Returns:
            dict: ``video_id`` and clamped ``predictions``.
        """
        inputs, masks, _ = unpack_batch(batch, self.feature_names)
        return {
            "video_id": batch["video_id"],
            "predictions": clamp_predictions(self(inputs, masks)),
        }

    def _log_epoch(self, split: str, metrics: TraitRegressionMetrics) -> None:
        """Log and reset one split's epoch metrics.

        Args:
            split (str): ``"train"``, ``"valid"``, or ``"test"``.
            metrics (TraitRegressionMetrics): The accumulator.
        """
        for name, value in metrics.compute().items():  # ty: ignore[missing-argument]
            self.log(
                f"{split}/{name}",
                value,
                prog_bar=name == PRIMARY_METRIC and split == "valid",
            )
        metrics.reset()

    def on_train_epoch_end(self) -> None:
        """Log the training epoch's metrics."""
        self._log_epoch("train", self.train_metrics)

    def on_validation_epoch_end(self) -> None:
        """Log the validation epoch's metrics."""
        self._log_epoch("valid", self.valid_metrics)

    def on_test_epoch_start(self) -> None:
        """Clear per-clip and metric state before a test epoch.

        Resetting here rather than at epoch end keeps the accumulated
        predictions readable by the artifact-writing callbacks, whatever order
        they run in, while still keeping repeated ``trainer.test()`` calls from
        accumulating on top of each other.
        """
        self.test_video_ids = []
        self.test_metrics.reset()

    def on_test_epoch_end(self) -> None:
        """Log the test epoch's metrics.

        The accumulator is deliberately left populated: the prediction-writing
        and plotting callbacks read it after this hook.
        """
        for name, value in self.test_metrics.compute().items():  # ty: ignore[missing-argument]
            self.log(f"test/{name}", value)

    def configure_optimizers(self) -> dict[str, Any]:  # ty: ignore[invalid-method-override]
        """Build the optimizer and LR schedule.

        Returns:
            dict: Lightning's optimizer configuration.
        """
        optimizer = build_optimizer(self.train_config.optimizer, self.parameters())
        scheduler = build_scheduler(
            self.train_config.optimizer,
            optimizer,
            steps_per_epoch=self.steps_per_epoch,
            max_epochs=self.train_config.max_epochs,
        )
        if scheduler is None:
            return {"optimizer": optimizer}
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
