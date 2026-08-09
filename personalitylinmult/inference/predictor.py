"""Load a trained FI checkpoint and predict Big Five scores from features.

A checkpoint is self-describing: :meth:`FiLightningModule.load_from_checkpoint`
rebuilds the exact model from the ``feature_names`` / ``input_feature_dim`` /
configs stored in its hyperparameters, so :class:`FiPredictor` needs no external
metadata — the path alone is enough.

The predictor is deliberately model-agnostic (the checkpoint decides which
features it consumes and which traits it emits), so the same class serves the
audio-only (A) and full multimodal (AVT) models without a code change.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from personalitylinmult.train.metrics import clamp_predictions
from personalitylinmult.train.module import FiLightningModule

logger = logging.getLogger(__name__)
"""Module-level logger."""


class InferenceError(Exception):
    """Raised when features do not match what the checkpoint expects."""


class FiPredictor:
    """Predicts Big Five trait scores from per-modality features.

    Args:
        module (FiLightningModule): A loaded, eval-mode Lightning module.

    Attributes:
        feature_names (list[str]): Features this model consumes, in input order.
        traits (list[str]): Trait names, in output-column order.
    """

    def __init__(self, module: FiLightningModule):
        self.module = module.eval()
        self.feature_names: list[str] = list(module.feature_names)
        self.traits: list[str] = list(module.train_config.targets)

    @classmethod
    def from_checkpoint(cls, path: str | Path, map_location: str = "cpu") -> FiPredictor:
        """Load a predictor from a training checkpoint.

        Args:
            path (str | Path): Path to a ``.ckpt`` written by training.
            map_location (str): Device to map tensors onto. ``"cpu"`` keeps
                inference portable (a demo host rarely has the training GPU).

        Returns:
            FiPredictor: Ready to predict.

        Raises:
            FileNotFoundError: If the checkpoint does not exist.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"No checkpoint at {path}.")
        module = FiLightningModule.load_from_checkpoint(str(path), map_location=map_location)
        logger.info(
            f"Loaded {type(module.model).__name__} from {path} "
            f"(features={module.feature_names}, traits={list(module.train_config.targets)})."
        )
        return cls(module)

    def _to_batch(
        self, features: dict[str, np.ndarray]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Turn a ``{name: (T, F)}`` mapping into the model's ``(inputs, masks)``.

        Each feature is padded to a batch of one and paired with an all-valid
        mask: a caller supplying features has already trimmed padding, so every
        provided timestep is real.

        Args:
            features (dict[str, np.ndarray]): One ``(T, F)`` array per feature
                the model expects.

        Returns:
            tuple: ``(inputs, masks)``, each a list of one ``(1, T, F)`` /
            ``(1, T)`` tensor per feature, in model input order.

        Raises:
            InferenceError: If a required feature is missing, an extra feature is
                supplied, or an array is not 2-D.
        """
        missing = [name for name in self.feature_names if name not in features]
        if missing:
            raise InferenceError(
                f"Missing features {missing}. This model needs {self.feature_names}."
            )
        extra = [name for name in features if name not in self.feature_names]
        if extra:
            raise InferenceError(
                f"Unexpected features {extra}. This model consumes only {self.feature_names}."
            )

        inputs: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for name in self.feature_names:
            array = np.asarray(features[name])
            if array.ndim != 2:
                raise InferenceError(
                    f"Feature {name!r} must be 2-D (T, F); got shape {array.shape}."
                )
            tensor = torch.from_numpy(array).float().unsqueeze(0)  # (1, T, F)
            inputs.append(tensor)
            masks.append(torch.ones(tensor.shape[:2], dtype=torch.bool))  # (1, T)
        return inputs, masks

    @torch.inference_mode()
    def predict(self, features: dict[str, np.ndarray]) -> dict[str, float]:
        """Predict Big Five scores for one sample.

        Args:
            features (dict[str, np.ndarray]): One ``(T, F)`` array per feature the
                model expects (see :attr:`feature_names`).

        Returns:
            dict[str, float]: ``{trait: score}`` with scores clamped to ``[0, 1]``,
            in the model's trait order.

        Raises:
            InferenceError: If the features do not match the model.
        """
        inputs, masks = self._to_batch(features)
        preds = clamp_predictions(self.module(inputs, masks))  # (1, M)
        row = preds.squeeze(0).tolist()
        return dict(zip(self.traits, row, strict=True))
