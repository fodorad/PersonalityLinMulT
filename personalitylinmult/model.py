"""The public, pip-install-only entry point: load a champion model by id.

``PersonalityModel.from_pretrained(model_id)`` downloads a champion checkpoint
from the public HF Hub model repo and wraps it for prediction -- no MLflow, no
repo clone, no external config. This is deliberately the *only* public model
class: a checkpoint already knows whether it is a single-stream LinT or a
cross-modal LinMulT (see
:class:`~personalitylinmult.inference.predictor.FiPredictor`), so exposing
separate ``PersonalityLinT``/``PersonalityLinMulT`` classes would force a
caller to track architecture per id for no benefit -- this class reads it off
the checkpoint instead.

Model ids follow ``{features joined by "_", multiword feature names use "-"}_best-{metric}``,
e.g. ``"wavlm_best-ccc"`` (LinT, audio only) or
``"wavlm-emotion2vec_best-ccc"`` (LinMulT, audio fusion) -- the same convention
used for the MLflow registered-model names (see
:mod:`personalitylinmult.inference.register_models`) and the HF model repo
filenames.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from personalitylinmult.inference.predictor import FiPredictor

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)
"""Module-level logger."""

HF_MODEL_REPO = "fodorad/personalitylinmult-fi-models"
"""The public HF Hub model repo champion checkpoints are published to."""


def _checkpoint_filename(model_id: str) -> str:
    """Map a model id to its filename in the HF model repo.

    Args:
        model_id (str): e.g. ``"wavlm_best-ccc"``.

    Returns:
        str: e.g. ``"wavlm_best-ccc.ckpt"``.
    """
    return f"{model_id}.ckpt"


def _download(model_id: str, revision: str | None = None) -> Path:
    """Download a champion checkpoint from the HF model repo.

    A separate function (not inlined into :meth:`PersonalityModel.from_pretrained`)
    so tests can substitute a local checkpoint path without a network call --
    see ``tests/test_model.py``.

    Args:
        model_id (str): The champion id to download.
        revision (str | None): HF Hub revision (branch/tag/commit); ``None``
            uses the repo's default branch.

    Returns:
        Path: Local path to the downloaded ``.ckpt`` (cached under
        :data:`personalitylinmult.WEIGHTS_DIR` across calls).
    """
    from huggingface_hub import hf_hub_download

    # Imported lazily (not at module scope) so this module never forms an
    # import cycle with the package `__init__` that re-exports PersonalityModel.
    from personalitylinmult import WEIGHTS_DIR

    path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=_checkpoint_filename(model_id),
        revision=revision,
        cache_dir=str(WEIGHTS_DIR),
    )
    # lgtm[py/log-injection] -- repr(), not str(): model_id is a public API
    # argument (caller-supplied), and repr() escapes newlines/control
    # characters so it cannot forge extra log lines.
    logger.info(f"{model_id!r}: downloaded {path}.")
    return Path(path)


class PersonalityModel:
    """A champion FI model, loaded by id from the public HF Hub repo.

    Args:
        predictor (FiPredictor): The loaded predictor to wrap.
    """

    def __init__(self, predictor: FiPredictor):
        self._predictor = predictor

    @property
    def feature_names(self) -> list[str]:
        """Features this model consumes, in input order."""
        return self._predictor.feature_names

    @property
    def traits(self) -> list[str]:
        """Trait names, in output-column order."""
        return self._predictor.traits

    @classmethod
    def from_pretrained(cls, model_id: str, revision: str | None = None) -> PersonalityModel:
        """Download (or reuse the cached) champion checkpoint and load it.

        Args:
            model_id (str): A champion id, e.g. ``"wavlm_best-ccc"`` or
                ``"avt_best-ccc"``. See the module docstring for the naming
                convention.
            revision (str | None): HF Hub revision (branch/tag/commit);
                ``None`` uses the repo's default branch.

        Returns:
            PersonalityModel: Ready to predict.
        """
        checkpoint = _download(model_id, revision=revision)
        return cls(FiPredictor.from_checkpoint(checkpoint))

    def predict(self, features: dict[str, np.ndarray]) -> dict[str, float]:
        """Predict Big Five scores for one sample.

        Args:
            features (dict[str, np.ndarray]): One ``(T, F)`` array per feature
                the model expects (see :attr:`feature_names`).

        Returns:
            dict[str, float]: ``{trait: score}``, in the model's trait order.
        """
        return self._predictor.predict(features)
