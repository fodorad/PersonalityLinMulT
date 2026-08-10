"""Inference for the trained FI Big Five models.

The reusable predictor every deployment surface (CLI, MLflow pyfunc, HF Space,
notebook) wraps: :class:`~personalitylinmult.inference.predictor.FiPredictor`
loads a self-describing checkpoint and maps per-modality features to Big Five
scores. :mod:`~personalitylinmult.inference.registry` resolves a model *name*
(``"AVT"``, ``"A"``) to its checkpoint so callers select a model without knowing
paths.
"""

from personalitylinmult.inference.predictor import FiPredictor
from personalitylinmult.inference.registry import ModelRegistry, ModelSpec

__all__ = ["FiPredictor", "ModelRegistry", "ModelSpec"]
