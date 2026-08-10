"""Resolve a model *name* to its checkpoint and load a predictor.

Deployment surfaces (a Gradio dropdown, a CLI ``--model`` flag) select a model by
a short name — ``"AVT"`` or ``"A"`` — rather than a checkpoint path. This module
holds that name-to-checkpoint mapping and hands back a ready
:class:`~personalitylinmult.inference.predictor.FiPredictor`, caching each so a
repeated selection does not reload from disk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from personalitylinmult.inference.predictor import FiPredictor

logger = logging.getLogger(__name__)
"""Module-level logger."""


@dataclass(frozen=True)
class ModelSpec:
    """One registered model.

    Args:
        name (str): Short selector, e.g. ``"AVT"`` or ``"A"``.
        checkpoint (Path): Path to its ``.ckpt``.
        description (str): One-line human summary shown in a UI.
    """

    name: str
    checkpoint: Path
    description: str = ""


class ModelRegistry:
    """A name-keyed set of models, loaded on demand and cached.

    Args:
        specs (list[ModelSpec]): The registered models.

    Raises:
        ValueError: If two specs share a name.
    """

    def __init__(self, specs: list[ModelSpec]):
        names = [s.name for s in specs]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"Duplicate model names in registry: {sorted(dupes)}.")
        self._specs = {s.name: s for s in specs}
        self._cache: dict[str, FiPredictor] = {}

    @property
    def names(self) -> list[str]:
        """The registered model names, in insertion order."""
        return list(self._specs)

    def spec(self, name: str) -> ModelSpec:
        """Return the spec for a name.

        Args:
            name (str): A registered model name.

        Returns:
            ModelSpec: Its spec.

        Raises:
            KeyError: If the name is not registered.
        """
        if name not in self._specs:
            raise KeyError(f"No model named {name!r}. Registered: {self.names}.")
        return self._specs[name]

    def load(self, name: str) -> FiPredictor:
        """Load (or return the cached) predictor for a name.

        Args:
            name (str): A registered model name.

        Returns:
            FiPredictor: The predictor.

        Raises:
            KeyError: If the name is not registered.
            FileNotFoundError: If its checkpoint is missing.
        """
        if name not in self._cache:
            spec = self.spec(name)
            self._cache[name] = FiPredictor.from_checkpoint(spec.checkpoint)
        return self._cache[name]

    @classmethod
    def from_checkpoints(cls, checkpoints: dict[str, Path | str]) -> ModelRegistry:
        """Build a registry from a ``{name: checkpoint_path}`` mapping.

        Args:
            checkpoints (dict[str, Path | str]): Name-to-path mapping.

        Returns:
            ModelRegistry: The registry (paths are not opened until
            :meth:`load`, so a missing file surfaces at load time, not here).
        """
        specs = [ModelSpec(name=name, checkpoint=Path(path)) for name, path in checkpoints.items()]
        return cls(specs)
