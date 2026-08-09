"""Wrap an FI checkpoint as an MLflow ``pyfunc`` model and register it.

Training logs raw ``.ckpt`` artifacts, which ``mlflow models serve`` cannot load
on their own. This module wraps a checkpoint in the standard ``pyfunc`` flavour —
a self-contained model MLflow can serve as a REST endpoint or bake into a Docker
image — and registers it in the Model Registry so it carries a version.

Promotion uses **aliases** (:func:`set_champion`), not the deprecated Staging /
Production stage mechanism: an alias is scoped to one registered-model *name*,
so each model family (e.g. the AVT fusion model and the audio-only model) keeps
its own independent ``@champion`` without any conflict, and moving the alias to
a new version is a single, deliberate call -- never automatic on registration.

The wrapped model consumes the same per-modality features as
:class:`~personalitylinmult.inference.predictor.FiPredictor`: at serving time the
JSON payload is ``{"inputs": {"wavlm": [[...]], "emotion2vec": [[...]]}}`` — one
``(T, F)`` array per feature the model expects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)
"""Module-level logger."""


class FiPyfunc:
    """An ``mlflow.pyfunc.PythonModel`` around an FI checkpoint.

    Not typed as a subclass at import time so this module stays importable
    without mlflow installed; :func:`log_and_register` binds it to
    ``mlflow.pyfunc.PythonModel`` where mlflow is guaranteed present.
    """

    def load_context(self, context: Any) -> None:
        """Load the checkpoint packaged with the model.

        Args:
            context: MLflow context; ``context.artifacts['checkpoint']`` is the
                packaged ``.ckpt`` path.
        """
        from personalitylinmult.inference.predictor import FiPredictor

        self.predictor = FiPredictor.from_checkpoint(context.artifacts["checkpoint"])

    def predict(self, context: Any, model_input: Any, params: Any = None) -> list[dict[str, float]]:  # noqa: ARG002 -- context/params are required by the mlflow pyfunc predict signature
        """Predict Big Five scores for one or more samples.

        Args:
            context: MLflow context (unused; the model is already loaded).
            model_input: A ``{feature_name: (T, F) array}`` mapping for one
                sample, or a list of such mappings for a batch.
            params: Optional inference params (unused).

        Returns:
            list[dict[str, float]]: One ``{trait: score}`` dict per input sample.
        """
        samples = model_input if isinstance(model_input, list) else [model_input]
        results = []
        for sample in samples:
            features = {name: np.asarray(array) for name, array in sample.items()}
            results.append(self.predictor.predict(features))
        return results


def _mlflow_model() -> Any:
    """Return an ``mlflow.pyfunc.PythonModel`` instance wrapping :class:`FiPyfunc`.

    Returns:
        A pyfunc model MLflow can log. Built here so the class body above does
        not import mlflow at module import time.
    """
    import mlflow.pyfunc

    # Compose rather than subclass at definition time so this module imports
    # without mlflow; the delegate holds the real logic.
    delegate = FiPyfunc()

    class _Model(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            delegate.load_context(context)

        def predict(self, context, model_input, params=None):
            return delegate.predict(context, model_input, params)

    return _Model()


REGISTRATION_EXPERIMENT = "FI-Model-Registry"
"""Experiment the pyfunc registration runs log under.

Set explicitly so registration never falls back to the default experiment
(id 0), which may be in a ``deleted`` state and would reject a new run.
"""


def log_and_register(
    checkpoint: str | Path,
    registered_model_name: str,
    run_id: str | None = None,
    artifact_path: str = "pyfunc_model",
    experiment: str = REGISTRATION_EXPERIMENT,
) -> str:
    """Log the checkpoint as a pyfunc model and register it.

    Args:
        checkpoint (str | Path): Path to the ``.ckpt`` to wrap.
        registered_model_name (str): Name in the Model Registry (e.g.
            ``"fi-bigfive-avt"``).
        run_id (str | None): Existing run to log under; a new run is started
            under ``experiment`` when ``None``.
        artifact_path (str): Artifact subpath for the model.
        experiment (str): Experiment to log the registration run under when
            ``run_id`` is ``None``.

    Returns:
        str: The registered model version.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
    """
    import mlflow

    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"No checkpoint at {checkpoint}.")

    # Ensure an active experiment so a fresh registration run is not dropped into
    # the (possibly deleted) default experiment.
    if run_id is None:
        mlflow.set_experiment(experiment)

    with mlflow.start_run(run_id=run_id):
        info = mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=_mlflow_model(),
            artifacts={"checkpoint": str(checkpoint)},
            registered_model_name=registered_model_name,
            code_paths=[str(Path(__file__).resolve().parents[1])],
        )
    version = _latest_version(registered_model_name)
    logger.info(
        f"Registered {registered_model_name} v{version} from {checkpoint} "
        f"(model_uri={info.model_uri})."
    )
    return version


def _latest_version(registered_model_name: str) -> str:
    """Return the newest version number of a registered model.

    Args:
        registered_model_name (str): The registered model name.

    Returns:
        str: The highest version number as a string.
    """
    import mlflow

    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name = '{registered_model_name}'")
    return str(max(int(v.version) for v in versions))


def set_champion(registered_model_name: str, version: str, alias: str = "champion") -> None:
    """Promote a registered version to an alias.

    A deliberate, separate step from registration: a new version is never
    auto-promoted, so a run finishing (or even being registered) never
    silently changes what a deployment currently serves. Aliases are per
    registered-model name, so multiple model families (AVT, audio-only, ...)
    each carry their own independent ``@champion`` with no conflict.

    Args:
        registered_model_name (str): The registered model name.
        version (str): The version to promote.
        alias (str): Alias to set. Defaults to ``"champion"``; a second alias
            (e.g. ``"candidate"``) can stage a version before promoting it.
    """
    import mlflow

    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(registered_model_name, alias, version)
    logger.info(f"{registered_model_name}@{alias} -> v{version}.")
