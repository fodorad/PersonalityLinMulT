"""MLflow wiring for FI training runs.

Logging is explicit rather than via ``mlflow.pytorch.autolog()``: autolog picks
its own cadence and parameter names, which would fight the flattened dataclass
params and make two runs hard to compare. Everything logged here is chosen.

What makes a run reproducible is not just its hyperparameters but the exact data
it saw, so :func:`environment_params` records the training artifact's hash and
size alongside the library versions and the commit.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from personalitylinmult.train.config import ExperimentConfig

logger = logging.getLogger(__name__)
"""Module-level logger."""

MAX_PARAM_LENGTH = 500
"""MLflow rejects parameter values longer than this."""


def git_sha() -> str:
    """Return the current commit SHA.

    Returns:
        str: The SHA, or ``"unknown"`` outside a git checkout.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return result.stdout.strip()


def file_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hash a file's contents.

    Args:
        path (Path): File to hash.
        chunk_size (int): Read size in bytes.

    Returns:
        str: Hex digest, or ``"missing"`` if the file does not exist.
    """
    if not path.is_file():
        return "missing"
    digest = hashlib.md5()  # noqa: S324 -- artifact identity, not security
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_params(config: ExperimentConfig) -> dict[str, Any]:
    """Collect the facts that make a run reproducible.

    Args:
        config (ExperimentConfig): The run's configuration.

    Returns:
        dict: Library versions, the commit, and the training artifact's
        identity.
    """
    import lightning
    import linmult
    import torch

    h5_path = Path(config.data.h5_path)
    params: dict[str, Any] = {
        "env.git_sha": git_sha(),
        "env.torch": torch.__version__,
        "env.lightning": lightning.__version__,
        "env.linmult": getattr(linmult, "__version__", "unknown"),
        # The dataset hash is what actually pins a run to its data: two runs
        # with identical hyperparameters and different h5 files are not
        # comparable, and nothing else in the config would reveal that.
        "data.h5_md5": file_md5(h5_path),
        "data.h5_bytes": h5_path.stat().st_size if h5_path.is_file() else 0,
    }

    if h5_path.is_file():
        import h5py

        with h5py.File(h5_path, "r") as fh:
            params["data.h5_schema_version"] = int(fh.attrs.get("schema_version", -1))
            params["data.h5_git_sha"] = str(fh.attrs.get("git_sha", "unknown"))
            params["data.h5_created_utc"] = str(fh.attrs.get("created_utc", "unknown"))

    return params


def truncate_params(params: dict[str, Any]) -> dict[str, Any]:
    """Clip over-long values so MLflow accepts every parameter.

    Args:
        params (dict): Parameters to log.

    Returns:
        dict: The same parameters, with long values truncated.
    """
    clipped: dict[str, Any] = {}
    for key, value in params.items():
        text = str(value)
        clipped[key] = (
            text[: MAX_PARAM_LENGTH - 3] + "..." if len(text) > MAX_PARAM_LENGTH else value
        )
    return clipped


def build_logger(config: ExperimentConfig, run_name: str | None = None, run_id: str | None = None):
    """Construct the MLflow logger for a run.

    Args:
        config (ExperimentConfig): The run's configuration.
        run_name (str | None): Overrides the configured run name.
        run_id (str | None): Existing MLflow run to append to. Passed when
            resuming so the continuation logs into the same run rather than
            minting a second, disjoint one. ``None`` starts a fresh run.

    Returns:
        MLFlowLogger: The configured logger.
    """
    from lightning.pytorch.loggers import MLFlowLogger

    return MLFlowLogger(
        experiment_name=config.train.mlflow.experiment_name,
        tracking_uri=config.train.mlflow.tracking_uri,
        run_name=run_name or config.train.mlflow.run_name,
        # Reuse the prior run on --resume so its metric curves stay continuous;
        # None makes MLflow create a new run.
        run_id=run_id,
        # Checkpoints are logged deliberately at the end of a run rather than on
        # every improvement, which would copy hundreds of MB repeatedly.
        log_model=False,
    )


SKIP_ARTIFACT_DIRS: frozenset[str] = frozenset({"checkpoints", "lightning_logs"})
"""Directories excluded from the bulk artifact upload.

Checkpoints are hundreds of MB and the chosen one is logged deliberately at the
end of a run; copying every one of them into the tracking store would bloat it
for no benefit.
"""


def log_artifacts(mlflow_logger, output_dir: Path) -> None:
    """Upload a run's output directory as artifacts.

    Args:
        mlflow_logger: The run's ``MLFlowLogger``.
        output_dir (Path): Directory of run outputs.
    """
    if not output_dir.is_dir():
        return

    count = 0
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        if SKIP_ARTIFACT_DIRS & set(path.relative_to(output_dir).parts):
            continue
        # MLflow rejects "." as an artifact path, which is what relative_to
        # yields for files sitting at the root of the output directory.
        relative = path.parent.relative_to(output_dir)
        artifact_path = None if relative == Path() else str(relative)
        mlflow_logger.experiment.log_artifact(
            mlflow_logger.run_id, str(path), artifact_path=artifact_path
        )
        count += 1

    logger.info(f"Logged {count} artifacts from {output_dir} to MLflow run {mlflow_logger.run_id}")
