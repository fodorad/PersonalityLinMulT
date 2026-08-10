"""Drive the FI feature-ablation matrix defined in :mod:`.ablation`.

Each run in :data:`~personalitylinmult.train.ablation.RUNS` is the shared
``avt`` / train configs, plus the ``lint.yaml`` or ``linmult.yaml`` model
config matching that run's architecture, with the run's overrides applied,
trained and tested through the same :func:`~personalitylinmult.train.cli.run`
used for a single experiment. Nothing about a run is special-cased here: an
ablation run is a normal run whose ``data.features``, ``model.family``, and
MLflow run name come from the manifest instead of the command line.

Examples:
    Smoke-test every run (one batch each), the pre-flight before an overnight
    sweep::

        make ablation-smoke

    Run the whole sweep for real::

        make ablation

    Run a single named arm::

        uv run python -m personalitylinmult.train.run_ablation --only full_avt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from personalitylinmult.train.ablation import RUNS, Run, run_by_name, validate_manifest
from personalitylinmult.train.cli import run
from personalitylinmult.train.config import ExperimentConfig
from personalitylinmult.train.metrics import PRIMARY_METRIC

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)
"""Module-level logger."""

SUMMARY_METRIC = f"test/{PRIMARY_METRIC}"
"""Test metric summarised per run. Derived from the shared ``PRIMARY_METRIC`` so
a rename cannot silently turn the summary into a column of ``None``."""

CONFIG_ROOT = Path("config/FI")
"""Root of the FI config tree the ablation composes its runs from."""

DEFAULT_DATA = CONFIG_ROOT / "data" / "avt.yaml"
"""Base data config; the ablation overrides only its feature list."""

DEFAULT_MODEL_BY_FAMILY: dict[str, Path] = {
    "lint": CONFIG_ROOT / "model" / "lint.yaml",
    "linmult": CONFIG_ROOT / "model" / "linmult.yaml",
}
"""Base model config per architecture -- each run picks its own by
``run_spec.family``, since ``lint.yaml`` and ``linmult.yaml`` are separate,
self-contained files (one per architecture, not a shared file toggled via a
``model.family`` override)."""

DEFAULT_TRAIN = CONFIG_ROOT / "train" / "bigfive.yaml"
"""Base train config; the ablation overrides only the MLflow run name."""


def build_config(
    run_spec: Run,
    data_path: Path,
    model_path: Path | None,
    train_path: Path,
) -> ExperimentConfig:
    """Compose one ablation run's :class:`ExperimentConfig`.

    Args:
        run_spec (Run): The run from the manifest.
        data_path (Path): Base data config.
        model_path (Path | None): Base model config. ``None`` (the default)
            picks ``lint.yaml`` or ``linmult.yaml`` from
            :data:`DEFAULT_MODEL_BY_FAMILY` to match ``run_spec.family``; pass
            an explicit path to force every run onto the same file.
        train_path (Path): Base train config.

    Returns:
        ExperimentConfig: The run's fully resolved configuration.
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_BY_FAMILY[run_spec.family]
    return ExperimentConfig.from_files(
        data_path, model_path, train_path, overrides=run_spec.overrides()
    )


def select_runs(only: list[str]) -> list[Run]:
    """Resolve the ``--only`` names to runs, or the whole matrix when empty.

    Validates the manifest first, so a duplicate name or unknown feature is
    caught before any run starts.

    Args:
        only (list[str]): Run names to keep; empty means the whole matrix.

    Returns:
        list[Run]: The selected runs, in manifest order for the full matrix or
        ``--only`` order for a subset.

    Raises:
        KeyError: If an ``--only`` name is not in the manifest.
        ValueError: If the manifest is inconsistent.
    """
    validate_manifest()
    return [run_by_name(name) for name in only] if only else RUNS


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Base data config.")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Base model config for every run. Default: pick lint.yaml or "
        "linmult.yaml per run's architecture.",
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN, help="Base train config.")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="RUN_NAME",
        help="Run only these named arms (repeatable). Default: the whole matrix.",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run one batch through each arm and exit — the sweep pre-flight.",
    )
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    runs = select_runs(args.only)

    def run_one(run_spec: Run) -> float | None:
        config = build_config(run_spec, args.data, args.model, args.train)
        metrics = run(config, fast_dev_run=args.fast_dev_run)
        return metrics.get(SUMMARY_METRIC)

    run_sweep(runs, run_one, summarise=not args.fast_dev_run)


def run_sweep(
    runs: list[Run],
    run_one: Callable[[Run], float | None],
    summarise: bool = True,
) -> dict[str, float | str | None]:
    """Run each arm in isolation, recording results and surviving failures.

    A run that OOMs or hits a transient CUDA error must not take the rest of an
    overnight sweep down with it: each is wrapped so a failure is recorded and
    the loop continues, and the summary prints in ``finally`` so partial results
    always survive whatever killed an individual run.

    Args:
        runs (list[Run]): Arms to run, in order.
        run_one (Callable[[Run], float | None]): Runs one arm and returns its
            summary score. Any exception it raises is caught and recorded.
        summarise (bool): Log the per-run summary at the end. Disabled for
            ``fast_dev_run`` smoke passes, which produce no meaningful scores.

    Returns:
        dict: Per-run summary score, or a ``"FAILED: <ExcType>"`` marker string
        for a run that raised.
    """
    logger.info(f"Ablation: {len(runs)} run(s): {[r.name for r in runs]}")
    results: dict[str, float | str | None] = {}
    try:
        for i, run_spec in enumerate(runs, start=1):
            logger.info(f"[{i}/{len(runs)}] {run_spec.name}: features={list(run_spec.features)}")
            try:
                results[run_spec.name] = run_one(run_spec)
            except Exception as exc:  # noqa: BLE001 -- one bad run must not abort the sweep
                logger.exception(f"{run_spec.name} FAILED; continuing with the rest.")
                results[run_spec.name] = f"FAILED: {type(exc).__name__}"
    finally:
        if summarise:
            logger.info(f"Ablation summary ({SUMMARY_METRIC} per run):")
            for name, score in results.items():
                logger.info(f"  {name:18s} {score}")
            failed = [n for n, s in results.items() if isinstance(s, str)]
            if failed:
                logger.warning(f"{len(failed)} run(s) failed: {failed}")
    return results


if __name__ == "__main__":
    main()
