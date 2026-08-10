"""Hyperparameter search for an FI training run, via Optuna.

Uses the TPE (Tree-structured Parzen Estimator) sampler — a Bayesian-family
method that fits two density estimators over "good" vs. "other" trials and
samples where their ratio is highest — combined with an ASHA-style
(Hyperband) pruner that kills clearly-bad trials early from their partial
learning curve. This is the current default combination for small (<10
dimension) search spaces: it converges faster than grid search (which wastes
trials on unimportant dimensions) without the per-trial overhead of a full
Gaussian-process Bayesian search, which only pays off above ~10-15 dimensions.

Each trial is a normal training run (:func:`~personalitylinmult.train.cli.run`)
under a temporary MLflow experiment, with a pruning callback reporting
``valid/mean_ccc`` (see :data:`personalitylinmult.train.metrics.PRIMARY_METRIC`)
after every epoch so Optuna can stop a trial that is clearly behind its peers
partway through training.

Example:
    Search ``fi_lint_wavlm``'s hyperparameters, 30 trials::

        uv run python -m personalitylinmult.train.hpo \
            --data config/FI/data/wavlm.yaml \
            --model config/FI/model/lint.yaml \
            --train config/FI/train/bigfive.yaml \
            --n-trials 30
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from optuna_integration import PyTorchLightningPruningCallback

from personalitylinmult.train.cli import MONITOR, run
from personalitylinmult.train.config import ExperimentConfig, load_yaml, parse_override

logger = logging.getLogger(__name__)
"""Module-level logger."""

DEFAULT_N_TRIALS = 30
"""Trial budget: enough for TPE to characterise a ~5-dimension space without
the combinatorial blowup a grid search over the same ranges would need."""

DEFAULT_MIN_RESOURCE_EPOCHS = 5
"""Epochs a trial always gets before Hyperband is allowed to prune it -- long
enough to tell a healthy run from a diverging one, short enough that pruning
still saves real time."""


def default_search_space(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest one trial's hyperparameters as ``ExperimentConfig`` overrides.

    The space is narrowed to the dimensions with real leverage, diagnosed from
    the ``fi_lint_wavlm`` baseline run: peak LR and weight decay (both
    log-scale, since good values span orders of magnitude), and the dropout
    profile -- ``dropout_attention`` was 0.0 in the baseline while every other
    dropout was already nonzero, an inconsistency worth searching rather than
    the whole 40-parameter surface.

    Args:
        trial (optuna.Trial): The trial to suggest values for.

    Returns:
        dict[str, Any]: Dotted ``section.key`` overrides for
        :meth:`~personalitylinmult.train.config.ExperimentConfig.from_files`.
    """
    return {
        "train.optimizer.lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "train.optimizer.weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
        "train.optimizer.warmup_ratio": trial.suggest_float("warmup_ratio", 0.05, 0.2),
        "model.dropout_attention": trial.suggest_float("dropout_attention", 0.0, 0.3),
        "model.dropout_ffn": trial.suggest_float("dropout_ffn", 0.1, 0.4),
        "model.head_dropout": trial.suggest_float("head_dropout", 0.1, 0.4),
    }


def build_objective(
    data_path: Path,
    model_path: Path,
    train_path: Path,
    base_overrides: dict[str, Any],
    study_name: str,
    search_space: Any = default_search_space,
):
    """Build an Optuna objective that trains one trial and returns its score.

    Args:
        data_path (Path): Data config shared by every trial.
        model_path (Path): Model config shared by every trial.
        train_path (Path): Train config shared by every trial.
        base_overrides (dict[str, Any]): Overrides applied to every trial
            before the search space's (e.g. ``data.window_seconds=10``).
        study_name (str): Identifies this search; used as the MLflow experiment
            name (unless ``base_overrides`` already sets one) so each search
            gets its own experiment and checkpoint directory. Using the shared
            train config's filename here would collide: two different searches
            (e.g. over wavlm vs. emotion2vec) would land in the same
            experiment and overwrite each other's checkpoints.
        search_space (Callable[[optuna.Trial], dict]): Suggests one trial's
            hyperparameter overrides. Defaults to :func:`default_search_space`.

    Returns:
        Callable[[optuna.Trial], float]: An Optuna objective maximising
        ``test/mean_ccc``.
    """

    def objective(trial: optuna.Trial) -> float:
        overrides = {**base_overrides, **search_space(trial)}
        overrides.setdefault("train.mlflow.experiment_name", study_name)
        overrides["train.mlflow.run_name"] = f"trial-{trial.number}"
        config = ExperimentConfig.from_files(data_path, model_path, train_path, overrides)

        pruning_callback = PyTorchLightningPruningCallback(trial, monitor=MONITOR)
        metrics = run(config, extra_callbacks=[pruning_callback])
        return float(metrics.get(f"test/{MONITOR.split('/', 1)[1]}", 0.0))

    return objective


def default_storage_path(train_path: Path, study_name: str) -> Path:
    """The default Optuna storage file for a study, under its results dir.

    Matches :func:`~personalitylinmult.train.cli.run`'s own layout
    (``{output_dir}/{task}/{experiment_name}/``): the study's SQLite file lives
    inside its own ``results/FI/bigfive/<study_name>/`` directory alongside its
    trial checkpoints, rather than a loose file at the repo root.

    Args:
        train_path (Path): Train config the search runs under (read for
            ``output_dir`` / ``task``).
        study_name (str): The study's name, also its results subdirectory.

    Returns:
        Path: ``{output_dir}/{task}/{study_name}/optuna.db``.
    """
    train_yaml = load_yaml(train_path)
    output_dir = Path(train_yaml.get("output_dir", "results/FI"))
    task = train_yaml.get("task", "bigfive")
    return output_dir / task / study_name / "optuna.db"


def run_search(
    data_path: Path,
    model_path: Path,
    train_path: Path,
    base_overrides: dict[str, Any],
    n_trials: int = DEFAULT_N_TRIALS,
    min_resource_epochs: int = DEFAULT_MIN_RESOURCE_EPOCHS,
    study_name: str = "fi-hpo",
    storage: str | None = None,
) -> optuna.Study:
    """Run a TPE + Hyperband (ASHA-style) hyperparameter search.

    Args:
        data_path (Path): Data config shared by every trial.
        model_path (Path): Model config shared by every trial.
        train_path (Path): Train config shared by every trial.
        base_overrides (dict[str, Any]): Overrides applied to every trial
            before the search space's.
        n_trials (int): Number of trials to run.
        min_resource_epochs (int): Epochs guaranteed before a trial can be
            pruned.
        study_name (str): Optuna study name -- also used as the MLflow
            experiment name (see :func:`build_objective`), so pick one specific
            to what is being searched (e.g. ``fi-lint-wavlm-hpo``), not a
            generic name shared across different searches.
        storage (str | None): Optuna storage URL. ``None`` (the default)
            persists to :func:`default_storage_path` -- a resumable SQLite file
            inside the study's own results directory -- rather than an
            in-memory study that loses all progress if the process dies.

    Returns:
        optuna.Study: The completed study; ``study.best_params`` and
        ``study.best_value`` hold the winning trial.
    """
    if storage is None:
        db_path = default_storage_path(train_path, study_name)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{db_path}"

    sampler = TPESampler(seed=base_overrides.get("train.seed", 42))
    pruner = HyperbandPruner(min_resource=min_resource_epochs, reduction_factor=3)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    objective = build_objective(data_path, model_path, train_path, base_overrides, study_name)
    study.optimize(objective, n_trials=n_trials)

    logger.info(f"Best trial: value={study.best_value:.4f} params={study.best_params}")
    return study


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
        help="Base override applied to every trial, e.g. --set data.window_seconds=10.",
    )
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--min-resource-epochs", type=int, default=DEFAULT_MIN_RESOURCE_EPOCHS)
    parser.add_argument("--study-name", default="fi-hpo")
    parser.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL, e.g. sqlite:///optuna.db. Default: a resumable "
        "SQLite file under the study's own results/FI/<task>/<study-name>/ dir.",
    )
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    base_overrides = dict(parse_override(text) for text in args.overrides)
    study = run_search(
        args.data,
        args.model,
        args.train,
        base_overrides,
        n_trials=args.n_trials,
        min_resource_epochs=args.min_resource_epochs,
        study_name=args.study_name,
        storage=args.storage,
    )
    print(f"best value: {study.best_value:.4f}")
    print(f"best params: {study.best_params}")


if __name__ == "__main__":
    main()
