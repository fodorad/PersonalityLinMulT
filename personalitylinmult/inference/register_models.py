"""Register the trained FI models into the MLflow Model Registry.

Reads the best checkpoint each training run logged (under its ``model/`` artifact
path), wraps it as a ``pyfunc`` model, and registers it under a stable name so it
can be served or containerised. Run after training; idempotent in the sense that
re-running adds a new version rather than clobbering the old one.

Example:
    Register the three audio models (LinMulT fusion + two LinT unimodal arms)::

        uv run python -m personalitylinmult.inference.register_models
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)
"""Module-level logger."""

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
"""Where the training runs were logged."""


@dataclass(frozen=True)
class Registration:
    """One model to register.

    Args:
        experiment (str): MLflow experiment holding the trained run.
        registered_name (str): Name to register under, e.g.
            ``fi_linmult_wavlm-emotion2vec``.
        run_id (str | None): Exact run to register. ``None`` falls back to the
            experiment's most recently *started* run -- convenient while
            iterating, but not a deliberate choice of "the best run", so a
            promotion decision should pin this explicitly (see
            :func:`set_champion` for the separate, deliberate promotion step).
    """

    experiment: str
    registered_name: str
    run_id: str | None = None


# Naming convention: {features joined by "_", multiword feature names use "-"}
# e.g. fi_lint_wavlm, fi_linmult_wavlm-emotion2vec. This lines up with the
# model-id scheme used for HF Hub publishing (see mlflow_pyfunc.set_champion's
# docstring and README.md's "Pretrained models" table): {features}_best-{metric}.
REGISTRATIONS: list[Registration] = [
    Registration("fi_linmult_wavlm-emotion2vec", "fi_linmult_wavlm-emotion2vec"),
    Registration("fi_lint_wavlm", "fi_lint_wavlm"),
    Registration("fi_lint_emotion2vec", "fi_lint_emotion2vec"),
    # AVT (all 7 features, LinMulT fusion) is added once a finished, evaluated
    # champion run exists and its run_id is pinned explicitly -- do not add an
    # entry that falls back to "latest run", the AVT experiment has multiple
    # in-progress/abandoned attempts (batch-size/OOM debugging) that must not
    # be registered by accident.
]
"""The audio models: LinMulT fusion (wavlm+emotion2vec) plus the two LinT
unimodal arms (wavlm only, emotion2vec only)."""


def _best_checkpoint(client, experiment: str, dest: Path, run_id: str | None = None) -> Path:
    """Download the best checkpoint logged by a run.

    Args:
        client: An ``MlflowClient``.
        experiment (str): Experiment name.
        dest (Path): Directory to download into.
        run_id (str | None): Exact run to download from. ``None`` falls back
            to the experiment's most recently *started* run, which is rarely
            the run you mean to promote -- pass a real run_id whenever the
            choice matters.

    Returns:
        Path: Local path to the downloaded ``.ckpt``.

    Raises:
        ValueError: If the experiment/run does not exist or has no logged
            checkpoint.
    """
    if run_id is None:
        exp = client.get_experiment_by_name(experiment)
        if exp is None:
            raise ValueError(f"No experiment named {experiment!r}.")
        runs = client.search_runs([exp.experiment_id], order_by=["start_time DESC"])
        if not runs:
            raise ValueError(f"Experiment {experiment!r} has no runs.")
        run_id = runs[0].info.run_id

    artifacts = client.list_artifacts(run_id, "model")
    ckpts = [a.path for a in artifacts if a.path.endswith(".ckpt")]
    if not ckpts:
        raise ValueError(f"Run {run_id} in {experiment!r} logged no .ckpt under model/.")

    local = client.download_artifacts(run_id, ckpts[0], str(dest))
    logger.info(f"{experiment}: downloaded {ckpts[0]} from run {run_id}.")
    return Path(local)


def register_all(
    registrations: list[Registration] | None = None,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> dict[str, str]:
    """Register every configured model, returning ``{name: version}``.

    Args:
        registrations (list[Registration] | None): Models to register; the
            module default when ``None``.
        tracking_uri (str): MLflow backend store.

    Returns:
        dict[str, str]: Registered name to version.
    """
    import tempfile

    import mlflow

    from personalitylinmult.inference.mlflow_pyfunc import log_and_register

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    registrations = registrations or REGISTRATIONS

    versions: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for reg in registrations:
            ckpt = _best_checkpoint(client, reg.experiment, Path(tmp), run_id=reg.run_id)
            version = log_and_register(ckpt, reg.registered_name)
            versions[reg.registered_name] = version
    return versions


def main() -> None:
    """CLI entry point.

    With no arguments, registers every model in :data:`REGISTRATIONS`. Passing
    ``--experiment``/``--registered-name`` registers exactly that one model
    instead -- the deliberate, ad hoc path for a run that is not (yet, or
    ever) part of the standing registration list. ``--alias`` additionally
    promotes the newly registered version, so a single command can both
    register and mark a version as the champion.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument(
        "--experiment", help="Register a single run instead of the standing REGISTRATIONS list."
    )
    parser.add_argument("--registered-name", help="Name to register the single run under.")
    parser.add_argument(
        "--run-id", help="Exact run to register (default: the experiment's latest)."
    )
    parser.add_argument(
        "--alias",
        help="Promote the newly registered version to this alias (e.g. 'champion').",
    )
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    if args.experiment or args.registered_name:
        if not (args.experiment and args.registered_name):
            parser.error("--experiment and --registered-name must be given together.")
        registrations = [Registration(args.experiment, args.registered_name, run_id=args.run_id)]
    else:
        registrations = None

    versions = register_all(registrations, tracking_uri=args.tracking_uri)
    for name, version in versions.items():
        logger.info(f"Registered {name} as version {version}.")
        print(f"{name}: v{version}")

    if args.alias:
        import mlflow

        from personalitylinmult.inference.mlflow_pyfunc import set_champion

        mlflow.set_tracking_uri(args.tracking_uri)
        for name, version in versions.items():
            set_champion(name, version, alias=args.alias)
            print(f"{name}@{args.alias} -> v{version}")


if __name__ == "__main__":
    main()
