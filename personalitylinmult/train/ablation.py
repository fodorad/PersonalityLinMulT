"""The FI feature-ablation matrix and its run manifest.

The ablation answers one question: *which features earn their place, and how
much does each modality contribute?* It is defined here as data, not as a tree
of near-duplicate YAML files, so the whole study is one editable list and a run
is a set of overrides applied to the shared ``avt.yaml`` / model / train
configs.

**Run set** (the "minimal" study; see :data:`RUNS`):

* **7 unimodal** ``LinT`` runs — one per feature. These are the baselines every
  multimodal claim is measured against.
* **3 within-modality** ``LinMulT`` runs — all-acoustic, all-visual,
  all-textual — showing what fusing *within* a modality buys.
* **1 full** ``LinMulT`` run — every feature, the ceiling.

Text is token-level XLM-RoBERTa and mmBERT over the ground-truth transcript.

Each :class:`Run` becomes a set of ``section.key=value`` overrides via
:meth:`Run.overrides`, consumed by the training CLI exactly like a hand-typed
``--set``. Changing the study is editing :data:`RUNS`; nothing else moves.
"""

from __future__ import annotations

from dataclasses import dataclass

# Feature groupings by modality. The ablation references these rather than
# literal name lists so a feature rename or a new extractor lands in one place.
ACOUSTIC: tuple[str, ...] = ("wavlm", "emotion2vec")
"""Acoustic features, in model input order."""

VISUAL: tuple[str, ...] = ("dinov2", "emotieffnet", "farl")
"""Visual features, in model input order."""

TEXTUAL: tuple[str, ...] = (
    "xlm_roberta_gt",
    "mmbert_gt",
)
"""Textual features: XLM-RoBERTa and mmBERT token streams over the GT transcript."""

ALL_FEATURES: tuple[str, ...] = ACOUSTIC + VISUAL + TEXTUAL
"""Every feature entering the full multimodal run, in model input order."""


@dataclass(frozen=True)
class Run:
    """One ablation run: a name, its features, and its model family.

    Args:
        name (str): Stable run identifier, used as the MLflow run name and the
            key a reader cites when comparing results. Must be unique.
        features (tuple[str, ...]): Features this run consumes, in input order.
        family (str): ``lint`` (single-stream, exactly one feature) or
            ``linmult`` (multimodal, one or more).
    """

    name: str
    features: tuple[str, ...]
    family: str = "linmult"

    def __post_init__(self) -> None:
        """Validate the run is internally consistent.

        Raises:
            ValueError: If a ``lint`` run does not have exactly one feature, or
                the feature list is empty.
        """
        if not self.features:
            raise ValueError(f"Run {self.name!r} has no features.")
        if self.family == "lint" and len(self.features) != 1:
            raise ValueError(
                f"Run {self.name!r} is family='lint' but has "
                f"{len(self.features)} features; LinT is single-stream."
            )

    def overrides(self) -> dict[str, object]:
        """The ``section.key`` overrides that turn the shared configs into this run.

        Returns:
            dict: Overrides for
            :meth:`~personalitylinmult.train.config.ExperimentConfig.from_files`.
        """
        return {
            "data.features": list(self.features),
            "model.family": self.family,
            "model.name": "LinT" if self.family == "lint" else "LinMulT",
            "train.mlflow.run_name": self.name,
        }


def _unimodal_runs() -> list[Run]:
    """One single-stream ``LinT`` run per feature.

    Returns:
        list[Run]: Seven unimodal baselines.
    """
    return [Run(name=f"uni_{f}", features=(f,), family="lint") for f in ALL_FEATURES]


RUNS: list[Run] = [
    *_unimodal_runs(),
    Run(name="within_acoustic", features=ACOUSTIC),
    Run(name="within_visual", features=VISUAL),
    Run(name="within_textual", features=TEXTUAL),
    Run(name="full_avt", features=ALL_FEATURES),
]
"""The full ablation manifest: 7 unimodal + 3 within-modality + 1 full = 11 runs."""


def run_by_name(name: str) -> Run:
    """Look up a single run by its name.

    Args:
        name (str): Run identifier from :data:`RUNS`.

    Returns:
        Run: The matching run.

    Raises:
        KeyError: If no run has that name.
    """
    for run in RUNS:
        if run.name == name:
            return run
    raise KeyError(f"No ablation run named {name!r}. Known: {[r.name for r in RUNS]}.")


def validate_manifest(runs: list[Run] = RUNS) -> None:
    """Assert the manifest has unique names and only known features.

    Args:
        runs (list[Run]): Manifest to validate.

    Raises:
        ValueError: On a duplicate name or an unknown feature.
    """
    names = [r.name for r in runs]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ValueError(f"Duplicate ablation run names: {sorted(dupes)}.")
    known = set(ALL_FEATURES)
    for run in runs:
        unknown = set(run.features) - known
        if unknown:
            raise ValueError(f"Run {run.name!r} names unknown features {sorted(unknown)}.")
