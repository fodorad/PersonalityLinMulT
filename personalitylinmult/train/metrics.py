"""Metrics for the FI regression tasks.

**Naming.** The old code called ``1 - MAE`` "racc" in the training loop and
Pearson *r* "racc" in the significance experiments — one name, two different
numbers, silently incomparable across scripts. Nothing here is named ``racc``.
Every metric says what it is:

``{split}/ccc/{trait}``, ``{split}/mean_ccc``
    Lin's concordance: agreement including bias and scale, stricter than *r*.
    **The primary metric, and what drives model selection.** Chosen over
    ``1 - MAE`` because the FI trait distributions cluster tightly around
    their mean, so a model that always predicts the population mean already
    scores a deceptively good ``1 - MAE`` without having learned anything
    from the input -- CCC requires the target's variance to be reproduced,
    not just its mean approximated, so it cannot be gamed by
    regression-to-the-mean the way ``1 - MAE`` can.
``{split}/one_minus_mae/{trait}``
    Per-trait ``1 - MAE``. Reported, not selected on -- see above.
``{split}/pearson_r/{trait}``, ``{split}/mean_pearson_r``
    Linear correlation: whether the model ranks clips correctly, ignoring bias.
``{split}/r2/{trait}``, ``{split}/mean_r2``
    Variance explained, relative to predicting the trait's mean.
``{split}/std_ratio/{trait}``, ``{split}/mean_std_ratio``
    Prediction spread over target spread. Diagnoses regression-to-the-mean:
    a model can score well on ``1 - MAE`` while collapsing toward the
    training mean on every clip, and this metric is the one that catches it.

Per-trait values are computed per trait and then averaged, never pooled. The old
code's ``r2_score(targets, preds)`` over the whole ``(N, 5)`` matrix averaged
over a flattened variance, which is a different — and more flattering — number.

**Clamping.** Targets live in ``[0, 1]``. Predictions are clamped to ``[0, 1]``
for metrics, matching the old behaviour, but never inside the loss, where
clamping would zero the gradient exactly when the model is most wrong.
"""

from __future__ import annotations

from typing import Any

import torch
from torchmetrics import Metric

EPSILON = 1e-8
"""Guards division when a trait or a prediction is constant."""

METRIC_NAMES: tuple[str, ...] = ("one_minus_mae", "pearson_r", "ccc", "r2", "std_ratio")
"""Per-trait metrics computed every epoch, in report order."""

PRIMARY_METRIC = "mean_ccc"
"""The metric that selects checkpoints and drives early stopping. CCC, not
1-MAE, because 1-MAE cannot detect regression-to-the-mean (see this module's
docstring)."""


def clamp_predictions(preds: torch.Tensor) -> torch.Tensor:
    """Clamp predictions into the target range.

    Args:
        preds (torch.Tensor): Raw predictions.

    Returns:
        torch.Tensor: Predictions clamped to ``[0, 1]``.
    """
    return preds.clamp(0.0, 1.0)


def one_minus_mae(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-trait ``1 - MAE``, the task's primary metric.

    Args:
        preds (torch.Tensor): Predictions, ``(N, M)``.
        target (torch.Tensor): Targets, ``(N, M)``.

    Returns:
        torch.Tensor: ``(M,)`` per-trait scores.
    """
    return 1.0 - (preds - target).abs().mean(dim=0)


def pearson_r(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-trait Pearson correlation.

    Args:
        preds (torch.Tensor): Predictions, ``(N, M)``.
        target (torch.Tensor): Targets, ``(N, M)``.

    Returns:
        torch.Tensor: ``(M,)`` correlations; ``0`` where either side is constant
        (an undefined correlation, reported as "no linear relationship" rather
        than NaN, which would poison the epoch average).
    """
    preds_centered = preds - preds.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    numerator = (preds_centered * target_centered).sum(dim=0)
    denominator = preds_centered.pow(2).sum(dim=0).sqrt() * target_centered.pow(2).sum(dim=0).sqrt()
    return torch.where(
        denominator > EPSILON,
        numerator / (denominator + EPSILON),
        torch.zeros_like(numerator),
    )


def concordance_correlation(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-trait Lin's concordance correlation coefficient.

    Stricter than Pearson *r*: a model that ranks clips perfectly but is biased
    or mis-scaled scores high on *r* and low here.

    Args:
        preds (torch.Tensor): Predictions, ``(N, M)``.
        target (torch.Tensor): Targets, ``(N, M)``.

    Returns:
        torch.Tensor: ``(M,)`` coefficients.
    """
    preds_mean = preds.mean(dim=0)
    target_mean = target.mean(dim=0)
    preds_var = preds.var(dim=0, unbiased=False)
    target_var = target.var(dim=0, unbiased=False)
    covariance = ((preds - preds_mean) * (target - target_mean)).mean(dim=0)
    denominator = preds_var + target_var + (preds_mean - target_mean).pow(2)
    return 2 * covariance / (denominator + EPSILON)


def r2_score(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-trait coefficient of determination.

    Args:
        preds (torch.Tensor): Predictions, ``(N, M)``.
        target (torch.Tensor): Targets, ``(N, M)``.

    Returns:
        torch.Tensor: ``(M,)`` scores; ``0`` where a trait is constant.
    """
    residual = (target - preds).pow(2).sum(dim=0)
    total = (target - target.mean(dim=0, keepdim=True)).pow(2).sum(dim=0)
    return torch.where(total > EPSILON, 1.0 - residual / (total + EPSILON), torch.zeros_like(total))


def std_ratio(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-trait ratio of prediction spread to target spread.

    Diagnoses regression-to-the-mean: a model minimising MAE/MSE can hedge
    toward the training mean on uncertain clips, since a per-sample loss
    penalises confident-but-wrong predictions no more than safe-but-mediocre
    ones. That collapses prediction variance well below the target's, even
    while ``1 - MAE`` looks reasonable -- this is a known, named failure mode
    on FI specifically (the annotation is a fixed, human-rated spread; a
    collapsed-variance model is silently less useful than its MAE suggests).

    A ratio near ``1.0`` means the model's predictions are about as spread out
    as the real trait values (the honest target); well below ``1.0`` means it
    is hedging toward the mean instead of committing to extreme values.

    Args:
        preds (torch.Tensor): Predictions, ``(N, M)``.
        target (torch.Tensor): Targets, ``(N, M)``.

    Returns:
        torch.Tensor: ``(M,)`` ratios; ``0`` where the target is constant
        (avoids a division blowing up an otherwise well-behaved epoch).
    """
    pred_std = preds.std(dim=0, unbiased=False)
    target_std = target.std(dim=0, unbiased=False)
    return torch.where(
        target_std > EPSILON, pred_std / (target_std + EPSILON), torch.zeros_like(target_std)
    )


def _stack_state(state: Any, width: int) -> torch.Tensor:
    """Concatenate one torchmetrics state into a single tensor.

    ``add_state`` declares the state as a ``Tensor`` but populates a ``list`` at
    runtime, and ``dist_reduce_fx="cat"`` swaps it back to a single tensor once
    gathered across devices — so the state's static type says little about what
    is actually there, and both forms are handled here.

    Args:
        state (Any): Accumulated batches, or the already-gathered tensor.
        width (int): Number of traits, for shaping the empty case.

    Returns:
        torch.Tensor: ``(N, width)``; ``(0, width)`` when nothing is accumulated.
    """
    if not isinstance(state, list):
        return state
    if not state:
        return torch.zeros((0, width))
    return torch.cat(state, dim=0)


class TraitRegressionMetrics(Metric):
    """Accumulates an epoch of predictions and reports every trait metric.

    A :class:`torchmetrics.Metric` rather than plain lists, so state is gathered
    across devices automatically. The old code appended to Python lists on the
    module, which under DDP silently reported only rank 0's shard.

    Args:
        trait_names (list[str]): Trait names, in prediction-column order.
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self, trait_names: list[str]):
        super().__init__()
        self.trait_names = list(trait_names)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate one batch.

        Args:
            preds (torch.Tensor): Predictions, ``(B, M)``.
            target (torch.Tensor): Targets, ``(B, M)``.

        Raises:
            ValueError: If the shapes disagree or the width is not the trait
                count.
        """
        if preds.shape != target.shape:
            raise ValueError(
                f"preds shape {tuple(preds.shape)} != target shape {tuple(target.shape)}."
            )
        if preds.shape[-1] != len(self.trait_names):
            raise ValueError(f"Expected {len(self.trait_names)} traits, got {preds.shape[-1]}.")
        # `add_state(default=[])` makes these lists at runtime; a static reader
        # only sees the declared Tensor default.
        self.preds.append(clamp_predictions(preds.detach()))  # ty: ignore[unresolved-attribute, call-non-callable]
        self.target.append(target.detach())  # ty: ignore[unresolved-attribute, call-non-callable]

    def _gathered(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate accumulated state.

        An empty accumulator yields empty tensors rather than raising: callers
        that run after a reset (or before any update) should see "no
        predictions", not a crash from ``torch.cat``.

        Returns:
            tuple: ``(preds, target)``, each ``(N, M)`` — or ``(0, M)`` when
            nothing has been accumulated.
        """
        width = len(self.trait_names)
        return (
            _stack_state(self.preds, width),
            _stack_state(self.target, width),
        )

    def compute(self) -> dict[str, torch.Tensor]:
        """Compute every metric over the accumulated epoch.

        Returns:
            dict: Per-trait metrics keyed ``"<metric>/<trait>"``, plus a
            ``"mean_<metric>"`` summary for each and an overall ``"mse"``.

        Raises:
            RuntimeError: If nothing has been accumulated. Reporting metrics
                over an empty epoch would mean logging NaNs as if they were
                scores.
        """
        preds, target = self._gathered()
        if preds.shape[0] == 0:
            raise RuntimeError(
                "No predictions accumulated: compute() was called before any "
                "update(), or after a reset()."
            )

        results: dict[str, torch.Tensor] = {}
        for name, values in (
            ("one_minus_mae", one_minus_mae(preds, target)),
            ("pearson_r", pearson_r(preds, target)),
            ("ccc", concordance_correlation(preds, target)),
            ("r2", r2_score(preds, target)),
            ("std_ratio", std_ratio(preds, target)),
        ):
            for trait, value in zip(self.trait_names, values, strict=True):
                results[f"{name}/{trait}"] = value
            results[f"mean_{name}"] = values.mean()

        results["mse"] = (preds - target).pow(2).mean()
        return results

    def predictions(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the accumulated predictions and targets.

        Used to dump per-clip test predictions as a run artifact, so downstream
        significance testing never needs the model re-run.

        Returns:
            tuple: ``(preds, target)``, each ``(N, M)``.
        """
        return self._gathered()
