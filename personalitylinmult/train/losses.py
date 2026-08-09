"""Regression losses for the FI tasks.

:func:`bell_l2_l1_loss` is reimplemented here rather than imported from
``exordium``, which keeps training installable without the full
``exordium[video,text]`` extraction stack (whisperx, torchcodec, opencv, …).
The behaviour is identical to the original; the tests pin it against
hand-computed values so the port cannot drift silently.

:func:`ccc_loss` deliberately duplicates the elementwise math of
:func:`~personalitylinmult.train.metrics.concordance_correlation` rather than
importing it, so this module stays free of a cross-import into ``metrics``
(which the rest of the training loop imports the other way around) and keeps
its own test coverage self-contained. The two are pinned equal by test.
"""

from __future__ import annotations

import torch
from torch import nn

BELL_AMPLITUDE = 300.0
"""Bell-term weight, from the original formulation."""

BELL_SCALE = 2 * 9**2
"""Bell-term width (``2 * 9^2 = 162``), from the original formulation.

Targets are in ``[0, 1]``, so squared error never exceeds 1 and the exponential
operates near its origin, where ``1 - exp(-x/162)`` is close to ``x/162``. The
bell term therefore behaves as a smooth, bounded stand-in for squared error —
saturating for the gross errors that would otherwise dominate the gradient.
"""

CCC_EPSILON = 1e-8
"""Guards division when a trait or a batch's predictions are constant."""


def bell_l2_l1_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Combined bell-shaped, squared, and absolute error.

    Args:
        y_pred (torch.Tensor): Predictions, ``(N, M)``.
        y_true (torch.Tensor): Targets, ``(N, M)``.

    Returns:
        torch.Tensor: Scalar loss.
    """
    diff = y_true - y_pred
    sqr_error = torch.pow(diff, 2)
    abs_error = torch.abs(diff)
    bell_loss = BELL_AMPLITUDE * (1 - torch.exp(-(sqr_error / BELL_SCALE)))
    return torch.mean(bell_loss) + torch.mean(sqr_error) + torch.mean(abs_error)


class BellL2L1Loss(nn.Module):
    """:func:`bell_l2_l1_loss` as a module, for use as a configured loss."""

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Args:
            y_pred (torch.Tensor): Predictions, ``(N, M)``.
            y_true (torch.Tensor): Targets, ``(N, M)``.

        Returns:
            torch.Tensor: Scalar loss.
        """
        return bell_l2_l1_loss(y_pred, y_true)


def ccc_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """``1 - CCC``, averaged over traits: a differentiable concordance loss.

    Concordance (Lin's CCC) is stricter than Pearson *r*: it penalises both
    poor ranking (like *r*) and any bias or scale mismatch (unlike *r*), which
    is exactly what "predict the trait well" should mean for FI's averaged
    annotator ratings. It is already this project's evaluation metric (see
    :func:`~personalitylinmult.train.metrics.concordance_correlation`) but was
    never the training signal -- optimising ``1 - MAE``-shaped losses and
    *measuring* CCC is a real mismatch this loss removes.

    Computed per training batch (mean/var/covariance over the batch axis),
    unlike the epoch-accumulated CCC *metric* -- the standard way CCC-loss is
    used in the affective-computing literature (e.g. AVEC baselines), at the
    cost of a noisier per-batch estimate than the epoch-level metric gives.

    Args:
        y_pred (torch.Tensor): Predictions, ``(N, M)``.
        y_true (torch.Tensor): Targets, ``(N, M)``.

    Returns:
        torch.Tensor: Scalar loss in roughly ``[0, 2]`` (0 = perfect
        concordance on every trait).
    """
    pred_mean = y_pred.mean(dim=0)
    true_mean = y_true.mean(dim=0)
    pred_var = y_pred.var(dim=0, unbiased=False)
    true_var = y_true.var(dim=0, unbiased=False)
    covariance = ((y_pred - pred_mean) * (y_true - true_mean)).mean(dim=0)
    denominator = pred_var + true_var + (pred_mean - true_mean).pow(2)
    ccc = 2 * covariance / (denominator + CCC_EPSILON)
    return torch.mean(1.0 - ccc)


class CCCLoss(nn.Module):
    """:func:`ccc_loss` as a module, for use as a configured loss."""

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Args:
            y_pred (torch.Tensor): Predictions, ``(N, M)``.
            y_true (torch.Tensor): Targets, ``(N, M)``.

        Returns:
            torch.Tensor: Scalar loss.
        """
        return ccc_loss(y_pred, y_true)


class UncertaintyWeightedLoss(nn.Module):
    """Learned-weight combination of two losses (Kendall, Gal & Cipolla, 2018).

    CCC loss alone preserves prediction variance far better than an elementwise
    loss (see the std_ratio metric) but is noisy per-batch and prone to
    overfitting once its early-epoch peak passes; a fixed hand-picked blend of
    CCC and an elementwise loss (e.g. ``0.5 * ccc + 0.5 * l1``) would trade one
    guess (the loss function) for another (the blend weight). This module
    removes that second guess: each loss term gets its own learnable
    ``log_sigma``, updated by backprop alongside the model, so the optimal
    blend is discovered rather than hand-picked.

    ``L = sum_i [ 0.5 * exp(-2 * log_sigma_i) * L_i + log_sigma_i ]``

    The ``log_sigma`` parameterisation (rather than a raw learnable weight)
    keeps the implied precision ``exp(-2*log_sigma)`` positive for any
    unconstrained parameter value, and the ``+ log_sigma_i`` regulariser stops
    the trivial collapse of driving every weight to zero: a term the model
    currently fits poorly gets ``log_sigma_i`` pushed up (down-weighting it),
    but only as far as the regulariser allows.

    This is the right tool for combining two loss *terms on one task* (unclear
    relative scale, not genuinely conflicting objectives). Techniques such as
    GradNorm, PCGrad, or MGDA solve a different problem -- balancing gradients
    across genuinely separate task heads -- and are not needed here.

    Args:
        losses (list[nn.Module]): The loss terms to combine.
    """

    def __init__(self, losses: list[nn.Module]):
        super().__init__()
        if len(losses) < 2:
            raise ValueError(f"Need at least 2 losses to combine, got {len(losses)}.")
        self.losses = nn.ModuleList(losses)
        self.log_sigma = nn.Parameter(torch.zeros(len(losses)))

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute the uncertainty-weighted sum of the wrapped losses.

        Args:
            y_pred (torch.Tensor): Predictions, ``(N, M)``.
            y_true (torch.Tensor): Targets, ``(N, M)``.

        Returns:
            torch.Tensor: Scalar combined loss.
        """
        total = torch.zeros((), device=y_pred.device, dtype=y_pred.dtype)
        for loss_fn, log_sigma in zip(self.losses, self.log_sigma, strict=True):
            precision = torch.exp(-2.0 * log_sigma)
            total = total + 0.5 * precision * loss_fn(y_pred, y_true) + log_sigma
        return total


def build_loss(name: str) -> nn.Module:
    """Construct a loss by config name.

    Args:
        name (str): One of ``"bell_l2_l1"``, ``"l1"``, ``"mse"``, ``"huber"``,
            ``"ccc"``, ``"ccc+l1"``. The last combines CCC and L1 with
            learned, per-term weights -- see :class:`UncertaintyWeightedLoss`.

    Returns:
        nn.Module: The loss module. For ``"ccc+l1"`` this module owns learnable
        parameters and must be registered as a submodule of the training
        module (not called as a bare function) so its weights are optimised
        jointly with the model.

    Raises:
        ValueError: If the name is unknown.
    """
    losses: dict[str, type[nn.Module]] = {
        "bell_l2_l1": BellL2L1Loss,
        "l1": nn.L1Loss,
        "mse": nn.MSELoss,
        "huber": nn.HuberLoss,
        "ccc": CCCLoss,
    }
    if name == "ccc+l1":
        return UncertaintyWeightedLoss([CCCLoss(), nn.L1Loss()])
    if name not in losses:
        raise ValueError(f"Unknown loss {name!r}; expected one of {sorted([*losses, 'ccc+l1'])}.")
    return losses[name]()
