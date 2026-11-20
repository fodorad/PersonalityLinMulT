"""Streaming closed-form ridge regression for linear probing.

Sufficient statistics (``XᵀX``, ``XᵀY``, sums) are accumulated chunk by chunk in
float64, so frame-level probes never need the full design matrix in memory.
Inputs are standardised with the accumulated train mean/std; the intercept
comes from centring. Evaluation is streaming too and reports per-output R² and
Pearson r plus the variance-weighted R².

Typical use::

    acc = RidgeAccumulator(d_in, d_out)
    for x, y in train_chunks:
        acc.update(x, y)
    probe, alpha = select_alpha(acc, ALPHA_GRID, valid_chunks)
    metrics = evaluate(probe, test_chunks)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

ALPHA_GRID: tuple[float, ...] = tuple(float(a) for a in torch.logspace(-2, 4, 13).tolist())
"""Default ridge penalties: log grid 1e-2 ... 1e4."""

STD_FLOOR: float = 1e-8
"""Floor on input std so constant input dimensions do not divide by zero."""


@dataclass
class RidgeProbe:
    """Fitted linear map on standardised inputs.

    Attributes:
        weight: Weights, shape ``(d_in, d_out)``, in standardised-input space.
        x_mean: Train input mean, shape ``(d_in,)``.
        x_std: Train input std (floored), shape ``(d_in,)``.
        y_mean: Train output mean (the intercept), shape ``(d_out,)``.
    """

    weight: Tensor
    x_mean: Tensor
    x_std: Tensor
    y_mean: Tensor

    def predict(self, x: Tensor) -> Tensor:
        """Predict outputs.

        Args:
            x: Inputs, shape ``(N, d_in)``.

        Returns:
            Float64 predictions, shape ``(N, d_out)``.
        """
        z = (x.double() - self.x_mean) / self.x_std
        return z @ self.weight + self.y_mean


@dataclass
class ProbeMetrics:
    """Probe quality on one split.

    Attributes:
        r2: Per-output R² (1 - SSres/SStot over all frames), shape ``(d_out,)``.
        pearson_r: Per-output Pearson r, shape ``(d_out,)``.
        mean_r2: Unweighted mean of ``r2``.
        mean_pearson_r: Unweighted mean of ``pearson_r``.
        weighted_r2: ``1 - sum(SSres) / sum(SStot)``, i.e. variance-weighted R².
    """

    r2: Tensor
    pearson_r: Tensor
    mean_r2: float
    mean_pearson_r: float
    weighted_r2: float


class RidgeAccumulator:
    """Accumulates sufficient statistics for ridge regression in float64.

    Args:
        d_in: Input dimension.
        d_out: Output dimension.
    """

    def __init__(self, d_in: int, d_out: int) -> None:
        self.n = 0
        self.sum_x = torch.zeros(d_in, dtype=torch.float64)
        self.sum_y = torch.zeros(d_out, dtype=torch.float64)
        self.xtx = torch.zeros(d_in, d_in, dtype=torch.float64)
        self.xty = torch.zeros(d_in, d_out, dtype=torch.float64)

    def update(self, x: Tensor, y: Tensor) -> None:
        """Add a chunk of frames.

        Args:
            x: Inputs, shape ``(N, d_in)``.
            y: Targets, shape ``(N, d_out)``.

        Raises:
            ValueError: If shapes do not match the accumulator.
        """
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError(
                f"expected (N, d_in) and (N, d_out), got {tuple(x.shape)}, {tuple(y.shape)}"
            )
        if x.shape[1] != self.sum_x.numel() or y.shape[1] != self.sum_y.numel():
            raise ValueError("chunk dimensions do not match the accumulator")
        x, y = x.double(), y.double()
        self.n += x.shape[0]
        self.sum_x += x.sum(0)
        self.sum_y += y.sum(0)
        self.xtx += x.T @ x
        self.xty += x.T @ y

    def solve(self, alphas: Iterable[float]) -> dict[float, RidgeProbe]:
        """Solve the ridge problem for several penalties at once.

        One eigendecomposition of the standardised Gram matrix serves all alphas.

        Args:
            alphas: Non-negative ridge penalties.

        Returns:
            Mapping ``alpha -> RidgeProbe``.

        Raises:
            ValueError: If no frames were accumulated.
        """
        if self.n == 0:
            raise ValueError("no frames accumulated")
        x_mean = self.sum_x / self.n
        y_mean = self.sum_y / self.n
        cov = self.xtx - self.n * torch.outer(x_mean, x_mean)
        x_std = (cov.diagonal() / self.n).clamp_min(0).sqrt().clamp_min(STD_FLOOR)
        gram = cov / torch.outer(x_std, x_std)
        cross = (self.xty - self.n * torch.outer(x_mean, y_mean)) / x_std[:, None]
        eigvals, eigvecs = torch.linalg.eigh(gram)
        projected = eigvecs.T @ cross
        probes = {}
        for alpha in alphas:
            weight = eigvecs @ (projected / (eigvals.clamp_min(0) + alpha)[:, None])
            probes[alpha] = RidgeProbe(weight, x_mean, x_std, y_mean)
        return probes


class _MetricsAccumulator:
    """Streaming sums for R² and Pearson r."""

    def __init__(self, d_out: int) -> None:
        zeros = lambda: torch.zeros(d_out, dtype=torch.float64)  # noqa: E731
        self.n = 0
        self.sum_y, self.sum_yy = zeros(), zeros()
        self.sum_p, self.sum_pp, self.sum_py = zeros(), zeros(), zeros()
        self.sse = zeros()

    def update(self, pred: Tensor, y: Tensor) -> None:
        y = y.double()
        self.n += y.shape[0]
        self.sum_y += y.sum(0)
        self.sum_yy += (y * y).sum(0)
        self.sum_p += pred.sum(0)
        self.sum_pp += (pred * pred).sum(0)
        self.sum_py += (pred * y).sum(0)
        self.sse += ((y - pred) ** 2).sum(0)

    def result(self) -> ProbeMetrics:
        if self.n == 0:
            raise ValueError("no frames to evaluate")
        sstot = self.sum_yy - self.sum_y**2 / self.n
        var_p = self.sum_pp - self.sum_p**2 / self.n
        cov = self.sum_py - self.sum_p * self.sum_y / self.n
        r2 = torch.where(sstot > 0, 1 - self.sse / sstot.clamp_min(1e-300), torch.zeros_like(sstot))
        denom = (var_p * sstot).clamp_min(0).sqrt()
        pearson = torch.where(denom > 0, cov / denom.clamp_min(1e-300), torch.zeros_like(denom))
        total = sstot.sum()
        weighted = 1 - self.sse.sum() / total if total > 0 else torch.zeros(())
        return ProbeMetrics(
            r2=r2,
            pearson_r=pearson,
            mean_r2=float(r2.mean()),
            mean_pearson_r=float(pearson.mean()),
            weighted_r2=float(weighted),
        )


def evaluate(probe: RidgeProbe, chunks: Iterable[tuple[Tensor, Tensor]]) -> ProbeMetrics:
    """Evaluate a probe on a stream of ``(x, y)`` chunks.

    Outputs with constant targets get R² 0 and Pearson r 0 (finite).

    Args:
        probe: Fitted probe.
        chunks: Iterable of ``(x, y)`` with shapes ``(N, d_in)``, ``(N, d_out)``.

    Returns:
        Metrics over all frames of all chunks.
    """
    acc = _MetricsAccumulator(probe.y_mean.numel())
    for x, y in chunks:
        acc.update(probe.predict(x), y)
    return acc.result()


def select_alpha(
    acc: RidgeAccumulator,
    alphas: Iterable[float],
    valid_chunks: Iterable[tuple[Tensor, Tensor]],
) -> tuple[RidgeProbe, float]:
    """Pick the alpha with the best mean R² on validation chunks.

    Args:
        acc: Train statistics.
        alphas: Candidate penalties.
        valid_chunks: Validation ``(x, y)`` chunks (materialised once).

    Returns:
        ``(probe, alpha)`` for the best alpha; ties go to the larger alpha.
    """
    valid = list(valid_chunks)
    probes = acc.solve(alphas)
    best_alpha = max(sorted(probes), key=lambda a: (evaluate(probes[a], valid).mean_r2, a))
    logger.info("selected ridge alpha=%g", best_alpha)
    return probes[best_alpha], best_alpha
