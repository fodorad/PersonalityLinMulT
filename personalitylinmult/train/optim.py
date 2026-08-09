"""Optimizer and learning-rate schedule construction."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from torch.optim import Adam, AdamW, Optimizer, RAdam
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    OneCycleLR,
    ReduceLROnPlateau,
    SequentialLR,
)

from personalitylinmult.train.metrics import PRIMARY_METRIC

if TYPE_CHECKING:
    from collections.abc import Iterable

    import torch

    from personalitylinmult.train.config import OptimConfig

logger = logging.getLogger(__name__)
"""Module-level logger."""

OPTIMIZER_NAMES: tuple[str, ...] = ("adam", "adamw", "radam")
"""Optimizers :func:`build_optimizer` can construct."""

MIN_ONECYCLE_STEPS = 3
"""Fewest optimizer steps a one-cycle schedule can be given.

Below this there is no room for a warmup step, a peak, and an annealing step,
and ``OneCycleLR`` divides by a zero-width phase.
"""


def build_optimizer(config: OptimConfig, parameters: Iterable[torch.nn.Parameter]) -> Optimizer:
    """Construct the configured optimizer.

    Args:
        config (OptimConfig): Optimizer settings.
        parameters (Iterable[torch.nn.Parameter]): Parameters to optimize.

    Returns:
        Optimizer: The optimizer.

    Raises:
        ValueError: If the optimizer name is unknown.
    """
    params = list(parameters)
    if config.name == "adam":
        return Adam(params, lr=config.lr, weight_decay=config.weight_decay)
    if config.name == "adamw":
        return AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    if config.name == "radam":
        return RAdam(params, lr=config.lr, weight_decay=config.weight_decay)
    raise ValueError(f"Unknown optimizer {config.name!r}; expected {sorted(OPTIMIZER_NAMES)}.")


def build_scheduler(
    config: OptimConfig,
    optimizer: Optimizer,
    steps_per_epoch: int,
    max_epochs: int,
) -> dict[str, Any] | None:
    """Construct the configured LR schedule, in Lightning's config format.

    Args:
        config (OptimConfig): Optimizer settings.
        optimizer (Optimizer): The optimizer to schedule.
        steps_per_epoch (int): Optimizer steps in one epoch.
        max_epochs (int): Total epochs.

    Returns:
        dict | None: Lightning ``lr_scheduler_config``, or ``None`` when no
        schedule is configured.

    Raises:
        ValueError: If the scheduler name is unknown.
    """
    if config.scheduler == "none":
        return None

    total_steps = max(1, steps_per_epoch * max_epochs)

    if config.scheduler == "onecycle":
        if total_steps < MIN_ONECYCLE_STEPS:
            logger.warning(
                f"onecycle needs at least {MIN_ONECYCLE_STEPS} optimizer steps but this "
                f"run has {total_steps}; falling back to a constant learning rate."
            )
            return None

        # OneCycle warms up internally; layering an external warmup on top would
        # double-count it.
        scheduler: LRScheduler = OneCycleLR(
            optimizer,
            max_lr=config.lr,
            total_steps=total_steps,
            pct_start=_safe_pct_start(config.warmup_ratio, total_steps),
        )
        return {"scheduler": scheduler, "interval": "step", "frequency": 1}

    if config.scheduler == "cosine":
        # The cosine's period must cover only the steps that remain after
        # warmup. Giving it the full run while a warmup is chained in front
        # makes it complete its half-cycle early and sit at lr=0 for the tail.
        warmup_steps = int(total_steps * config.warmup_ratio)
        cosine = CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=config.lr * 0.01
        )
        return {
            "scheduler": _with_warmup(config, optimizer, cosine, total_steps),
            "interval": "step",
            "frequency": 1,
        }

    if config.scheduler == "plateau":
        return {
            "scheduler": ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=max(1, max_epochs // 10)
            ),
            "interval": "epoch",
            "frequency": 1,
            "monitor": f"valid/{PRIMARY_METRIC}",
        }

    raise ValueError(f"Unknown scheduler {config.scheduler!r}.")


def _safe_pct_start(warmup_ratio: float, total_steps: int) -> float:
    """Nudge ``pct_start`` off the value that makes ``OneCycleLR`` divide by zero.

    ``OneCycleLR`` places its first phase boundary at
    ``pct_start * total_steps - 1`` and later divides by that phase's width. When
    ``pct_start * total_steps`` is exactly ``1`` the boundary lands on step 0,
    the width is zero, and the scheduler raises ``ZeroDivisionError`` on
    construction. This bites real configurations — the default ``warmup_ratio``
    of 0.1 over exactly 10 optimizer steps is enough to trigger it.

    Args:
        warmup_ratio (float): Configured warmup fraction.
        total_steps (int): Total optimizer steps in the run.

    Returns:
        float: A ``pct_start`` that keeps both phases non-degenerate.
    """
    pct = warmup_ratio
    if abs(pct * total_steps - 1.0) < 1e-9:
        pct = 1.5 / total_steps
        logger.debug(
            f"nudged onecycle pct_start to {pct:.4f} to keep its warmup phase non-empty "
            f"at {total_steps} steps"
        )
    # Keep at least one step either side of the peak.
    return float(min(max(pct, 1.5 / total_steps), 1.0 - 1.0 / total_steps))


def _with_warmup(
    config: OptimConfig,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    total_steps: int,
) -> LRScheduler:
    """Prefix a schedule with a linear warmup, if one is configured.

    Args:
        config (OptimConfig): Optimizer settings.
        optimizer (Optimizer): The optimizer.
        scheduler (LRScheduler): The main schedule.
        total_steps (int): Total optimizer steps.

    Returns:
        LRScheduler: The main schedule, or a warmup chained into it.
    """
    warmup_steps = int(total_steps * config.warmup_ratio)
    if warmup_steps < 1:
        return scheduler

    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    return SequentialLR(optimizer, schedulers=[warmup, scheduler], milestones=[warmup_steps])
