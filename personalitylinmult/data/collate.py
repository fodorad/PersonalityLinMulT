"""Batching for FI samples, producing the model's feature+mask input pairs.

The dataset already fixes every feature's time dimension, so collation is a
stack rather than a pad. What this module owns is the shape of the model call:
:func:`unpack_batch` turns a batch dict into the ``(inputs, masks, names)``
triple LinMulT expects, keeping the mask contract (``True`` = valid) in one
place instead of spread through the training step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence


def collate_fi(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate FI samples into a batch.

    Args:
        samples (list[dict]): Per-clip dicts from
            :class:`~personalitylinmult.data.dataset.FiHdf5Dataset`.

    Returns:
        dict: Every tensor key stacked to a leading batch axis; ``video_id``
        collected into a list.

    Raises:
        ValueError: If the batch is empty.
    """
    if not samples:
        raise ValueError("Cannot collate an empty batch.")

    batch: dict[str, Any] = {"video_id": [s["video_id"] for s in samples]}
    for key, value in samples[0].items():
        if isinstance(value, torch.Tensor):
            batch[key] = torch.stack([s[key] for s in samples], dim=0)
    return batch


def unpack_batch(
    batch: dict[str, Any], feature_names: Sequence[str]
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[str]]:
    """Split a batch into the ``(inputs, masks, names)`` LinMulT expects.

    Every feature is passed with its mask. LinMulT drops all-``True`` masks
    internally, so a dense feature costs one ``.all()`` check and nothing more —
    which is why the contract stays uniform rather than special-casing which
    features "need" a mask.

    Args:
        batch (dict): A collated batch.
        feature_names (Sequence[str]): Features in model input order.

    Returns:
        tuple: ``inputs`` of ``(B, T, F)`` (or ``(B, L, T, F)`` for a
        multi-layer feature), ``masks`` of ``(B, T)`` with ``True`` = valid, and
        the feature names, which LinMulT needs to route special handling.

    Raises:
        KeyError: If a named feature or its mask is absent from the batch.
    """
    inputs: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []

    for name in feature_names:
        if name not in batch:
            raise KeyError(f"Batch has no feature {name!r}. Present: {sorted(batch)}.")
        mask_key = f"{name}_mask"
        if mask_key not in batch:
            raise KeyError(
                f"Batch has no mask for {name!r}. Every feature must be paired with a mask."
            )
        inputs.append(batch[name])
        masks.append(batch[mask_key])

    return inputs, masks, list(feature_names)
