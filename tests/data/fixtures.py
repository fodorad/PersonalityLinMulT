"""Builders for real, on-disk fixture data mirroring the FI preprocess output.

Tests write genuine ``.pkl``/``.npy``/``.st``/``.h5`` files in the formats the
real extractors emit, rather than mocking the readers. That way a change to an
extractor's contract surfaces as a test failure instead of passing against a
mock that encodes an assumption which is no longer true.

The synthetic clips use the same rates as the real data (WavLM/emotion2vec+ at
~49.79 Hz, DINOv2 at ~25.02 Hz) but only a couple of seconds long, so a full
fixture dataset builds in milliseconds.
"""

from __future__ import annotations

import csv
import pickle
from typing import TYPE_CHECKING

import h5py
import numpy as np
from safetensors.numpy import save_file

from personalitylinmult.data.schema import ALL_TRAITS, SUBSETS

if TYPE_CHECKING:
    from pathlib import Path

WAVLM_RATE_HZ = 49.79
"""Matches the measured WavLM/emotion2vec+ rate (20 ms stride)."""

DINOV2_RATE_HZ = 25.02
"""Matches the measured DINOv2 rate (25 fps video)."""

CLIP_SECONDS = 2.0
"""Fixture clip length; long enough to window, short enough to be instant."""

AUDIO_T = round(WAVLM_RATE_HZ * CLIP_SECONDS)
"""Timesteps per fixture audio feature."""

VISUAL_T = round(DINOV2_RATE_HZ * CLIP_SECONDS)
"""Timesteps per fixture visual feature."""

N_WAVLM_LAYERS = 12
"""Layer count emitted by the real WavLM extractor."""


def write_label_csv(path: Path, video_ids: list[str], subsets: list[str] | None = None) -> None:
    """Write a ``label.csv`` in the exact shape ``preprocess/fi`` produces.

    Args:
        path (Path): Destination CSV path.
        video_ids (list[str]): Clip ids, in file order.
        subsets (list[str] | None): Split per clip. Defaults to round-robin
            across train/valid/test so every split is populated.
    """
    if subsets is None:
        subsets = [SUBSETS[i % len(SUBSETS)] for i in range(len(video_ids))]

    rng = np.random.default_rng(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["video_id", "subset", *ALL_TRAITS, "text"])
        for video_id, subset in zip(video_ids, subsets, strict=True):
            neuroticism = float(rng.uniform(0, 1))
            row = {
                "openness": float(rng.uniform(0, 1)),
                "conscientiousness": float(rng.uniform(0, 1)),
                "extraversion": float(rng.uniform(0, 1)),
                "agreeableness": float(rng.uniform(0, 1)),
                "neuroticism": neuroticism,
                "emotional_stability": 1.0 - neuroticism,
                "interview": float(rng.uniform(0, 1)),
            }
            writer.writerow(
                [video_id, subset, *[f"{row[t]:.6f}" for t in ALL_TRAITS], "hello world"]
            )


def write_wavlm(directory: Path, video_ids: list[str], dim: int = 8) -> None:
    """Write pickled per-layer WavLM features.

    Args:
        directory (Path): Output directory.
        video_ids (list[str]): Clips to write.
        dim (int): Feature dimensionality.
    """
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    for video_id in video_ids:
        layers = [
            rng.standard_normal((AUDIO_T, dim)).astype(np.float32) for _ in range(N_WAVLM_LAYERS)
        ]
        with (directory / f"{video_id}.pkl").open("wb") as fh:
            pickle.dump(layers, fh)


def write_emotion2vec(directory: Path, video_ids: list[str], dim: int = 8) -> None:
    """Write ``.npy`` emotion2vec+ features.

    Args:
        directory (Path): Output directory.
        video_ids (list[str]): Clips to write.
        dim (int): Feature dimensionality.
    """
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2)
    for video_id in video_ids:
        features = rng.standard_normal((AUDIO_T, dim)).astype(np.float32)
        np.save(directory / f"{video_id}.npy", features)


def write_dinov2(directory: Path, video_ids: list[str], dim: int = 8) -> None:
    """Write safetensors DINOv2 features, which carry no ``mask`` key.

    Args:
        directory (Path): Output directory.
        video_ids (list[str]): Clips to write.
        dim (int): Feature dimensionality.
    """
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    for video_id in video_ids:
        save_file(
            {
                "frame_ids": np.arange(VISUAL_T, dtype=np.int64),
                "features": rng.standard_normal((VISUAL_T, dim)).astype(np.float32),
            },
            str(directory / f"{video_id}.st"),
        )


def write_masked_visual(
    directory: Path, video_ids: list[str], dim: int = 8, n_invalid: int = 3
) -> None:
    """Write safetensors features that carry a real ``mask`` (FaRL-style).

    Args:
        directory (Path): Output directory.
        video_ids (list[str]): Clips to write.
        dim (int): Feature dimensionality.
        n_invalid (int): Trailing timesteps to mark as undetected.
    """
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(4)
    for video_id in video_ids:
        features = rng.standard_normal((VISUAL_T, dim)).astype(np.float32)
        mask = np.ones((VISUAL_T,), dtype=bool)
        if n_invalid:
            mask[-n_invalid:] = False
            features[-n_invalid:] = 0.0
        save_file(
            {
                "frame_ids": np.arange(VISUAL_T, dtype=np.int64),
                "features": features,
                "mask": mask,
            },
            str(directory / f"{video_id}.st"),
        )


def write_text_h5(path: Path, video_ids: list[str], dim: int = 8) -> None:
    """Write a per-id HDF5 token-level text feature file.

    Args:
        path (Path): Destination file.
        video_ids (list[str]): Clips to write.
        dim (int): Feature dimensionality.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)
    with h5py.File(path, "w") as fh:
        for i, video_id in enumerate(video_ids):
            n_tokens = 3 + i  # ragged on purpose: text length genuinely varies
            fh.create_dataset(
                video_id, data=rng.standard_normal((n_tokens, dim)).astype(np.float32)
            )


def write_appledouble(path: Path) -> None:
    """Write a macOS AppleDouble sidecar that must never be read as a feature.

    Args:
        path (Path): Path of the *real* file; the sidecar is created beside it.
    """
    sidecar = path.parent / f"._{path.name}"
    sidecar.write_bytes(b"\x00\x05\x16\x07not a real feature file")
