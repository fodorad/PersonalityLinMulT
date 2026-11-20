"""Train-only per-LLD z-score normalisation for eGeMAPS features.

eGeMAPS low-level descriptors (LLDs) live on very different scales (Hz, dB,
unitless ratios), so they are z-scored before they reach a network. The
statistics are fitted on **train clips only** and over **valid frames only**, so
neither padding nor validation/test data leaks into the normalisation. After
z-scoring, the train mean is 0, which makes the builder's zero padding neutral.

The core functions (:func:`fit_zscore_stats`, :func:`apply_zscore`) do no I/O so
they run on random arrays; :func:`normalize_directory` and :func:`main` are the
thin file layer.

Example:
    ``uv run python -m personalitylinmult.data.egemaps``
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from personalitylinmult.data.builder import DEFAULT_ROOT, load_labels

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)
"""Module-level logger."""

STD_FLOOR = 1e-6
"""Lower bound for a fitted std, so a constant LLD never divides by ~0."""

EPS = 1e-8
"""Added to the std when normalising, as in exordium's ``standardization``."""

DEFAULT_PROCESSED = Path("data/processed/FI")
"""Processed FI directory, relative to the repository root."""


class EgemapsError(ValueError):
    """Raised when eGeMAPS statistics cannot be fitted or applied."""


@dataclass(frozen=True)
class EgemapsStats:
    """Per-LLD z-score statistics.

    Args:
        names (tuple[str, ...]): LLD names, one per feature column.
        mean (np.ndarray): ``(D,)`` float64 per-LLD mean.
        std (np.ndarray): ``(D,)`` float64 per-LLD std, floored at
            :data:`STD_FLOOR`.

    Raises:
        EgemapsError: If the three fields disagree on the feature count.
    """

    names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self) -> None:
        """Validate that names, mean and std share one length."""
        if not (len(self.names) == len(self.mean) == len(self.std)):
            raise EgemapsError(
                f"names/mean/std lengths differ: "
                f"{len(self.names)}/{len(self.mean)}/{len(self.std)}."
            )

    def save(self, path: Path) -> None:
        """Write the statistics as JSON (``names``, ``mean``, ``std``).

        Args:
            path (Path): Destination file; parent directories are created.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "names": list(self.names),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path) -> EgemapsStats:
        """Read statistics written by :meth:`save`.

        Args:
            path (Path): JSON file.

        Returns:
            EgemapsStats: The loaded statistics.
        """
        payload = json.loads(path.read_text())
        return cls(
            names=tuple(payload["names"]),
            mean=np.asarray(payload["mean"], dtype=np.float64),
            std=np.asarray(payload["std"], dtype=np.float64),
        )


def fit_zscore_stats(
    clips: Iterable[np.ndarray],
    names: Iterable[str] | None = None,
    masks: Iterable[np.ndarray] | None = None,
) -> EgemapsStats:
    """Fit per-LLD mean and std over valid frames, one clip at a time.

    Clips are merged with the parallel (Chan) update in float64, so no giant
    concatenated matrix is held and the result is numerically stable.

    Args:
        clips (Iterable[np.ndarray]): ``(T_i, D)`` arrays, train clips only.
        names (Iterable[str] | None): LLD names. Defaults to ``lld_00``, ...
        masks (Iterable[np.ndarray] | None): ``(T_i,)`` boolean arrays aligned
            with ``clips``; ``False`` frames (padding) are ignored. ``None``
            treats every frame as valid.

    Returns:
        EgemapsStats: Statistics with the std floored at :data:`STD_FLOOR`.

    Raises:
        EgemapsError: If there are no valid frames, shapes are inconsistent, or
            ``masks`` and ``clips`` differ in length.
    """
    count = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None

    mask_iter = iter(masks) if masks is not None else None
    for clip in clips:
        clip = np.asarray(clip, dtype=np.float64)
        if clip.ndim != 2:
            raise EgemapsError(f"Expected a (T, D) clip, got shape {clip.shape}.")
        if mask_iter is not None:
            try:
                mask = np.asarray(next(mask_iter), dtype=bool)
            except StopIteration:
                raise EgemapsError("Fewer masks than clips.") from None
            if mask.shape != (clip.shape[0],):
                raise EgemapsError(f"Mask shape {mask.shape} does not match clip {clip.shape}.")
            clip = clip[mask]
        if mean is None or m2 is None:
            mean = np.zeros(clip.shape[1])
            m2 = np.zeros(clip.shape[1])
        elif clip.shape[1] != mean.shape[0]:
            raise EgemapsError(f"Clip has {clip.shape[1]} LLDs, expected {mean.shape[0]}.")
        n = clip.shape[0]
        if n == 0:
            continue
        clip_mean = clip.mean(axis=0)
        clip_m2 = ((clip - clip_mean) ** 2).sum(axis=0)
        total = count + n
        delta = clip_mean - mean
        mean = mean + delta * n / total
        m2 = m2 + clip_m2 + delta**2 * count * n / total
        count = total

    if mask_iter is not None and next(mask_iter, None) is not None:
        raise EgemapsError("More masks than clips.")
    if mean is None or m2 is None or count == 0:
        raise EgemapsError("No valid frames to fit statistics on.")

    std = np.maximum(np.sqrt(m2 / count), STD_FLOOR)
    resolved = (
        tuple(names) if names is not None else tuple(f"lld_{i:02d}" for i in range(len(mean)))
    )
    return EgemapsStats(names=resolved, mean=mean, std=std)


def apply_zscore(x: np.ndarray, stats: EgemapsStats) -> np.ndarray:
    """Z-score a clip: ``(x - mean) / (std + EPS)``.

    Args:
        x (np.ndarray): ``(T, D)`` raw LLDs.
        stats (EgemapsStats): Fitted statistics.

    Returns:
        np.ndarray: ``(T, D)`` float32 normalised LLDs.

    Raises:
        EgemapsError: If ``x`` does not have ``len(stats.names)`` columns.
    """
    if x.ndim != 2 or x.shape[1] != stats.mean.shape[0]:
        raise EgemapsError(f"Expected (T, {stats.mean.shape[0]}) input, got shape {x.shape}.")
    return ((x - stats.mean) / (stats.std + EPS)).astype(np.float32)


def normalize_directory(
    video_ids: list[str],
    train_ids: list[str],
    raw_dir: Path,
    out_dir: Path,
    stats_path: Path,
    names: list[str] | None = None,
) -> EgemapsStats:
    """Fit train-only statistics, then write a z-scored copy of every clip.

    Args:
        video_ids (list[str]): All clips to write to ``out_dir``.
        train_ids (list[str]): The train subset the statistics are fitted on.
        raw_dir (Path): Directory of raw ``{video_id}.npy`` ``(T, D)`` arrays.
        out_dir (Path): Destination for ``{video_id}.npy`` z-scored arrays.
        stats_path (Path): Where the statistics JSON is written.
        names (list[str] | None): LLD names to record in the stats file.

    Returns:
        EgemapsStats: The fitted statistics.
    """
    stats = fit_zscore_stats((np.load(raw_dir / f"{v}.npy") for v in train_ids), names=names)
    stats.save(stats_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    for video_id in video_ids:
        raw = np.load(raw_dir / f"{video_id}.npy")
        np.save(out_dir / f"{video_id}.npy", apply_zscore(raw, stats))
    logger.info(f"Wrote {len(video_ids)} z-scored clips to {out_dir} (stats: {stats_path}).")
    return stats


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: fit train-only stats and write ``egemaps_z/``.

    Args:
        argv (list[str] | None): Arguments; defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Repository root.")
    parser.add_argument(
        "--processed",
        type=Path,
        default=DEFAULT_PROCESSED,
        help="Processed FI directory (relative to --root) holding label.csv and egemaps/.",
    )
    parser.add_argument(
        "--names-file",
        type=Path,
        default=None,
        help="Optional JSON list of LLD names to record in the stats file.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    base = args.root / args.processed
    video_ids, subsets, _ = load_labels(base / "label.csv")
    train_ids = [v for v, s in zip(video_ids, subsets, strict=True) if s == "train"]
    names = json.loads(args.names_file.read_text()) if args.names_file else None
    normalize_directory(
        video_ids,
        train_ids,
        raw_dir=base / "egemaps",
        out_dir=base / "egemaps_z",
        stats_path=base / "egemaps_zscore.json",
        names=names,
    )


if __name__ == "__main__":
    main()
