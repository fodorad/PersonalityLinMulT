"""Probe how much of eGeMAPS WavLM layer 9 encodes, and the reverse.

Frame-level closed-form ridge regression (see :mod:`personalitylinmult.probe.ridge`)
from WavLM layer-9 frames (50 Hz) to the 25 z-scored eGeMAPS LLDs, average-pooled
2:1 from 100 Hz to 50 Hz. Controls give R² a floor: the train mean, a randomly
initialised WavLM, and real WavLM on the same clip subset the random-init control
uses. A reverse probe maps the 25 LLDs to the 768 WavLM dimensions.

Every clip contributes its first ``min(T_wavlm, T_egemaps // 2, 750)`` frames
(15 s, the training window); all kept frames are real, unvoiced ones included.

Example:
    ``uv run python -m personalitylinmult.probe.egemaps_probe``
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from personalitylinmult.data.builder import DEFAULT_ROOT, load_labels
from personalitylinmult.data.sources import read_pickle_layers
from personalitylinmult.probe.ridge import (
    ALPHA_GRID,
    ProbeMetrics,
    RidgeAccumulator,
    evaluate,
    select_alpha,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

logger = logging.getLogger(__name__)
"""Module-level logger."""

WAVLM_LAYER = 9
"""WavLM layer probed."""

MAX_FRAMES = 750
"""Frames kept per clip at 50 Hz (the first 15 s)."""

POOL = 2
"""Average-pooling factor taking eGeMAPS from 100 Hz to WavLM's 50 Hz."""

SEED = 42
"""Seed for subset selection."""

N_SUBSET_TRAIN = 1000
"""Train clips in the control subset."""

N_SUBSET_VALID = 200
"""Valid clips in the control subset."""

DEFAULT_PROCESSED = Path("data/processed/FI")
"""Processed FI directory, relative to the repository root."""

DEFAULT_OUT_DIR = Path("results/FI/probe")
"""Where results land, relative to the repository root."""

SUBSET_CSV_NAME = "label_probe_subset.csv"
"""Subset ids in ``label.csv`` schema, consumable by the WavLM extractor."""


def pool_time(x: np.ndarray, factor: int = POOL) -> np.ndarray:
    """Average-pool along time in non-overlapping groups.

    Args:
        x: Array of shape ``(T, D)``.
        factor: Group size.

    Returns:
        Array of shape ``(T // factor, D)``; a trailing partial group is dropped.
    """
    n = x.shape[0] // factor
    return x[: n * factor].reshape(n, factor, -1).mean(axis=1)


def align(
    wavlm: np.ndarray, egemaps: np.ndarray, max_frames: int = MAX_FRAMES
) -> tuple[Tensor, Tensor]:
    """Pool eGeMAPS to 50 Hz and trim both streams to a common length.

    Args:
        wavlm: WavLM frames, ``(T_w, 768)``.
        egemaps: z-scored LLDs at 100 Hz, ``(T_e, 25)``.
        max_frames: Upper bound on the kept frames.

    Returns:
        ``(wavlm, egemaps)`` float32 tensors of shape ``(T, 768)`` and ``(T, 25)``
        with ``T = min(T_w, T_e // 2, max_frames)``.
    """
    pooled = pool_time(egemaps)
    n = min(wavlm.shape[0], pooled.shape[0], max_frames)
    return (
        torch.from_numpy(np.ascontiguousarray(wavlm[:n], dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(pooled[:n], dtype=np.float32)),
    )


def load_clip(
    video_id: str, wavlm_dir: Path, egemaps_dir: Path, layer: int = WAVLM_LAYER
) -> tuple[Tensor, Tensor]:
    """Load and align one clip.

    Args:
        video_id: Clip id.
        wavlm_dir: Directory of ``{video_id}.pkl`` WavLM pickles.
        egemaps_dir: Directory of ``{video_id}.npy`` z-scored LLDs.
        layer: WavLM layer to read.

    Returns:
        Aligned ``(wavlm, egemaps)`` tensors, see :func:`align`.
    """
    wavlm = read_pickle_layers(wavlm_dir / f"{video_id}.pkl", (layer,)).features
    egemaps = np.load(egemaps_dir / f"{video_id}.npy")
    return align(wavlm, egemaps)


def select_subset(
    video_ids: Sequence[str],
    subsets: Sequence[str],
    n_train: int = N_SUBSET_TRAIN,
    n_valid: int = N_SUBSET_VALID,
    seed: int = SEED,
) -> list[str]:
    """Sample a fixed number of train and valid clips; test is not involved.

    Args:
        video_ids: All clip ids.
        subsets: Split name per clip.
        n_train: Train clips to draw (capped at availability).
        n_valid: Valid clips to draw (capped at availability).
        seed: Sampling seed.

    Returns:
        Sorted ids of the drawn clips, deterministic for a given seed.
    """
    rng = np.random.default_rng(seed)
    chosen: list[str] = []
    for split, n in (("train", n_train), ("valid", n_valid)):
        pool = sorted(v for v, s in zip(video_ids, subsets, strict=True) if s == split)
        take = min(n, len(pool))
        chosen += [pool[i] for i in rng.choice(len(pool), size=take, replace=False)]
    return sorted(chosen)


def write_subset_csv(label_csv: Path, out_path: Path, subset_ids: Sequence[str]) -> None:
    """Write the control subset plus every test clip in ``label.csv`` schema.

    Args:
        label_csv: Full ``label.csv``.
        out_path: Destination CSV.
        subset_ids: Chosen train and valid ids, see :func:`select_subset`.
    """
    frame = pd.read_csv(label_csv, dtype={"video_id": str})
    keep = frame["video_id"].isin(set(subset_ids)) | (frame["subset"] == "test")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame[keep].to_csv(out_path, index=False)
    logger.info(f"Wrote {int(keep.sum())} clips to {out_path}.")


def iter_chunks(
    ids: Sequence[str], wavlm_dir: Path, egemaps_dir: Path, reverse: bool = False
) -> Iterator[tuple[Tensor, Tensor]]:
    """Stream aligned ``(x, y)`` frames, one clip per chunk.

    Args:
        ids: Clip ids.
        wavlm_dir: WavLM pickle directory.
        egemaps_dir: z-scored LLD directory.
        reverse: ``False`` for WavLM -> eGeMAPS, ``True`` for eGeMAPS -> WavLM.

    Yields:
        ``(x, y)`` frame matrices of one clip.
    """
    for video_id in ids:
        wavlm, egemaps = load_clip(video_id, wavlm_dir, egemaps_dir)
        yield (egemaps, wavlm) if reverse else (wavlm, egemaps)


@dataclass
class ProbeResult:
    """Outcome of one probe experiment.

    Attributes:
        alpha: Selected ridge penalty.
        valid_mean_r2: Mean R² on the validation split at ``alpha``.
        test: Metrics on the test split.
    """

    alpha: float
    valid_mean_r2: float
    test: ProbeMetrics


def _fit_accumulator(
    train_ids: Sequence[str], wavlm_dir: Path, egemaps_dir: Path, reverse: bool
) -> RidgeAccumulator:
    """Stream the train clips into a fresh accumulator."""
    acc: RidgeAccumulator | None = None
    for x, y in iter_chunks(train_ids, wavlm_dir, egemaps_dir, reverse):
        if acc is None:
            acc = RidgeAccumulator(x.shape[1], y.shape[1])
        acc.update(x, y)
    if acc is None:
        raise ValueError("no train clips")
    return acc


def run_probe(
    splits: Mapping[str, Sequence[str]],
    wavlm_dir: Path,
    egemaps_dir: Path,
    reverse: bool = False,
    alphas: Sequence[float] = ALPHA_GRID,
) -> ProbeResult:
    """Fit on train, pick alpha on valid, score on test.

    Args:
        splits: Clip ids for ``"train"``, ``"valid"`` and ``"test"``.
        wavlm_dir: WavLM pickle directory.
        egemaps_dir: z-scored LLD directory.
        reverse: Probe eGeMAPS -> WavLM instead of WavLM -> eGeMAPS.
        alphas: Candidate ridge penalties.

    Returns:
        The :class:`ProbeResult`.
    """
    acc = _fit_accumulator(splits["train"], wavlm_dir, egemaps_dir, reverse)
    valid = list(iter_chunks(splits["valid"], wavlm_dir, egemaps_dir, reverse))
    probe, alpha = select_alpha(acc, alphas, valid)
    return ProbeResult(
        alpha=alpha,
        valid_mean_r2=evaluate(probe, valid).mean_r2,
        test=evaluate(probe, iter_chunks(splits["test"], wavlm_dir, egemaps_dir, reverse)),
    )


def run_mean_control(
    splits: Mapping[str, Sequence[str]], wavlm_dir: Path, egemaps_dir: Path
) -> ProbeResult:
    """Score the train-mean predictor (a probe with zero weights).

    Args:
        splits: Clip ids for ``"train"``, ``"valid"`` and ``"test"``.
        wavlm_dir: WavLM pickle directory.
        egemaps_dir: z-scored LLD directory.

    Returns:
        A :class:`ProbeResult` with ``alpha=inf``; test R² is about 0.
    """
    acc = _fit_accumulator(splits["train"], wavlm_dir, egemaps_dir, reverse=False)
    probe = acc.solve([0.0])[0.0]
    probe = replace(probe, weight=torch.zeros_like(probe.weight))
    valid = iter_chunks(splits["valid"], wavlm_dir, egemaps_dir)
    return ProbeResult(
        alpha=float("inf"),
        valid_mean_r2=evaluate(probe, valid).mean_r2,
        test=evaluate(probe, iter_chunks(splits["test"], wavlm_dir, egemaps_dir)),
    )


def result_to_dict(result: ProbeResult, names: Sequence[str] | None = None) -> dict:
    """Convert a result to a JSON-serialisable dict.

    Args:
        result: Probe result.
        names: Optional output names; when given, per-output ``r2`` and
            ``pearson_r`` are included keyed by name.

    Returns:
        Dict with ``alpha``, ``valid_mean_r2``, test summary metrics and, with
        ``names``, per-output values.
    """
    test = result.test
    summary: dict[str, Any] = {
        "mean_r2": test.mean_r2,
        "mean_pearson_r": test.mean_pearson_r,
        "weighted_r2": test.weighted_r2,
    }
    if names is not None:
        summary["r2"] = dict(zip(names, test.r2.tolist(), strict=True))
        summary["pearson_r"] = dict(zip(names, test.pearson_r.tolist(), strict=True))
    return {"alpha": result.alpha, "valid_mean_r2": result.valid_mean_r2, "test": summary}


def format_table(results: dict[str, dict]) -> str:
    """Render the summary of several experiments as a markdown table.

    Args:
        results: Experiment name -> :func:`result_to_dict` output.

    Returns:
        Markdown table text.
    """
    lines = [
        "| experiment | alpha | valid mean R2 | test mean R2 | test mean r | test weighted R2 |",
        "|---|---|---|---|---|---|",
    ]
    for name, res in results.items():
        t = res["test"]
        lines.append(
            f"| {name} | {res['alpha']:g} | {res['valid_mean_r2']:.4f} | {t['mean_r2']:.4f} "
            f"| {t['mean_pearson_r']:.4f} | {t['weighted_r2']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def write_results(results: dict[str, dict], out_dir: Path) -> None:
    """Write ``probe_results.json`` and ``probe_results.md``.

    Args:
        results: Experiment name -> :func:`result_to_dict` output.
        out_dir: Destination directory (created).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe_results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "probe_results.md").write_text(format_table(results))
    logger.info(f"Wrote probe results to {out_dir}.")


def split_ids(video_ids: Sequence[str], subsets: Sequence[str]) -> dict[str, list[str]]:
    """Group clip ids by split.

    Args:
        video_ids: Clip ids.
        subsets: Split name per clip.

    Returns:
        Mapping ``split -> ids`` for train, valid and test.
    """
    return {
        split: [v for v, s in zip(video_ids, subsets, strict=True) if s == split]
        for split in ("train", "valid", "test")
    }


def run_all(
    label_csv: Path,
    wavlm_dir: Path,
    egemaps_dir: Path,
    out_dir: Path,
    subset_csv: Path,
    random_wavlm_dir: Path | None = None,
    names: Sequence[str] | None = None,
    n_train: int = N_SUBSET_TRAIN,
    n_valid: int = N_SUBSET_VALID,
    seed: int = SEED,
) -> dict[str, dict]:
    """Run the forward probe, controls and reverse probe, and write results.

    Args:
        label_csv: Full ``label.csv``.
        wavlm_dir: Real WavLM layer-9 pickles.
        egemaps_dir: z-scored LLD ``.npy`` files.
        out_dir: Results directory.
        subset_csv: Where the control subset ids are written.
        random_wavlm_dir: Random-init WavLM pickles; that control is skipped
            when ``None`` or missing.
        names: LLD names for per-LLD results.
        n_train: Control-subset train clips.
        n_valid: Control-subset valid clips.
        seed: Subset seed.

    Returns:
        Experiment name -> result dict, as written to disk.
    """
    video_ids, subsets, _ = load_labels(label_csv)
    full = split_ids(video_ids, subsets)
    chosen = select_subset(video_ids, subsets, n_train, n_valid, seed)
    write_subset_csv(label_csv, subset_csv, chosen)
    chosen_set = set(chosen)
    sub = {
        "train": [v for v in full["train"] if v in chosen_set],
        "valid": [v for v in full["valid"] if v in chosen_set],
        "test": full["test"],
    }

    results: dict[str, dict] = {}

    def record(name: str, result: ProbeResult, per_output: bool = True) -> None:
        results[name] = result_to_dict(result, names if per_output else None)
        logger.info(f"{name}: test mean R2 {result.test.mean_r2:.4f} (alpha {result.alpha:g})")

    record("train_mean", run_mean_control(full, wavlm_dir, egemaps_dir))
    record("forward_full", run_probe(full, wavlm_dir, egemaps_dir))
    record("forward_subset", run_probe(sub, wavlm_dir, egemaps_dir))
    if random_wavlm_dir is not None and random_wavlm_dir.is_dir():
        record("forward_subset_random_init", run_probe(sub, random_wavlm_dir, egemaps_dir))
    else:
        logger.warning("Random-init WavLM directory missing; skipping that control.")
    record("reverse_full", run_probe(full, wavlm_dir, egemaps_dir, reverse=True), per_output=False)
    write_results(results, out_dir)
    return results


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    Args:
        argv: Arguments; defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Repository root.")
    parser.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-train", type=int, default=N_SUBSET_TRAIN)
    parser.add_argument("--n-valid", type=int, default=N_SUBSET_VALID)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    base = args.root / args.processed
    stats_path = base / "egemaps_zscore.json"
    names = json.loads(stats_path.read_text()).get("names") if stats_path.is_file() else None
    run_all(
        label_csv=base / "label.csv",
        wavlm_dir=base / "wavlm_baseplus_l9",
        egemaps_dir=base / "egemaps_z",
        out_dir=args.root / args.out_dir,
        subset_csv=base / SUBSET_CSV_NAME,
        random_wavlm_dir=base / "wavlm_baseplus_l9_random",
        names=names or None,
        n_train=args.n_train,
        n_valid=args.n_valid,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
