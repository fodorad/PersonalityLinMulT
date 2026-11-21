"""Extract eGeMAPS low-level descriptors for the FI dataset.

Iterates over ``video_id``s listed in ``data/processed/FI/label.csv``, reads the
matching ``{video_id}.wav`` clip from ``data/processed/FI/audio`` and extracts
the openSMILE eGeMAPSv02 LLDs via :class:`~exordium.audio.smile.OpensmileWrapper`.

Output:
    * ``data/processed/FI/egemaps/{video_id}.npy``: raw ``(T, 25)`` float32
      array (``T`` is about 1530 for a 15.3 s clip at 100 Hz).
    * ``data/processed/FI/egemaps_names.json``: the 25 LLD column names, as
      reported by openSMILE (``smile.feature_names``). Pass it to
      ``python -m personalitylinmult.data.egemaps --names-file`` so the
      z-score statistics file records which LLD each column is.

The arrays are *raw*: z-scoring needs train-only statistics, which is the job
of :mod:`personalitylinmult.data.egemaps`.

The script is resumable: a clip whose output file already exists is skipped.
Use ``--overwrite`` to force recomputation.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)
"""Module-level logger."""

DB_PROCESSED = Path("data/processed/FI")
"""Processed FI root (contains ``label.csv`` and ``audio/``)."""

DEFAULT_LABEL_CSV = DB_PROCESSED / "label.csv"
"""Source of ``video_id``s to process."""

DEFAULT_AUDIO_DIR = DB_PROCESSED / "audio"
"""Input directory of per-clip ``{video_id}.wav`` files."""

DEFAULT_EGEMAPS_DIR = DB_PROCESSED / "egemaps"
"""Output directory of raw per-clip ``{video_id}.npy`` arrays."""

DEFAULT_NAMES_PATH = DB_PROCESSED / "egemaps_names.json"
"""Output JSON list of LLD column names."""

APPLEDOUBLE_PREFIX = "._"
"""macOS AppleDouble sidecar prefix; such entries are never real clips."""


def load_video_ids(label_csv: Path) -> list[str]:
    """Return the ordered ``video_id``s from the label CSV, without sidecars.

    Args:
        label_csv (Path): Path to ``label.csv`` with a ``video_id`` column.

    Returns:
        list[str]: Video IDs in file order, AppleDouble ``._*`` entries dropped.
    """
    df = pd.read_csv(label_csv)
    ids = df["video_id"].astype(str).tolist()
    return [v for v in ids if not v.startswith(APPLEDOUBLE_PREFIX)]


def is_pending(output_path: Path, overwrite: bool) -> bool:
    """Tell whether a clip still has to be extracted.

    Args:
        output_path (Path): Destination ``{video_id}.npy``.
        overwrite (bool): Recompute even if the output exists.

    Returns:
        bool: True if the clip should be processed.
    """
    return overwrite or not output_path.exists()


def save_array(array: np.ndarray, output_path: Path) -> None:
    """Write a float32 array atomically.

    A temp file plus rename keeps an interrupted run from leaving a truncated
    ``.npy`` that the resume logic would mistake for a finished clip.

    Args:
        array (np.ndarray): Array to store.
        output_path (Path): Destination ``.npy`` path.
    """
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    with tmp_path.open("wb") as fh:
        np.save(fh, array.astype(np.float32))
    tmp_path.replace(output_path)


def save_names(names: list[str], path: Path) -> None:
    """Write the LLD column names as a JSON list.

    Args:
        names (list[str]): LLD names, one per column.
        path (Path): Destination JSON path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(names), indent=2))


def select_shard(items: list[str], shard: int, num_shards: int) -> list[str]:
    """Return the round-robin slice of ``items`` owned by one worker.

    Workers given the same ``num_shards`` and distinct ``shard`` indices
    partition ``items`` without overlap, so extraction can run in parallel
    processes. Round-robin keeps the shards balanced.

    Args:
        items (list[str]): Items to split, e.g. video ids.
        shard (int): This worker's index in ``[0, num_shards)``.
        num_shards (int): Total number of workers.

    Returns:
        list[str]: The items assigned to ``shard``.

    Raises:
        ValueError: If ``num_shards < 1`` or ``shard`` is out of range.
    """
    if num_shards < 1 or not 0 <= shard < num_shards:
        raise ValueError(f"Invalid shard {shard} of {num_shards}.")
    return items[shard::num_shards]


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label-csv", type=Path, default=DEFAULT_LABEL_CSV)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--egemaps-dir", type=Path, default=DEFAULT_EGEMAPS_DIR)
    parser.add_argument("--names-path", type=Path, default=DEFAULT_NAMES_PATH)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N clips (smoke test).",
    )
    parser.add_argument(
        "--shard",
        type=int,
        default=0,
        help="Worker index in [0, --num-shards); run one process per shard to parallelise.",
    )
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of workers.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute features even if the output file already exists.",
    )
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    video_ids = load_video_ids(args.label_csv)[: args.limit]
    video_ids = select_shard(video_ids, args.shard, args.num_shards)
    logger.info(f"Found {len(video_ids)} video_ids in {args.label_csv}")
    if not video_ids:
        logger.warning(f"No video_ids in {args.label_csv}; nothing to do.")
        return

    # Imported here, not at module level: openSMILE ships only with the
    # preprocess extra, and the pure helpers above must stay importable (and
    # testable) in environments that lack it, such as CI.
    from exordium.audio.smile import OpensmileWrapper

    extractor = OpensmileWrapper("egemaps", "lld")
    names = list(extractor.smile.feature_names)
    save_names(names, args.names_path)
    logger.info(f"Saved {len(names)} LLD names -> {args.names_path}")

    args.egemaps_dir.mkdir(parents=True, exist_ok=True)
    n_skipped = 0
    n_failed = 0
    for video_id in tqdm(video_ids, desc="eGeMAPS LLDs"):
        audio_path = args.audio_dir / f"{video_id}.wav"
        output_path = args.egemaps_dir / f"{video_id}.npy"

        if not audio_path.exists():
            logger.warning(f"{video_id}: missing audio file {audio_path}; skipping.")
            n_failed += 1
            continue

        if not is_pending(output_path, args.overwrite):
            n_skipped += 1
            continue

        try:
            save_array(np.asarray(extractor(audio_path)), output_path)
        except Exception as e:
            logger.exception(f"{video_id}: {e}")
            n_failed += 1

    logger.info(f"Done. egemaps_dir={args.egemaps_dir}, skipped={n_skipped}, failed={n_failed}")


if __name__ == "__main__":
    main()
