"""Extract emotion2vec+ features for the FI dataset.

Iterates over ``video_id``s listed in ``data/processed/FI/label.csv``, reads
the matching ``{video_id}.wav`` clip from ``data/processed/FI/audio``, and
extracts frame-level emotion2vec+ features via
:class:`~exordium.audio.emotion2vec.Emotion2vecWrapper`.

Output:
    ``data/processed/FI/emotion2vecplus/{video_id}.npy`` — a
    ``(T, 768)`` numpy array (~50 frames per second).

The script is resumable: extraction uses exordium's ``load_or_create``
caching, so a clip whose output file already exists is loaded from disk
instead of being recomputed. Use ``--overwrite`` to force recomputation.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from exordium.audio.emotion2vec import Emotion2vecWrapper
from tqdm import tqdm

logger = logging.getLogger(__name__)
"""Module-level logger."""

DB_PROCESSED = Path("data/processed/FI")
"""Processed FI root (contains ``label.csv`` and ``audio/``)."""

DEFAULT_LABEL_CSV = DB_PROCESSED / "label.csv"
"""Source of ``video_id``s to process."""

DEFAULT_AUDIO_DIR = DB_PROCESSED / "audio"
"""Input directory of per-clip ``{video_id}.wav`` files."""

DEFAULT_EMOTION2VEC_DIR = DB_PROCESSED / "emotion2vecplus"
"""Output directory for per-clip emotion2vec+ feature arrays."""


def load_video_ids(label_csv: Path) -> list[str]:
    """Return the ordered list of ``video_id``s from the label CSV.

    Args:
        label_csv: Path to ``label.csv`` with a ``video_id`` column.

    Returns:
        List of video IDs in file order.
    """
    df = pd.read_csv(label_csv)
    return df["video_id"].astype(str).tolist()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label-csv", type=Path, default=DEFAULT_LABEL_CSV)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--emotion2vec-dir", type=Path, default=DEFAULT_EMOTION2VEC_DIR)
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="Device index for feature extraction (negative = CPU).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute features even if the output file already exists.",
    )
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    video_ids = load_video_ids(args.label_csv)
    logger.info(f"Found {len(video_ids)} video_ids in {args.label_csv}")
    if not video_ids:
        logger.warning(f"No video_ids in {args.label_csv}; nothing to do.")
        return

    device_id = args.gpu_id if args.gpu_id >= 0 else -1
    logger.info(f"Loading emotion2vec+ on gpu={device_id}...")
    emotion2vec_extractor = Emotion2vecWrapper(device_id=device_id)

    args.emotion2vec_dir.mkdir(parents=True, exist_ok=True)

    n_skipped = 0
    n_failed = 0
    for video_id in tqdm(video_ids, desc="emotion2vec+ features"):
        audio_path = args.audio_dir / f"{video_id}.wav"
        emotion2vec_path = args.emotion2vec_dir / f"{video_id}.npy"

        if not audio_path.exists():
            logger.warning(f"{video_id}: missing audio file {audio_path}; skipping.")
            n_failed += 1
            continue

        if not args.overwrite and emotion2vec_path.exists():
            n_skipped += 1
            continue

        try:
            emotion2vec_extractor.audio_to_feature(
                audio_path, output_path=emotion2vec_path, overwrite=args.overwrite
            )
        except Exception as e:
            logger.exception(f"{video_id}: {e}")
            n_failed += 1

    logger.info(
        f"Done. emotion2vec_dir={args.emotion2vec_dir}, skipped={n_skipped}, failed={n_failed}"
    )


if __name__ == "__main__":
    main()
