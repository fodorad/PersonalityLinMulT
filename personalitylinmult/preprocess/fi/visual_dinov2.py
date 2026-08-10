"""Extract DINOv2 features from full frames of the FI dataset videos.

Iterates over ``video_id``s listed in ``data/processed/FI/label.csv``, reads
the matching ``{video_id}.mp4`` clip from ``data/processed/FI/video``, and
extracts per-frame DINOv2 embeddings via
:class:`~exordium.video.deep.dinov2.DINOv2Wrapper.video_to_feature`.

Unlike MARLIN (which operates on the protagonist's face crops), DINOv2 here
runs on the **full, unmodified video frame** — no face detection or
tracking involved.

Output:
    ``data/processed/FI/dinov2_base/{video_id}.st`` — safetensors dict with
    ``frame_ids`` (``(T,)``) and ``features`` (``(T, D)``, one L2-normalised
    CLS-token embedding per frame).

The script is resumable: extraction uses exordium's ``load_or_create``
caching, so a clip whose output file already exists is loaded from disk
instead of being recomputed. Use ``--overwrite`` to force recomputation.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from exordium.video.deep.dinov2 import DINOv2Wrapper
from tqdm import tqdm

logger = logging.getLogger(__name__)
"""Module-level logger."""

DB_PROCESSED = Path("data/processed/FI")
"""Processed FI root."""

DEFAULT_LABEL_CSV = DB_PROCESSED / "label.csv"
"""Source of ``video_id``s to process."""

DEFAULT_VIDEO_DIR = DB_PROCESSED / "video"
"""Input directory of per-clip ``{video_id}.mp4`` files."""

DEFAULT_DINOV2_MODEL = "base"
"""Default DINOv2 variant."""

DINOV2_DIR_NAMES = {
    "small": "dinov2_small",
    "base": "dinov2_base",
    "large": "dinov2_large",
    "giant": "dinov2_giant",
}
"""Maps each DINOv2 variant to its default output directory name."""


def default_dinov2_dir(dinov2_model: str) -> Path:
    """Return the default DINOv2 output directory for a given model variant.

    Args:
        dinov2_model: DINOv2 variant (``"small"``, ``"base"``, ``"large"``,
            or ``"giant"``).

    Returns:
        Default output directory, e.g. ``data/processed/FI/dinov2_base``.
    """
    return DB_PROCESSED / DINOV2_DIR_NAMES[dinov2_model]


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
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument(
        "--dinov2-dir",
        type=Path,
        default=None,
        help="Output directory for DINOv2 features. Defaults to "
        "data/processed/FI/dinov2_<model> based on --dinov2-model.",
    )
    parser.add_argument(
        "--dinov2-model",
        default=DEFAULT_DINOV2_MODEL,
        choices=["small", "base", "large", "giant"],
        help="DINOv2 variant to use.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=30,
        help="Frames to process per batch.",
    )
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
    if args.dinov2_dir is None:
        args.dinov2_dir = default_dinov2_dir(args.dinov2_model)

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    video_ids = load_video_ids(args.label_csv)
    logger.info(f"Found {len(video_ids)} video_ids in {args.label_csv}")
    if not video_ids:
        logger.warning(f"No video_ids in {args.label_csv}; nothing to do.")
        return

    device_id = args.gpu_id if args.gpu_id >= 0 else -1
    logger.info(f"Loading DINOv2 ({args.dinov2_model}) on gpu={device_id}...")
    dinov2_extractor = DINOv2Wrapper(model_name=args.dinov2_model, device_id=device_id)

    args.dinov2_dir.mkdir(parents=True, exist_ok=True)

    n_skipped = 0
    n_failed = 0
    for video_id in tqdm(video_ids, desc="DINOv2 features"):
        video_path = args.video_dir / f"{video_id}.mp4"
        output_path = args.dinov2_dir / f"{video_id}.st"

        if not video_path.exists():
            logger.warning(f"{video_id}: missing video file {video_path}; skipping.")
            n_failed += 1
            continue

        if not args.overwrite and output_path.exists():
            n_skipped += 1
            continue

        try:
            dinov2_extractor.video_to_feature(
                video_path,
                batch_size=args.batch_size,
                output_path=output_path,
                overwrite=args.overwrite,
            )
        except Exception as e:
            logger.exception(f"{video_id}: {e}")
            n_failed += 1

    logger.info(f"Done. dinov2_dir={args.dinov2_dir}, skipped={n_skipped}, failed={n_failed}")


if __name__ == "__main__":
    main()
