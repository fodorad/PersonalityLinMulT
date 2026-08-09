"""Extract MARLIN features from the FI dataset's protagonist face tracks.

Iterates over ``video_id``s listed in ``data/processed/FI/label.csv``, loads
the protagonist's face track (selected by ``create_tracks.py`` and recorded
in ``data/processed/FI/protagonist_track_heuristic.csv``), and extracts
clip-wise MARLIN embeddings via
:class:`~exordium.video.deep.marlin.MarlinWrapper.track_to_feature`.

MARLIN operates on fixed 16-frame clips, so the track is windowed over the
**full video timeline** rather than just its own detections. The number of
frames per video is read from the video file itself
(:func:`~exordium.video.core.io.get_video_metadata`) so the output grid
covers the entire clip, not just the span of the track's detections.

Output:
    ``data/processed/FI/marlin_base/{video_id}.st`` — safetensors dict with
    ``frame_ids`` (window start indices), ``features`` (``(W, D)``), and
    ``mask`` (``(W,)``, True where the window has at least one detection).

The script is resumable: extraction uses exordium's ``load_or_create``
caching, so a clip whose output file already exists is loaded from disk
instead of being recomputed. Use ``--overwrite`` to force recomputation.
Clips with no usable protagonist track (``track_id == -1``) are skipped.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from exordium.video.core.detection import Track
from exordium.video.core.io import get_video_metadata
from exordium.video.deep.marlin import MarlinWrapper
from tqdm import tqdm

logger = logging.getLogger(__name__)
"""Module-level logger."""

DB_PROCESSED = Path("data/processed/FI")
"""Processed FI root."""

DEFAULT_LABEL_CSV = DB_PROCESSED / "label.csv"
"""Source of ``video_id``s to process."""

DEFAULT_PROTAGONIST_CSV = DB_PROCESSED / "protagonist_track_heuristic.csv"
"""Maps ``video_id`` to its protagonist ``track_id`` (``-1`` = none)."""

DEFAULT_TRACK_DIR = DB_PROCESSED / "track"
"""Input directory of ``{video_id}__track{track_id}.csv`` files."""

DEFAULT_VIDEO_DIR = DB_PROCESSED / "video"
"""Input directory of ``{video_id}.mp4`` files (for frame count)."""

DEFAULT_MARLIN_MODEL = "base"
"""Default MARLIN variant."""

MARLIN_DIR_NAMES = {"small": "marlin_small", "base": "marlin_base", "large": "marlin_large"}
"""Maps each MARLIN variant to its default output directory name."""


def default_marlin_dir(marlin_model: str) -> Path:
    """Return the default MARLIN output directory for a given model variant.

    Args:
        marlin_model: MARLIN variant (``"small"``, ``"base"``, or ``"large"``).

    Returns:
        Default output directory, e.g. ``data/processed/FI/marlin_base``.
    """
    return DB_PROCESSED / MARLIN_DIR_NAMES[marlin_model]


def load_protagonist_tracks(protagonist_csv: Path) -> dict[str, int]:
    """Return a ``{video_id: track_id}`` mapping from the protagonist CSV.

    Args:
        protagonist_csv: Path to ``protagonist_track_heuristic.csv``.

    Returns:
        Mapping of video ID to its protagonist track ID (``-1`` = none).
    """
    df = pd.read_csv(protagonist_csv)
    return dict(zip(df["uuid"].astype(str), df["track_id"].astype(int), strict=True))


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
    parser.add_argument("--protagonist-csv", type=Path, default=DEFAULT_PROTAGONIST_CSV)
    parser.add_argument("--track-dir", type=Path, default=DEFAULT_TRACK_DIR)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument(
        "--marlin-dir",
        type=Path,
        default=None,
        help="Output directory for MARLIN features. Defaults to "
        "data/processed/FI/marlin_<model> based on --marlin-model.",
    )
    parser.add_argument(
        "--marlin-model",
        default=DEFAULT_MARLIN_MODEL,
        choices=["small", "base", "large"],
        help="MARLIN variant to use.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=16,
        help="Step between clip windows in frames (16 = non-overlapping).",
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
    if args.marlin_dir is None:
        args.marlin_dir = default_marlin_dir(args.marlin_model)

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    video_ids = load_video_ids(args.label_csv)
    logger.info(f"Found {len(video_ids)} video_ids in {args.label_csv}")
    if not video_ids:
        logger.warning(f"No video_ids in {args.label_csv}; nothing to do.")
        return

    protagonist_tracks = load_protagonist_tracks(args.protagonist_csv)

    device_id = args.gpu_id if args.gpu_id >= 0 else -1
    logger.info(f"Loading MARLIN ({args.marlin_model}) on gpu={device_id}...")
    marlin_extractor = MarlinWrapper(model_name=args.marlin_model, device_id=device_id)

    args.marlin_dir.mkdir(parents=True, exist_ok=True)

    n_skipped = 0
    n_failed = 0
    n_no_track = 0
    for video_id in tqdm(video_ids, desc="MARLIN features"):
        output_path = args.marlin_dir / f"{video_id}.st"

        if not args.overwrite and output_path.exists():
            n_skipped += 1
            continue

        track_id = protagonist_tracks.get(video_id, -1)
        if track_id < 0:
            logger.warning(f"{video_id}: no protagonist track; skipping.")
            n_no_track += 1
            continue

        track_path = args.track_dir / f"{video_id}__track{track_id}.csv"
        video_path = args.video_dir / f"{video_id}.mp4"

        if not track_path.exists():
            logger.warning(f"{video_id}: missing track file {track_path}; skipping.")
            n_failed += 1
            continue
        if not video_path.exists():
            logger.warning(f"{video_id}: missing video file {video_path}; skipping.")
            n_failed += 1
            continue

        try:
            track = Track().load(track_path)
            num_frames = get_video_metadata(video_path)["num_frames"]
            marlin_extractor.track_to_feature(
                track,
                num_frames=num_frames,
                stride=args.stride,
                output_path=output_path,
                overwrite=args.overwrite,
            )
        except Exception as e:
            logger.exception(f"{video_id}: {e}")
            n_failed += 1

    logger.info(
        f"Done. marlin_dir={args.marlin_dir}, skipped={n_skipped}, "
        f"no_track={n_no_track}, failed={n_failed}"
    )


if __name__ == "__main__":
    main()
