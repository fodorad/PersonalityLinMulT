"""Extract EmotiEffNet features from the FI dataset's protagonist face tracks.

Iterates over ``video_id``s listed in ``data/processed/FI/label.csv``, loads
the protagonist's face track (selected by ``create_tracks.py`` and recorded
in ``data/processed/FI/protagonist_track_heuristic.csv``), and extracts
per-detection EmotiEffNet embeddings via
:meth:`~exordium.video.deep.emotieffnet.EmotiEffNetWrapper.track_to_feature`.

**Densification.** ``track_to_feature`` returns only the frames the track
actually covers (``(N, D)`` with ``N == len(track)``). The protagonist is not
always detected on every frame, so ``N`` can be smaller than the video's frame
count. Passing ``densify=True`` scatters that sparse output onto the full
timeline, keeping the time axis aligned with the video and therefore with the
other modalities: frames covered by the track hold their feature and
``mask=True``; frames with no detection hold a zero vector and ``mask=False``.

``end_frame_id`` is passed explicitly. Without it exordium ends the grid at the
last *detection*, which silently truncates a clip whose protagonist leaves
before the final frame — the time axis would then no longer match the audio.

Output:
    ``data/processed/FI/emotieffnet_<model>/{video_id}.st`` — safetensors dict
    with ``frame_ids`` (``(T,)``, always ``0..T-1``), ``features``
    (``(T, D)``), and ``mask`` (``(T,)`` bool, True where the protagonist was
    detected).

The script is resumable: a clip whose output file already exists is skipped.
Use ``--overwrite`` to force recomputation. Clips with no usable protagonist
track (``track_id == -1``) are skipped.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from exordium.video.core.detection import Track
from exordium.video.core.io import get_video_metadata
from exordium.video.deep.emotieffnet import EmotiEffNetWrapper
from safetensors.torch import save_file
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

DEFAULT_EMOTIEFFNET_MODEL = "enet_b0_8_best_vgaf"
"""Default EmotiEffNet variant (exordium's own default)."""


def default_output_dir(model_name: str) -> Path:
    """Return the default output directory for a given EmotiEffNet variant.

    Args:
        model_name: EmotiEffNet variant, e.g. ``"enet_b0_8_best_vgaf"``.

    Returns:
        Output directory, e.g. ``data/processed/FI/emotieffnet_enet_b0_8_best_vgaf``.
    """
    return DB_PROCESSED / f"emotieffnet_{model_name}"


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
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to data/processed/FI/emotieffnet_<model>.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_EMOTIEFFNET_MODEL,
        help="EmotiEffNet variant to use.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=30,
        help="Detections to process per batch.",
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
    if args.output_dir is None:
        args.output_dir = default_output_dir(args.model_name)

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    video_ids = load_video_ids(args.label_csv)
    logger.info(f"Found {len(video_ids)} video_ids in {args.label_csv}")
    if not video_ids:
        logger.warning(f"No video_ids in {args.label_csv}; nothing to do.")
        return

    protagonist_tracks = load_protagonist_tracks(args.protagonist_csv)

    device_id = args.gpu_id if args.gpu_id >= 0 else -1
    logger.info(f"Loading EmotiEffNet ({args.model_name}) on gpu={device_id}...")
    extractor = EmotiEffNetWrapper(model_name=args.model_name, device_id=device_id)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    n_skipped = 0
    n_failed = 0
    n_no_track = 0
    for video_id in tqdm(video_ids, desc="EmotiEffNet features"):
        output_path = args.output_dir / f"{video_id}.st"

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
            dense = extractor.track_to_feature(
                track,
                batch_size=args.batch_size,
                densify=True,
                end_frame_id=num_frames,
            )
            save_file(dense, str(output_path))
        except Exception as e:
            logger.exception(f"{video_id}: {e}")
            n_failed += 1

    logger.info(
        f"Done. output_dir={args.output_dir}, skipped={n_skipped}, "
        f"no_track={n_no_track}, failed={n_failed}"
    )


if __name__ == "__main__":
    main()
