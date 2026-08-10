"""Create face tracks from FI vdet files and select the protagonist.

For every sample in ``data/processed/FI/label.csv`` the script:

1. Loads per-frame face detections from ``vdets/{uuid}.vdet`` (a
   :class:`exordium.video.core.detection.VideoDetections` CSV).
2. Runs a :class:`exordium.video.core.detection.FaceClusterTracker` — the
   offline IoU-tracklet tracker with global AdaFace face-ID clustering
   introduced in exordium 2.5.  Plain IoU tracking fragments a single
   person into many short tracklets across occlusions, scene cuts, blur,
   and brief exits; the clustering stage re-identifies and merges those
   fragments into stable identity tracks, so the protagonist survives as
   one track for the whole clip.
3. Saves every resulting identity track to
   ``tracks/{uuid}__track{track_id}.csv``.
4. Selects the dominant identity (on-screen time weighted by face size)
   as the protagonist and appends ``(uuid, protagonist_track_id)`` to
   ``protagonist_track_heuristic.csv``.  Clips with no usable track get
   ``-1``.

This file is the *immutable* heuristic layer.  Manual QA corrections are
recorded separately (``protagonist_track_review.csv``) and merged into the
final ``protagonist_track.csv`` by
:mod:`emotionlinmult.preprocess.FI.protagonist_review`, so re-running this
script never clobbers manual annotation work.

The clustering stage embeds face crops with AdaFace and therefore needs
the AdaFace weights (downloaded on first use) and benefits from a GPU;
pass ``--gpu-id -1`` to run on CPU.

The script is resumable: on restart it skips any ``uuid`` already present
in ``protagonist_track_heuristic.csv``.
"""

import argparse
import csv
import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from exordium.video.core.detection import (
    FaceClusterTracker,
    VideoDetections,
)
from exordium.video.deep.adaface import AdaFaceWrapper
from tqdm import tqdm

logger = logging.getLogger(__name__)
"""Module-level logger."""

DB_PROCESSED = Path("data/processed/FI")
"""Processed FI root (contains ``vdets/`` and will hold ``tracks/``)."""

DEFAULT_LABELS = DB_PROCESSED / "label.csv"
"""Canonical sample list (``video_id`` is the clip id column)."""

DEFAULT_VDET_DIR = DB_PROCESSED / "vdet"
"""Input directory of per-clip ``.vdet`` CSVs."""

DEFAULT_TRACK_DIR = DB_PROCESSED / "track"
"""Output directory for per-clip ``{uuid}__track{track_id}.csv`` files."""

DEFAULT_PROTAGONIST_CSV = DB_PROCESSED / "protagonist_track_heuristic.csv"
"""Destination for the heuristic ``(uuid, track_id)`` protagonist mapping."""


def save_all_tracks(tracker: FaceClusterTracker, uuid: str, track_dir: Path) -> None:
    """Write every non-empty identity track to ``{uuid}__track{track_id}.csv``."""
    track_dir.mkdir(parents=True, exist_ok=True)
    for tid, track in tracker.tracks.items():
        if len(track) == 0:
            continue
        track.save(track_dir / f"{uuid}__track{tid}.csv")


def load_processed_uuids(csv_path: Path) -> set[str]:
    """Return the set of uuids already recorded in ``protagonist_track_heuristic.csv``."""
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    return set(df["uuid"].astype(str).tolist())


def build_tracker_factory(
    encoder: AdaFaceWrapper,
    iou_threshold: float,
    max_lost: int,
    distance_threshold: float,
    samples_per_track: int,
    min_det_score: float,
    min_bb_size: int,
) -> Callable[[], FaceClusterTracker]:
    """Return a zero-arg callable that produces a fresh tracker per clip.

    The AdaFace ``encoder`` is instantiated once and shared across all
    tracker instances (single model load, single GPU handle); only the
    per-clip tracker state is rebuilt on each call.
    """
    return lambda: FaceClusterTracker(
        encoder=encoder,
        verbose=False,
        iou_threshold=iou_threshold,
        max_lost=max_lost,
        distance_threshold=distance_threshold,
        samples_per_track=samples_per_track,
        min_det_score=min_det_score,
        min_bb_size=min_bb_size,
    )


def process_one(
    uuid: str,
    vdet_path: Path,
    make_tracker: Callable[[], FaceClusterTracker],
    min_score: float,
    track_dir: Path,
) -> int:
    """Build identity tracks for a single clip and return the protagonist track id.

    Returns ``-1`` for missing vdets or clips where no face is tracked.
    Saves every identity track produced to ``track_dir`` as a side effect
    so downstream QA can override the heuristic choice.
    """
    if not vdet_path.exists():
        logger.warning(f"{uuid}: missing vdet {vdet_path}")
        return -1

    vdets = VideoDetections().load(vdet_path)
    if len(vdets) == 0:
        return -1

    tracker = make_tracker()
    # run() chains label (IoU tracklets) -> cluster (face-ID) -> select_protagonist.
    protagonist = tracker.run(vdets, min_score=min_score)

    if protagonist is None or not tracker.tracks:
        return -1

    save_all_tracks(tracker, uuid, track_dir)
    logger.debug(
        f"{uuid}: {tracker.n_tracklets} tracklets -> {tracker.n_identities} identities; "
        f"protagonist=track{protagonist.track_id} ({len(protagonist)} detections)"
    )
    return int(protagonist.track_id)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--vdet-dir", type=Path, default=DEFAULT_VDET_DIR)
    parser.add_argument("--track-dir", type=Path, default=DEFAULT_TRACK_DIR)
    parser.add_argument("--protagonist-csv", type=Path, default=DEFAULT_PROTAGONIST_CSV)
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU index for the AdaFace encoder (negative = CPU).",
    )
    parser.add_argument(
        "--backbone",
        default="ir_50",
        choices=["ir_18", "ir_50", "ir_101"],
        help="AdaFace backbone (ir_50 = balanced quality/speed).",
    )
    parser.add_argument(
        "--min-score", type=float, default=0.5, help="Minimum detection confidence to track."
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        help="Minimum IoU to continue an IoU tracklet in the labelling stage.",
    )
    parser.add_argument(
        "--max-lost",
        type=int,
        default=10,
        help="Maximum frame gap to keep an IoU tracklet open (-1 = unlimited).",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.4,
        help="Cosine-distance cut for face-ID clustering (lower = stricter identity merging).",
    )
    parser.add_argument(
        "--samples-per-track",
        type=int,
        default=5,
        help="Number of high-quality detections averaged into each tracklet embedding.",
    )
    parser.add_argument(
        "--min-det-score",
        type=float,
        default=0.5,
        help="Minimum detection score for a crop to feed a tracklet embedding.",
    )
    parser.add_argument(
        "--min-bb-size",
        type=int,
        default=0,
        help="Minimum bounding-box side (px) for a crop to feed a tracklet embedding.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Reprocess uuids already present in --protagonist-csv.",
    )
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    labels = pd.read_csv(args.labels_csv)
    logger.info(f"Loaded {len(labels)} labels from {args.labels_csv}")

    processed = set() if args.no_resume else load_processed_uuids(args.protagonist_csv)
    if processed:
        logger.info(f"Resuming: {len(processed)} uuids already processed")

    logger.info(f"Loading AdaFace ({args.backbone}) on gpu={args.gpu_id}...")
    encoder = AdaFaceWrapper(
        backbone=args.backbone,
        device_id=args.gpu_id if args.gpu_id >= 0 else None,
    )
    make_tracker = build_tracker_factory(
        encoder=encoder,
        iou_threshold=args.iou_threshold,
        max_lost=args.max_lost,
        distance_threshold=args.distance_threshold,
        samples_per_track=args.samples_per_track,
        min_det_score=args.min_det_score,
        min_bb_size=args.min_bb_size,
    )
    logger.info("Tracker: FaceClusterTracker (IoU tracklets + AdaFace face-ID clustering)")

    args.protagonist_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.protagonist_csv.exists()

    with args.protagonist_csv.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["uuid", "track_id"])
            f.flush()

        for _, row in tqdm(labels.iterrows(), total=len(labels), desc="Tracks"):
            uuid = str(row["video_id"])
            if uuid in processed:
                continue
            vdet_path = args.vdet_dir / f"{uuid}.vdet"
            try:
                track_id = process_one(
                    uuid=uuid,
                    vdet_path=vdet_path,
                    make_tracker=make_tracker,
                    min_score=args.min_score,
                    track_dir=args.track_dir,
                )
            except Exception as e:
                logger.exception(f"{uuid}: {e}")
                track_id = -1
            writer.writerow([uuid, track_id])
            f.flush()

    logger.info(f"Done. Wrote {args.protagonist_csv}")


if __name__ == "__main__":
    main()
