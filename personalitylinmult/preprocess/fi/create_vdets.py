"""Detect faces in FI clips and write one ``{uuid}.vdet`` file per video.

Globs ``*.mp4`` from a video directory, runs the YOLO11 face detector on each
clip and stores the per-frame detections as a
:class:`exordium.video.core.detection.VideoDetections` CSV.  Clips that cannot
be processed (e.g. audio-only files with no video stream) are appended to a
failure CSV instead of aborting the run.

All input/output locations default to the burnt-in ``data/...`` layout but can
be overridden from the command line.
"""

import argparse
import random
from pathlib import Path

from exordium.video.face.detector.yolo11 import YoloFace11Detector
from tqdm import tqdm

DB_PROCESSED = Path("data/processed/FI")
"""Default processed FI root (holds ``vdets/`` and the failure log)."""

DEFAULT_VIDEO_DIR = DB_PROCESSED / "video"
"""Default directory of input ``*.mp4`` clips."""

DEFAULT_VDET_DIR = DB_PROCESSED / "vdet"
"""Default output directory for per-clip ``{uuid}.vdet`` files."""

DEFAULT_FAILED_CSV = DB_PROCESSED / "failed_no_vdets.csv"
"""Default CSV logging ``{uuid},{error}`` for clips that could not be processed."""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments (paths default to the burnt-in layout)."""
    parser = argparse.ArgumentParser(description="Script to detect FI faces.")
    parser.add_argument(
        "--video_dir",
        type=Path,
        default=DEFAULT_VIDEO_DIR,
        help="Directory of input .mp4 clips.",
    )
    parser.add_argument(
        "--vdet_dir",
        type=Path,
        default=DEFAULT_VDET_DIR,
        help="Output directory for per-clip .vdet files.",
    )
    parser.add_argument(
        "--failed_csv",
        type=Path,
        default=DEFAULT_FAILED_CSV,
        help="CSV logging clips that failed face detection.",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="ID of the GPU to use")
    parser.add_argument(
        "--conf", type=float, default=0.25, help="YOLO face-detector confidence threshold"
    )
    parser.add_argument("--start", type=int, default=0, help="start index")
    parser.add_argument("--end", type=int, default=25000, help="end index")
    return parser.parse_args()


def main() -> None:
    """Run face detection over the video directory and write ``.vdet`` files."""
    args = parse_args()

    args.vdet_dir.mkdir(parents=True, exist_ok=True)
    args.failed_csv.parent.mkdir(parents=True, exist_ok=True)

    # get video paths as mp4 files in the video dir (skip macOS ._ resource forks)
    video_paths = [elem for elem in args.video_dir.glob("*.mp4") if not elem.stem.startswith("._")][
        args.start : args.end
    ]
    random.shuffle(video_paths)
    print(f"Found {len(video_paths)} videos in {args.video_dir} ({args.start}-{args.end})...")

    print(f"Using GPU ID: {args.gpu_id}")
    detector = YoloFace11Detector(device_id=args.gpu_id, conf=args.conf)

    for video_path in tqdm(video_paths, total=len(video_paths), desc="FI face detection"):
        try:
            # detect_video streams frames in batches and returns VideoDetections
            detector.detect_video(video_path, output_path=args.vdet_dir / f"{video_path.stem}.vdet")
        except Exception as e:
            # log the id of the video that failed in a csv with the error message
            with open(args.failed_csv, "a") as f:
                f.write(f"{video_path.stem},{str(e)}\n")
            print(f"Failed to process video {video_path.stem}: {e}")
            continue


if __name__ == "__main__":
    main()
