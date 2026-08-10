"""Prepare the First Impressions (FI) dataset for the processing pipeline.

Reads videos from ``train/``, ``valid/``, ``test/`` directories and
annotations/transcripts from ``gt/`` pickle files under the raw FI dataset
root. For each video:
    - Re-encodes to 25 fps MP4 (libx264/aac) into the output ``video/``
      directory (only if the source has a video stream)
    - Extracts audio to mono WAV at 44100 Hz into the output ``audio/``
      directory (only if the source has an audio stream)

Produces ``label.csv`` with columns:
    video_id, subset, openness, conscientiousness, extraversion,
    agreeableness, neuroticism, emotional_stability, interview, text

Note: the raw ChaLearn FI pickles store their ``"neuroticism"`` key as
``1 - original_neuroticism`` -- the authors already inverted it so all six
fields (OCEAN + interview) point the same positive direction, but kept the old
key name instead of renaming it. Its values are positively correlated with the
other traits and with ``interview``, the signature of emotional stability, not
neuroticism (verified: raw ``"neuroticism"`` correlates +0.72 to +0.88 with
every other field across all 6000 training clips). ``emotional_stability`` is
therefore read directly from the pickle's ``"neuroticism"`` field, and
``neuroticism`` here is the genuine, derived ``1 - emotional_stability``.
"""

import argparse
import csv
import json
import logging
import pickle
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path("data/raw/FI")
"""Default root of the original, untouched FI dataset."""

DEFAULT_OUTPUT_DIR = Path("data/processed/FI")
"""Default root for re-encoded video/audio and label.csv."""

OCEAN_TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]


def build_subsets(raw_dir: Path) -> dict[str, dict[str, Path]]:
    """Build the per-subset source paths under the raw FI dataset root.

    Args:
        raw_dir: Root of the raw FI dataset (contains train/, valid/, test/, gt/).

    Returns:
        Mapping of subset name to its video directory, annotation pickle, and
        transcription pickle paths.
    """
    return {
        "train": {
            "video_dir": raw_dir / "train",
            "annotation": raw_dir / "gt" / "annotation_training.pkl",
            "transcription": raw_dir / "gt" / "transcription_training.pkl",
        },
        "valid": {
            "video_dir": raw_dir / "valid",
            "annotation": raw_dir / "gt" / "annotation_validation.pkl",
            "transcription": raw_dir / "gt" / "transcription_validation.pkl",
        },
        "test": {
            "video_dir": raw_dir / "test",
            "annotation": raw_dir / "gt" / "annotation_test.pkl",
            "transcription": raw_dir / "gt" / "transcription_test.pkl",
        },
    }


def load_annotations(pkl_path: Path) -> dict[str, dict[str, float]]:
    """Load annotation pickle and return {video_id: {trait: value}}.

    Args:
        pkl_path: Path to the annotation pickle file.

    Returns:
        Dictionary mapping video filenames to their OCEAN + interview values.
    """
    with open(pkl_path, "rb") as f:
        ann = pickle.load(f, encoding="latin1")  # noqa: S301

    result: dict[str, dict[str, float]] = {}
    all_keys = set()
    for trait in OCEAN_TRAITS + ["interview"]:
        all_keys.update(ann[trait].keys())

    for vid in all_keys:
        result[vid] = {}
        for trait in OCEAN_TRAITS + ["interview"]:
            result[vid][trait] = float(ann[trait].get(vid, 0.0))
    return result


def load_transcriptions(pkl_path: Path) -> dict[str, str]:
    """Load transcription pickle and return {video_id: text}.

    Args:
        pkl_path: Path to the transcription pickle file.

    Returns:
        Dictionary mapping video filenames to their transcript text.
    """
    with open(pkl_path, "rb") as f:
        trans = pickle.load(f, encoding="latin1")  # noqa: S301
    return {k: str(v).strip() for k, v in trans.items()}


def probe_streams(video_path: Path) -> tuple[bool, bool]:
    """Check whether a video file has video and audio streams.

    Args:
        video_path: Path to the video file.

    Returns:
        Tuple of (has_video, has_audio).
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "json",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return False, False
        info = json.loads(result.stdout)
        streams = info.get("streams", [])
        codec_types = [s.get("codec_type") for s in streams]
        return "video" in codec_types, "audio" in codec_types
    except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError):
        return False, False


def convert_video(input_path: Path, output_path: Path) -> bool:
    """Re-encode video to 25 fps MP4.

    Args:
        input_path: Source video path.
        output_path: Destination MP4 path.

    Returns:
        True if conversion succeeded.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-r",
        "25",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        str(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0
    except subprocess.TimeoutExpired:
        return False


def convert_audio(input_path: Path, output_path: Path) -> bool:
    """Extract audio to mono WAV at 44100 Hz.

    Args:
        input_path: Source video path.
        output_path: Destination WAV path.

    Returns:
        True if conversion succeeded.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "44100",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0
    except subprocess.TimeoutExpired:
        return False


def validate_video(video_path: Path) -> bool:
    """Check that a video file can be read and at least one frame decoded.

    Args:
        video_path: Path to the video file.

    Returns:
        True if at least one frame can be decoded.
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def validate_audio(audio_path: Path) -> bool:
    """Check that a WAV file has valid audio.

    Args:
        audio_path: Path to the WAV file.

    Returns:
        True if the audio has a positive duration.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return False
        dur = result.stdout.strip()
        return bool(dur) and float(dur) > 0
    except (subprocess.TimeoutExpired, ValueError):
        return False


def _load_all_records(subsets: dict[str, dict[str, Path]]) -> list[dict]:
    """Load every subset's annotations and transcriptions into flat records.

    Args:
        subsets: ``{subset_name: {"annotation": ..., "transcription": ..., "video_dir": ...}}``.

    Returns:
        One record per annotated clip, across every subset.
    """
    all_records: list[dict] = []
    for subset_name, paths in subsets.items():
        logger.info("Loading %s annotations and transcriptions...", subset_name)
        annotations = load_annotations(paths["annotation"])
        transcriptions = load_transcriptions(paths["transcription"])

        for vid_filename, traits in annotations.items():
            video_path = paths["video_dir"] / vid_filename
            text = transcriptions.get(vid_filename, "")
            video_id = vid_filename.replace(".mp4", "")

            all_records.append(
                {
                    "video_id": video_id,
                    "subset": subset_name,
                    "video_path": video_path,
                    "traits": traits,
                    "text": text,
                }
            )
    return all_records


def _ensure_media_available(
    source_path: Path,
    out_path: Path,
    convert_fn,
    validate_fn,
) -> bool:
    """Ensure one re-encoded media file exists and is valid, (re)converting if not.

    Shared by both the video and audio branches of :func:`_process_record`
    -- identical logic, only ``convert_fn``/``validate_fn`` differ.

    Args:
        source_path: The raw source file to convert from.
        out_path: The target re-encoded file.
        convert_fn: ``(source_path, out_path) -> bool`` (``convert_video``
            or ``convert_audio``).
        validate_fn: ``(out_path) -> bool`` (``validate_video`` or
            ``validate_audio``).

    Returns:
        Whether ``out_path`` now exists and is valid.
    """
    if out_path.exists() and out_path.stat().st_size > 0:
        if validate_fn(out_path):
            return True
        out_path.unlink()

    if convert_fn(source_path, out_path):
        if validate_fn(out_path):
            return True
        out_path.unlink(missing_ok=True)
    else:
        out_path.unlink(missing_ok=True)
    return False


def _process_record(
    rec: dict,
    video_out_dir: Path,
    audio_out_dir: Path,
    labels_only: bool,
    stats: dict[str, int],
) -> list[str]:
    """Ensure one record's video/audio are available, and build its CSV row.

    Args:
        rec: One record from :func:`_load_all_records`.
        video_out_dir: Directory for re-encoded videos.
        audio_out_dir: Directory for re-encoded audio.
        labels_only: If True, skip ffprobe/ffmpeg entirely and just check
            whether an output file already exists and is non-empty.
        stats: Mutated in place: ``total``/``video_ok``/``audio_ok``/``missing_source``.

    Returns:
        The label CSV row for this record.
    """
    video_id = rec["video_id"]
    source_path = rec["video_path"]
    video_out = video_out_dir / f"{video_id}.mp4"
    audio_out = audio_out_dir / f"{video_id}.wav"
    traits = rec["traits"]

    video_available = False
    audio_available = False

    if labels_only:
        video_available = video_out.exists() and video_out.stat().st_size > 0
        audio_available = audio_out.exists() and audio_out.stat().st_size > 0
    elif not source_path.exists():
        stats["missing_source"] += 1
    else:
        has_video_stream, has_audio_stream = probe_streams(source_path)
        if has_video_stream:
            video_available = _ensure_media_available(
                source_path, video_out, convert_video, validate_video
            )
        if has_audio_stream:
            audio_available = _ensure_media_available(
                source_path, audio_out, convert_audio, validate_audio
            )

    stats["total"] += 1
    stats["video_ok"] += int(video_available)
    stats["audio_ok"] += int(audio_available)

    # The pickle's "neuroticism" key is misnamed: ChaLearn already computed
    # 1 - original_neuroticism so all six fields point the same positive
    # direction, but kept the old key name instead of renaming it. Use it
    # directly as emotional_stability; derive genuine neuroticism as its
    # complement for traceability.
    emotional_stability = traits["neuroticism"]
    neuroticism = 1.0 - emotional_stability

    return [
        video_id,
        rec["subset"],
        f"{traits['openness']:.6f}",
        f"{traits['conscientiousness']:.6f}",
        f"{traits['extraversion']:.6f}",
        f"{traits['agreeableness']:.6f}",
        f"{neuroticism:.6f}",
        f"{emotional_stability:.6f}",
        f"{traits['interview']:.6f}",
        rec["text"],
    ]


def _write_label_csv(label_csv: Path, rows: list[list[str]]) -> None:
    """Write the label CSV with its fixed header.

    Args:
        label_csv: Output path.
        rows: One row per record, in :func:`_process_record`'s column order.
    """
    header = [
        "video_id",
        "subset",
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
        "emotional_stability",
        "interview",
        "text",
    ]
    with open(label_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main(raw_dir: Path, output_dir: Path, labels_only: bool = False) -> None:
    """Run the FI preparation pipeline.

    Args:
        raw_dir: Root of the raw FI dataset (contains train/, valid/, test/, gt/).
        output_dir: Root for re-encoded video/audio and label.csv.
        labels_only: If True, skip ffprobe/ffmpeg conversion and validation
            entirely and just rebuild label.csv, treating an existing
            non-empty output file as available.
    """
    video_out_dir = output_dir / "video"
    audio_out_dir = output_dir / "audio"
    label_csv = output_dir / "label.csv"
    video_out_dir.mkdir(parents=True, exist_ok=True)
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    subsets = build_subsets(raw_dir)
    all_records = _load_all_records(subsets)
    logger.info("Total records: %d", len(all_records))

    output_rows: list[list[str]] = []
    stats = {"total": 0, "video_ok": 0, "audio_ok": 0, "missing_source": 0}

    for rec in tqdm(all_records, desc="Processing FI", unit="vid"):
        output_rows.append(_process_record(rec, video_out_dir, audio_out_dir, labels_only, stats))

    _write_label_csv(label_csv, output_rows)

    logger.info("Label CSV written to %s", label_csv)
    logger.info(
        "Statistics: total=%d, video_ok=%d, audio_ok=%d, missing_source=%d",
        stats["total"],
        stats["video_ok"],
        stats["audio_ok"],
        stats["missing_source"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Root of the raw FI dataset (contains train/, valid/, test/, gt/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root for re-encoded video/audio and label.csv.",
    )
    parser.add_argument(
        "--labels-only",
        action="store_true",
        help="Skip ffprobe/ffmpeg conversion and validation; only rebuild label.csv.",
    )
    args = parser.parse_args()
    main(raw_dir=args.raw_dir, output_dir=args.output_dir, labels_only=args.labels_only)
