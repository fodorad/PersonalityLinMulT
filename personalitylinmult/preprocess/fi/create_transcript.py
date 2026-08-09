"""Predict speech transcripts for FI audio clips with whisperX.

Iterates over the per-clip ``*.wav`` files in ``data/processed/FI/audio`` and
runs exordium's :class:`~exordium.text.pipeline.SpeechAlignmentPipeline`
(``backend="whisperx"``) to transcribe each clip.

Two outputs are supported:

1. **Plain transcript CSV** (default) — a single
   ``transcript_whisperx.csv`` with columns ``video_id,text`` where
   ``video_id`` is the audio filename stem (matching ``label.csv``). This is
   the drop-in companion to the existing ground-truth ``text`` column.
2. **Word-level JSON** (``--save-words``) — additionally writes one
   ``transcript/{video_id}.json`` per clip containing word-level
   ``{text, start, end, score}`` timestamps.

The script is resumable: on restart it skips any ``video_id`` already present
in the output CSV. Use ``--no-resume`` to re-transcribe everything.

whisperX loads a Whisper ASR model (and a wav2vec2 aligner for word
timestamps) once and reuses it across all clips; a GPU is strongly
recommended (``--gpu-id -1`` forces CPU).
"""

import argparse
import csv
import json
import logging
import warnings
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from exordium.text.pipeline import SpeechAlignmentPipeline
from tqdm import tqdm

logger = logging.getLogger(__name__)
"""Module-level logger."""

DB_PROCESSED = Path("data/processed/FI")
"""Processed FI root (contains ``audio/`` and will hold ``transcript/``)."""

DEFAULT_AUDIO_DIR = DB_PROCESSED / "audio"
"""Input directory of per-clip ``{video_id}.wav`` files."""

DEFAULT_TRANSCRIPT_CSV = DB_PROCESSED / "transcript_whisperx.csv"
"""Destination for the ``(video_id, text)`` transcript mapping."""

DEFAULT_WORDS_DIR = DB_PROCESSED / "transcript"
"""Output directory for per-clip word-timestamp JSON (``--save-words``)."""


def load_processed_video_ids(csv_path: Path) -> set[str]:
    """Return the set of ``video_id``s already present in the output CSV."""
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    return set(df["video_id"].astype(str).tolist())


def save_words(words: list, video_id: str, words_dir: Path) -> None:
    """Write word-level timestamps to ``{video_id}.json``.

    Args:
        words: A list of :class:`~exordium.text.base.Word` dataclasses.
        video_id: Clip identifier (audio filename stem).
        words_dir: Destination directory for the JSON file.
    """
    words_dir.mkdir(parents=True, exist_ok=True)
    payload = [asdict(w) for w in words]
    (words_dir / f"{video_id}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def process_one(
    audio_path: Path,
    pipeline: SpeechAlignmentPipeline,
    save_word_timestamps: bool,
    words_dir: Path,
) -> str:
    """Transcribe a single clip and optionally dump its word timestamps.

    Args:
        audio_path: Path to the ``.wav`` clip.
        pipeline: A loaded whisperX pipeline (reused across clips).
        save_word_timestamps: If ``True``, also write per-clip word JSON.
        words_dir: Destination directory for the word JSON.

    Returns:
        The transcribed text (empty string when the clip has no speech).
    """
    if save_word_timestamps:
        words = pipeline.transcribe_words(str(audio_path))
        save_words(words, audio_path.stem, words_dir)
        return " ".join(w.text for w in words).strip()
    return pipeline.transcribe(str(audio_path)).strip()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--transcript-csv", type=Path, default=DEFAULT_TRANSCRIPT_CSV)
    parser.add_argument("--words-dir", type=Path, default=DEFAULT_WORDS_DIR)
    parser.add_argument(
        "--save-words",
        action="store_true",
        help="Also write per-clip word-level timestamp JSON to --words-dir.",
    )
    parser.add_argument("--language", default="en", help="ASR/alignment language code.")
    parser.add_argument(
        "--whisper-model",
        default="distil-whisper/distil-large-v3",
        help="HuggingFace Whisper model id for transcription.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="Device index for whisperX (negative = CPU).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-transcribe video_ids already present in --transcript-csv.",
    )
    args = parser.parse_args()

    # whisperX and its deps (httpx, huggingface_hub, transformers) are extremely
    # chatty, and whisperX's alignment emits a pandas chained-assignment
    # FutureWarning on every clip. Keep the root at WARNING so only our own INFO
    # lines show, drop the noisiest loggers to ERROR, and mute those warnings.
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.WARNING)
    logger.setLevel(logging.INFO)
    for _noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "transformers"):
        logging.getLogger(_noisy).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # Skip macOS AppleDouble sidecar files (e.g. "._clip.wav") that appear
    # next to the real clips on non-native filesystems.
    audio_paths = sorted(p for p in args.audio_dir.glob("*.wav") if not p.name.startswith("._"))
    logger.info(f"Found {len(audio_paths)} audio clips in {args.audio_dir}")
    if not audio_paths:
        logger.warning(f"No .wav files under {args.audio_dir}; nothing to do.")
        return

    processed = set() if args.no_resume else load_processed_video_ids(args.transcript_csv)
    if processed:
        logger.info(f"Resuming: {len(processed)} clips already transcribed")

    logger.info(f"Loading whisperX (model={args.whisper_model}) on gpu={args.gpu_id}...")
    pipeline = SpeechAlignmentPipeline(
        backend="whisperx",
        language=args.language,
        whisper_model=args.whisper_model,
        device_id=args.gpu_id if args.gpu_id >= 0 else -1,
    )

    args.transcript_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.transcript_csv.exists()

    with args.transcript_csv.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["video_id", "text"])
            f.flush()

        for audio_path in tqdm(audio_paths, desc="Transcribe"):
            video_id = audio_path.stem
            if video_id in processed:
                continue
            try:
                text = process_one(audio_path, pipeline, args.save_words, args.words_dir)
            except Exception as e:
                logger.exception(f"{video_id}: {e}")
                text = ""
            writer.writerow([video_id, text])
            f.flush()

    logger.info(f"Done. Wrote {args.transcript_csv}")


if __name__ == "__main__":
    main()
