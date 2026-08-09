"""Extract the audio features the A / AT / AVT models consume, from a media file.

The trained models take pre-extracted per-modality features, not raw audio. To
predict on a *new* clip a caller must first run the same extractors used to build
``fi.h5``. This module wraps the two acoustic ones — WavLM and emotion2vec — so a
demo (notebook or Space) can go from an uploaded file to a features dict the
predictor accepts, without importing the batch preprocessing scripts.

exordium (the extractor stack) is imported lazily inside the functions so this
module stays importable where those heavy deps are absent (e.g. a unit-test
environment that only exercises the predictor).
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
"""Module-level logger."""

WAVLM_LAYER = 9
"""WavLM transformer layer used everywhere in this project (matches fi.h5)."""

AUDIO_SUFFIXES = frozenset({".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"})
"""Extensions treated as audio; anything else is decoded through ffmpeg first."""

# The training pipeline extracts audio as mono WAV at 44.1 kHz (preprocess.py's
# convert_audio). The demo must match exactly, or the extractors would see a
# different distribution than the models trained on.
AUDIO_SAMPLE_RATE = 44100
"""Sample rate the training audio was extracted at."""


class AudioExtractionError(Exception):
    """Raised when audio cannot be decoded from an input file."""


def ensure_audio(input_path: str | Path) -> tuple[Path, bool]:
    """Return a WAV path for the input, extracting audio from video if needed.

    Audio files are passed through unchanged; any other container (video) is
    decoded to a temporary mono 44.1 kHz WAV with the same ffmpeg settings the
    training pipeline used.

    Args:
        input_path (str | Path): An audio or video file.

    Returns:
        tuple[Path, bool]: ``(wav_path, is_temporary)`` -- the caller deletes the
        WAV when ``is_temporary`` is True.

    Raises:
        AudioExtractionError: If ffmpeg fails or the input has no audio stream.
    """
    # lgtm[py/path-injection] -- ensure_audio's caller (app.py's Gradio
    # upload handling, deploy/serve.py's tempfile.NamedTemporaryFile) always
    # passes a server-managed path, never a raw request path; this resolves
    # it to a canonical form as defensive validation, not the injection.
    input_path = Path(input_path).resolve()
    # lgtm[py/path-injection] -- the existence check itself, not a use of
    # unvalidated input; see the comment above.
    if not input_path.is_file():
        raise AudioExtractionError(f"Input file does not exist: {input_path}")
    if input_path.suffix.lower() in AUDIO_SUFFIXES:
        return input_path, False

    wav_path = Path(tempfile.mkstemp(suffix=".wav")[1])
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(AUDIO_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    # lgtm[py/log-injection] -- repr(), not str(): input_path ultimately
    # traces back to caller-supplied input (e.g. an uploaded filename), and
    # repr() escapes newlines/control characters so it cannot forge extra
    # log lines.
    logger.info(f"Extracting audio from video {input_path!r} -> {wav_path} (44.1 kHz mono).")
    try:
        # lgtm[py/command-line-injection] -- cmd is a Python list passed to
        # subprocess.run with no shell=True, so no shell ever parses it;
        # input_path enters only as str(input_path), a single argv element
        # ffmpeg receives verbatim, not a place shell metacharacters matter.
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except FileNotFoundError as exc:
        raise AudioExtractionError("ffmpeg is not installed or not on PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioExtractionError(f"ffmpeg timed out decoding {input_path}.") from exc
    if result.returncode != 0 or not wav_path.exists() or wav_path.stat().st_size == 0:
        wav_path.unlink(missing_ok=True)
        raise AudioExtractionError(
            f"ffmpeg could not extract audio from {input_path} "
            f"(the file may have no audio stream). stderr: {result.stderr[-500:]}"
        )
    return wav_path, True


def extract_wavlm(audio_path: str | Path, device_id: int = -1) -> np.ndarray:
    """Extract the WavLM layer-9 feature sequence for one clip.

    Args:
        audio_path (str | Path): Path to an audio/video file exordium can read.
        device_id (int): Device index; ``-1`` for CPU (portable default).

    Returns:
        np.ndarray: ``(T, 768)`` float32 features.
    """
    from exordium.audio.wavlm import WavlmWrapper

    extractor = WavlmWrapper(device_id=device_id, model_name="base+")
    features = extractor.audio_to_feature(Path(audio_path))  # all layers
    layer = features[WAVLM_LAYER]
    return np.asarray(layer, dtype="float32")


def extract_emotion2vec(audio_path: str | Path, device_id: int = -1) -> np.ndarray:
    """Extract the emotion2vec+ feature sequence for one clip.

    Args:
        audio_path (str | Path): Path to an audio/video file exordium can read.
        device_id (int): Device index; ``-1`` for CPU.

    Returns:
        np.ndarray: ``(T, 768)`` float32 features.
    """
    from exordium.audio.emotion2vec import Emotion2vecWrapper

    extractor = Emotion2vecWrapper(device_id=device_id)
    features = extractor.audio_to_feature(Path(audio_path))
    return np.asarray(features, dtype="float32")


def extract_audio_features(audio_path: str | Path, device_id: int = -1) -> dict[str, np.ndarray]:
    """Extract both acoustic features for the audio (A) model.

    Accepts audio *or video*: a video's audio track is extracted to a temporary
    mono 44.1 kHz WAV (matching the training pipeline) before feature extraction,
    then cleaned up.

    Args:
        audio_path (str | Path): Path to an audio or video file.
        device_id (int): Device index; ``-1`` for CPU.

    Returns:
        dict[str, np.ndarray]: ``{"wavlm": (T, 768), "emotion2vec": (T, 768)}``,
        the exact input the ``A`` model's predictor expects.

    Raises:
        AudioExtractionError: If the input is a video with no decodable audio.
    """
    logger.info(f"Extracting acoustic features from {audio_path} (device={device_id})...")
    wav_path, is_temporary = ensure_audio(audio_path)
    try:
        return {
            "wavlm": extract_wavlm(wav_path, device_id),
            "emotion2vec": extract_emotion2vec(wav_path, device_id),
        }
    finally:
        if is_temporary:
            Path(wav_path).unlink(missing_ok=True)
