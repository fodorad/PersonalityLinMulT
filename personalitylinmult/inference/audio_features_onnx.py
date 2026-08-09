"""Extract WavLM features via ONNX Runtime -- no ``torch``/``transformers``.

The ONNX Runtime-backed counterpart to
:func:`~personalitylinmult.inference.audio_features.extract_wavlm`. Decoding
still shells out to ``ffmpeg`` (a subprocess, not a Python dependency) to
produce mono 16 kHz PCM WAV directly -- WavLM's own required sample rate --
avoiding ``torchaudio``, which would otherwise pull ``torch`` back in just for
resampling. Reading the WAV uses only the standard library's :mod:`wave`
module plus ``numpy``, both already required.

**Known limitation**: ffmpeg's resampler and ``torchaudio``'s
``Resample`` transform (used by the PyTorch path, via
:class:`exordium.audio.wavlm.WavlmWrapper`) use different filter
implementations, so they do not produce bit-identical 16 kHz waveforms from
the same source file (~3e-4 mean, ~2.5e-2 max absolute sample difference,
measured on a real clip). WavLM's transformer layers can amplify this at a
handful of frames, but the final Big Five predictions shift only ~0.001-0.004
per trait end to end -- small and bounded, not exact. Accepted deliberately
to keep this module genuinely ``torch``-free.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import onnxruntime as ort

logger = logging.getLogger(__name__)
"""Module-level logger."""

WAVLM_ONNX_SAMPLE_RATE = 16000
"""WavLM's required input sample rate, matching ``exordium``'s
``WAVLM_SAMPLE_RATE`` -- see ``onnx_export.py``'s ``WAVLM_MODEL_ID``."""


class AudioDecodeError(Exception):
    """Raised when ffmpeg cannot decode the input into a WAV."""


def _decode_to_wav_16k_mono(input_path: str | Path) -> tuple[Path, bool]:
    """Decode any audio/video file to mono 16 kHz PCM WAV via ffmpeg.

    Args:
        input_path (str | Path): An audio or video file.

    Returns:
        tuple[Path, bool]: ``(wav_path, is_temporary)`` -- the caller deletes
        the WAV when ``is_temporary`` is True. An input that is already a
        16 kHz mono WAV is still re-encoded (correctness over the rare
        micro-optimization of skipping ffmpeg for an already-matching file).

    Raises:
        AudioDecodeError: If ffmpeg is missing, times out, or the input has
            no audio stream.
    """
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
        str(WAVLM_ONNX_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    logger.info(f"Decoding {input_path} -> {wav_path} (16 kHz mono, for ONNX WavLM).")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except FileNotFoundError as exc:
        wav_path.unlink(missing_ok=True)
        raise AudioDecodeError("ffmpeg is not installed or not on PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        wav_path.unlink(missing_ok=True)
        raise AudioDecodeError(f"ffmpeg timed out decoding {input_path}.") from exc
    if result.returncode != 0 or not wav_path.exists() or wav_path.stat().st_size == 0:
        wav_path.unlink(missing_ok=True)
        raise AudioDecodeError(
            f"ffmpeg could not decode {input_path} "
            f"(the file may have no audio stream). stderr: {result.stderr[-500:]}"
        )
    return wav_path, True


def _read_wav_mono_float32(wav_path: str | Path) -> np.ndarray:
    """Read a mono 16-bit PCM WAV into a ``[-1, 1]`` float32 array.

    Args:
        wav_path (str | Path): Path to a mono, 16-bit PCM WAV file (the
            format :func:`_decode_to_wav_16k_mono` always produces).

    Returns:
        np.ndarray: ``(samples,)`` float32 waveform in ``[-1, 1]``.

    Raises:
        AudioDecodeError: If the file is not mono 16-bit PCM.
    """
    with wave.open(str(wav_path), "rb") as f:
        if f.getnchannels() != 1 or f.getsampwidth() != 2:
            raise AudioDecodeError(
                f"{wav_path}: expected mono 16-bit PCM, got "
                f"{f.getnchannels()} channel(s), {f.getsampwidth() * 8}-bit."
            )
        frames = f.readframes(f.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype("float32") / 32768.0


def load_waveform_16k(input_path: str | Path) -> np.ndarray:
    """Load any audio/video file as a mono 16 kHz waveform, via ffmpeg.

    Args:
        input_path (str | Path): An audio or video file.

    Returns:
        np.ndarray: ``(samples,)`` float32 waveform in ``[-1, 1]``, at
        :data:`WAVLM_ONNX_SAMPLE_RATE`.
    """
    wav_path, is_temp = _decode_to_wav_16k_mono(input_path)
    try:
        return _read_wav_mono_float32(wav_path)
    finally:
        if is_temp:
            wav_path.unlink(missing_ok=True)


def extract_wavlm_onnx(input_path: str | Path, session: ort.InferenceSession) -> np.ndarray:
    """Extract WavLM layer features using an ONNX Runtime session.

    Args:
        input_path (str | Path): An audio or video file.
        session (onnxruntime.InferenceSession): A session over the graph
            :func:`~personalitylinmult.inference.onnx_export.export_wavlm_to_onnx`
            produced.

    Returns:
        np.ndarray: ``(T, 768)`` float32 features, matching
        :func:`~personalitylinmult.inference.audio_features.extract_wavlm`'s
        output for the same input.
    """
    waveform = load_waveform_16k(input_path)
    input_values = waveform[np.newaxis, :]  # (1, samples)
    (hidden_state,) = session.run(["hidden_state"], {"input_values": input_values})
    array: np.ndarray = np.asarray(hidden_state)
    return array[0]  # (T, 768)


def extract_emotion2vec_onnx(input_path: str | Path, session: ort.InferenceSession) -> np.ndarray:
    """Extract emotion2vec+ features using an ONNX Runtime session.

    Reuses :func:`load_waveform_16k` -- emotion2vec+, like WavLM, expects a
    16 kHz mono waveform, so the same decode step covers both.

    Args:
        input_path (str | Path): An audio or video file.
        session (onnxruntime.InferenceSession): A session over the graph
            :func:`~personalitylinmult.inference.onnx_export.export_emotion2vec_to_onnx`
            produced.

    Returns:
        np.ndarray: ``(T, 768)`` float32 features, matching
        ``exordium.audio.emotion2vec.Emotion2vecWrapper``'s output for the
        same input.
    """
    waveform = load_waveform_16k(input_path)
    source = waveform[np.newaxis, :]  # (1, samples)
    (features,) = session.run(["features"], {"source": source})
    array: np.ndarray = np.asarray(features)
    return array[0]  # (T, 768)
