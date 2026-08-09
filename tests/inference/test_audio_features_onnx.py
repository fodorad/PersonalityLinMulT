"""Tests for the ONNX-only audio decode path.

The real WavLM ONNX export/session is not exercised here (downloads a large
model and traces it, similar to why test_audio_features.py skips the real
exordium extractors) -- these tests pin the decode contract (ffmpeg -> mono
16 kHz PCM WAV -> float32 array) that extract_wavlm_onnx builds on.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from personalitylinmult.inference import audio_features_onnx
from personalitylinmult.inference.audio_features_onnx import (
    WAVLM_ONNX_SAMPLE_RATE,
    AudioDecodeError,
    load_waveform_16k,
)

HAS_FFMPEG = shutil.which("ffmpeg") is not None


class TestAudioFeaturesOnnxContract(unittest.TestCase):
    def test_module_imports_without_onnxruntime(self):
        # onnxruntime is imported lazily inside extract_wavlm_onnx, so
        # importing the module (done at file load) must not require it.
        self.assertTrue(hasattr(audio_features_onnx, "extract_wavlm_onnx"))
        self.assertTrue(hasattr(audio_features_onnx, "extract_emotion2vec_onnx"))

    def test_sample_rate_matches_wavlm(self):
        self.assertEqual(WAVLM_ONNX_SAMPLE_RATE, 16000)


class _FakeSession:
    """Returns a deterministic, fixed-shape array under a named output key,
    so the decode->session->reshape plumbing can be tested without a real
    ONNX Runtime session."""

    def __init__(self, output_name: str, t: int = 20, dim: int = 8):
        self._output_name = output_name
        self._t = t
        self._dim = dim

    def run(self, output_names, inputs):
        del output_names, inputs
        return [np.random.randn(1, self._t, self._dim).astype("float32")]


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg not installed")
class TestExtractEmotion2vecOnnx(unittest.TestCase):
    def test_extracts_features_with_the_right_shape(self):
        from personalitylinmult.inference.audio_features_onnx import extract_emotion2vec_onnx

        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "clip.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", str(clip)],
                capture_output=True,
                check=True,
            )
            session = _FakeSession("features", t=20, dim=8)
            features = extract_emotion2vec_onnx(clip, session)

        self.assertEqual(features.shape, (20, 8))
        self.assertEqual(features.dtype, np.float32)


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg not installed")
class TestLoadWaveform16k(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _make_audio(self, sample_rate: int = 44100) -> Path:
        path = self.root / "clip.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=440:duration=1:sample_rate={sample_rate}",
                str(path),
            ],
            capture_output=True,
            check=True,
        )
        return path

    def _make_video(self) -> Path:
        path = self.root / "clip.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=1:size=64x64:rate=10",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=1",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-shortest",
                str(path),
            ],
            capture_output=True,
            check=True,
        )
        return path

    def test_resamples_audio_to_16k_mono_float32(self):
        waveform = load_waveform_16k(self._make_audio(sample_rate=44100))
        self.assertEqual(waveform.dtype, np.float32)
        self.assertEqual(waveform.ndim, 1)
        # ~1 second at 16 kHz.
        self.assertAlmostEqual(len(waveform) / WAVLM_ONNX_SAMPLE_RATE, 1.0, delta=0.05)
        self.assertLessEqual(np.abs(waveform).max(), 1.0)

    def test_decodes_video_to_16k_mono(self):
        waveform = load_waveform_16k(self._make_video())
        self.assertEqual(waveform.dtype, np.float32)
        self.assertGreater(len(waveform), 0)

    def test_missing_audio_stream_raises(self):
        silent = self.root / "silent.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=1:size=64x64:rate=10",
                "-c:v",
                "libx264",
                str(silent),
            ],
            capture_output=True,
            check=True,
        )
        with self.assertRaises(AudioDecodeError):
            load_waveform_16k(silent)

    def test_missing_file_raises(self):
        with self.assertRaises(AudioDecodeError):
            load_waveform_16k(self.root / "nope.wav")


if __name__ == "__main__":
    unittest.main()
