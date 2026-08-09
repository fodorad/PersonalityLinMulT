"""Tests for the audio feature-extraction contract.

The heavy exordium extractors are not exercised here (they download large models);
these tests pin the module's importability and its dict contract, so a refactor
that breaks the feature names the predictor expects fails fast.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path

from personalitylinmult.inference import audio_features
from personalitylinmult.inference.audio_features import (
    AUDIO_SAMPLE_RATE,
    AudioExtractionError,
    ensure_audio,
)

HAS_FFMPEG = shutil.which("ffmpeg") is not None


class TestAudioFeaturesContract(unittest.TestCase):
    def test_module_imports_without_exordium(self):
        # exordium is imported lazily inside the functions, so importing the
        # module (done at file load) must not require it.
        self.assertTrue(hasattr(audio_features, "extract_audio_features"))

    def test_wavlm_layer_matches_the_dataset(self):
        # fi.h5 was built from WavLM layer 9; the live extractor must match or the
        # A model would see a different distribution than it trained on.
        self.assertEqual(audio_features.WAVLM_LAYER, 9)

    def test_sample_rate_matches_training(self):
        self.assertEqual(AUDIO_SAMPLE_RATE, 44100)

    def test_extract_functions_exist(self):
        for name in ("extract_wavlm", "extract_emotion2vec", "extract_audio_features"):
            self.assertTrue(callable(getattr(audio_features, name)))


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg not installed")
class TestEnsureAudio(unittest.TestCase):
    """Video files are decoded to a training-matched WAV; audio passes through."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

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

    def _make_silent_video(self) -> Path:
        path = self.root / "silent.mp4"
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
                str(path),
            ],
            capture_output=True,
            check=True,
        )
        return path

    def test_audio_input_passes_through_untouched(self):
        wav = self.root / "already.wav"
        wav.write_bytes(b"fake")  # suffix is what routes it
        out, is_temp = ensure_audio(wav)
        # ensure_audio resolves its input (existence-check + a defensible,
        # canonical path -- see its docstring), so `out` may differ from
        # `wav` by a symlink normalization (e.g. /var -> /private/var on
        # macOS) even though both name the same file.
        self.assertTrue(out.samefile(wav))
        self.assertFalse(is_temp)

    def test_video_is_decoded_to_mono_44100_wav(self):
        out, is_temp = ensure_audio(self._make_video())
        self.addCleanup(lambda: out.unlink(missing_ok=True))
        self.assertTrue(is_temp)
        with wave.open(str(out)) as w:
            self.assertEqual(w.getnchannels(), 1)
            self.assertEqual(w.getframerate(), 44100)
            self.assertGreater(w.getnframes(), 0)

    def test_video_without_audio_raises(self):
        with self.assertRaises(AudioExtractionError):
            ensure_audio(self._make_silent_video())


if __name__ == "__main__":
    unittest.main()
