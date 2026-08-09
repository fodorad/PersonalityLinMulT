"""Tests for the Gradio Space entry point (``app/app.py``).

Lightweight: no real model download or Gradio server. ``_predict_pytorch``/
``_predict_onnx`` are monkeypatched so ``predict_clip``'s validation and
dispatch logic (the actual risk surface -- picking the right model/backend,
propagating errors as ``gr.Error``) is exercised without ever touching
HF Hub, ffmpeg, or a real inference session.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import gradio as gr
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import app as app_module  # noqa: E402

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "emotional_stability"]


def _write_silent_clip(path: Path, duration: float = 1.0) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}", str(path)],
        capture_output=True,
        check=True,
    )


class FakePytorchModel:
    feature_names = ["wavlm"]

    def predict(self, features):
        del features
        return dict.fromkeys(TRAITS, 0.5)


class FakeOnnxModel:
    feature_names = ["wavlm"]

    def predict_from_audio(self, clip_path):
        del clip_path
        return dict.fromkeys(TRAITS, 0.5)


class FakeAvtOnnxModel:
    feature_names = [
        "wavlm",
        "emotion2vec",
        "dinov2",
        "emotieffnet",
        "farl",
        "xlm_roberta_gt",
        "mmbert_gt",
    ]

    def predict_from_media(self, clip_path, transcript=None):
        del clip_path, transcript
        return dict.fromkeys(TRAITS, 0.5)


class TestScoresToPlot(unittest.TestCase):
    def test_returns_a_figure_with_one_bar_per_trait(self):
        scores = dict.fromkeys(TRAITS, 0.5)
        fig = app_module._scores_to_plot(scores)
        ax = fig.axes[0]
        self.assertEqual(len(ax.patches), len(TRAITS))


class TestPredictOnnxBody(unittest.TestCase):
    """Exercises _predict_onnx's real body, not just predict_clip's dispatch."""

    def setUp(self):
        self._orig_load = app_module._load

    def tearDown(self):
        app_module._load = self._orig_load

    def test_predicts_from_a_real_clip_audio_only_champion(self):
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not installed")

        app_module._load = lambda model_id, backend: FakeOnnxModel()
        lines: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "clip.wav"
            _write_silent_clip(clip)
            scores = app_module._predict_onnx("wavlm_best-ccc", str(clip), "", False, lines.append)

        self.assertEqual(set(scores), set(TRAITS))
        self.assertTrue(any("total:" in line for line in lines))

    def test_avt_champion_with_explicit_transcript_calls_predict_from_media(self):
        calls = []

        class RecordingAvtModel(FakeAvtOnnxModel):
            def predict_from_media(self, clip_path, transcript=None):
                calls.append(transcript)
                return dict.fromkeys(TRAITS, 0.5)

        app_module._load = lambda model_id, backend: RecordingAvtModel()
        scores = app_module._predict_onnx(
            "avt_best-ccc", "clip.mp4", "hello world", False, lambda message: None
        )
        self.assertEqual(set(scores), set(TRAITS))
        self.assertEqual(calls, ["hello world"])

    def test_avt_champion_with_skip_text_passes_empty_string(self):
        calls = []

        class RecordingAvtModel(FakeAvtOnnxModel):
            def predict_from_media(self, clip_path, transcript=None):
                calls.append(transcript)
                return dict.fromkeys(TRAITS, 0.5)

        app_module._load = lambda model_id, backend: RecordingAvtModel()
        app_module._predict_onnx("avt_best-ccc", "clip.mp4", "ignored", True, lambda message: None)
        self.assertEqual(calls, [""])

    def test_avt_champion_with_no_transcript_passes_none(self):
        calls = []

        class RecordingAvtModel(FakeAvtOnnxModel):
            def predict_from_media(self, clip_path, transcript=None):
                calls.append(transcript)
                return dict.fromkeys(TRAITS, 0.5)

        app_module._load = lambda model_id, backend: RecordingAvtModel()
        app_module._predict_onnx("avt_best-ccc", "clip.mp4", "", False, lambda message: None)
        self.assertEqual(calls, [None])


class TestPredictPytorchBody(unittest.TestCase):
    """Exercises _predict_pytorch's real body with fake extractors."""

    def setUp(self):
        self._orig_load = app_module._load
        app_module._load = lambda model_id, backend: FakePytorchModel()

    def tearDown(self):
        app_module._load = self._orig_load

    def test_predicts_from_a_real_audio_clip(self):
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not installed")

        import personalitylinmult.inference.audio_features as audio_features

        original_extract_wavlm = audio_features.extract_wavlm
        audio_features.extract_wavlm = lambda path, device_id=-1: np.random.randn(10, 8).astype(
            "float32"
        )
        try:
            lines: list[str] = []
            with tempfile.TemporaryDirectory() as tmp:
                clip = Path(tmp) / "clip.wav"
                _write_silent_clip(clip)
                scores = app_module._predict_pytorch("wavlm_best-ccc", str(clip), lines.append)
            self.assertEqual(set(scores), set(TRAITS))
            self.assertTrue(any("predict:" in line for line in lines))
        finally:
            audio_features.extract_wavlm = original_extract_wavlm

    def test_rejects_a_champion_needing_an_unsupported_feature(self):
        app_module._load = lambda model_id, backend: type("M", (), {"feature_names": ["dinov2"]})()
        with self.assertRaises(gr.Error):
            app_module._predict_pytorch("avt_best-ccc", "clip.wav", lambda message: None)


class TestPredictClipValidation(unittest.TestCase):
    def test_rejects_missing_clip(self):
        with self.assertRaises(gr.Error):
            list(app_module.predict_clip("wavlm_best-ccc", "", "pytorch", "", False))

    def test_rejects_a_nonexistent_clip_path(self):
        with self.assertRaises(gr.Error) as ctx:
            list(
                app_module.predict_clip("wavlm_best-ccc", "/no/such/clip.wav", "pytorch", "", False)
            )
        self.assertIn("could not be found", str(ctx.exception))

    def test_rejects_unknown_model(self):
        with tempfile.NamedTemporaryFile(suffix=".wav") as clip:
            with self.assertRaises(gr.Error):
                list(app_module.predict_clip("not-a-model", clip.name, "pytorch", "", False))

    def test_rejects_unknown_backend(self):
        with tempfile.NamedTemporaryFile(suffix=".wav") as clip:
            with self.assertRaises(gr.Error):
                list(app_module.predict_clip("wavlm_best-ccc", clip.name, "tensorflow", "", False))


class TestPredictClipDispatch(unittest.TestCase):
    def setUp(self):
        self._orig_pytorch = app_module._predict_pytorch
        self._orig_onnx = app_module._predict_onnx
        # predict_clip validates clip_path exists on disk before dispatching
        # (mirrors what Gradio's own File component already guarantees) --
        # a real, if empty, temp file so that check passes and each test
        # still exercises only what it's meant to.
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        # predict_clip resolves clip_path (Path.resolve()) before dispatching,
        # so pre-resolving here too keeps these paths exactly equal to what
        # the mocked backend functions actually receive (e.g. macOS symlinks
        # /var -> /private/var, which .resolve() normalizes).
        wav = Path(self._tmp.name) / "clip.wav"
        wav.touch()
        self.clip_wav = str(wav.resolve())
        mp4 = Path(self._tmp.name) / "clip.mp4"
        mp4.touch()
        self.clip_mp4 = str(mp4.resolve())

    def tearDown(self):
        app_module._predict_pytorch = self._orig_pytorch
        app_module._predict_onnx = self._orig_onnx

    def test_pytorch_backend_calls_the_pytorch_path(self):
        calls = []
        app_module._predict_pytorch = lambda model_id, clip_path, log: (
            calls.append(("pytorch", model_id, clip_path)),
            dict.fromkeys(TRAITS, 0.5),
        )[1]
        app_module._predict_onnx = lambda *args: self.fail("onnx path should not run")

        *_, (figure, log) = app_module.predict_clip(
            "wavlm_best-ccc", self.clip_wav, "pytorch", "", False
        )
        self.assertEqual(calls, [("pytorch", "wavlm_best-ccc", self.clip_wav)])
        self.assertIsNotNone(figure)
        self.assertIn("Model: wavlm_best-ccc", log)
        self.assertIn("Backend: pytorch", log)

    def test_onnx_backend_calls_the_onnx_path(self):
        calls = []
        app_module._predict_onnx = lambda model_id, clip_path, transcript, skip_text, log: (
            calls.append(("onnx", model_id, clip_path, transcript, skip_text)),
            dict.fromkeys(TRAITS, 0.5),
        )[1]
        app_module._predict_pytorch = lambda *args: self.fail("pytorch path should not run")

        *_, (figure, log) = app_module.predict_clip(
            "avt_best-ccc", self.clip_mp4, "onnx", "hi", True
        )
        self.assertEqual(calls, [("onnx", "avt_best-ccc", self.clip_mp4, "hi", True)])
        self.assertIsNotNone(figure)
        self.assertIn("Model: avt_best-ccc", log)
        self.assertIn("Backend: onnx", log)

    def test_a_plain_exception_from_the_backend_surfaces_as_gr_error(self):
        def _boom(model_id, clip_path, log):
            raise RuntimeError("network unreachable")

        app_module._predict_pytorch = _boom

        with self.assertRaises(gr.Error) as ctx:
            list(app_module.predict_clip("wavlm_best-ccc", self.clip_wav, "pytorch", "", False))
        self.assertIn("network unreachable", str(ctx.exception))

    def test_a_gr_error_from_the_backend_passes_through_unwrapped(self):
        def _boom(model_id, clip_path, log):
            raise gr.Error("missing feature wavlm")

        app_module._predict_pytorch = _boom

        with self.assertRaises(gr.Error) as ctx:
            list(app_module.predict_clip("wavlm_best-ccc", self.clip_wav, "pytorch", "", False))
        self.assertEqual(ctx.exception.message, "missing feature wavlm")


class TestOnModelChange(unittest.TestCase):
    def test_audio_only_champion_hides_transcript_controls(self):
        transcript_update, skip_text_update, backend_update = app_module._on_model_change(
            "wavlm_best-ccc"
        )
        self.assertFalse(transcript_update["visible"])
        self.assertFalse(skip_text_update["visible"])

    def test_avt_champion_shows_transcript_controls_and_forces_onnx(self):
        transcript_update, skip_text_update, backend_update = app_module._on_model_change(
            "avt_best-ccc"
        )
        self.assertTrue(transcript_update["visible"])
        self.assertTrue(skip_text_update["visible"])
        self.assertEqual(backend_update["value"], "onnx")


class TestBuildDemo(unittest.TestCase):
    def test_builds_without_error(self):
        demo = app_module.build_demo()
        self.assertIsInstance(demo, gr.Blocks)


if __name__ == "__main__":
    unittest.main()
