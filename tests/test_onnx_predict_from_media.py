"""Tests for PersonalityModelONNX.predict_from_media -- the full AVT pipeline.

Downloads real models (YOLO face detector, AdaFace, all 6 feature
extractors, Whisper), so this only runs when explicitly requested
(RUN_HEAVY_ONNX_TESTS=1), matching the rest of this project's heavy-test
convention.

Uses a synthetic (non-face) test-pattern clip via ffmpeg -- no licensed
dataset clip needed. This deliberately exercises the "no protagonist found"
fallback path in _extract_face_features (a synthetic test pattern has no
real face for YOLO to detect), which is itself a real, legitimate code path
predict_from_media must handle gracefully, not just the happy path.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

RUN_HEAVY = os.environ.get("RUN_HEAVY_ONNX_TESTS") == "1"
HAS_FFMPEG = shutil.which("ffmpeg") is not None

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "emotional_stability"]


# Real per-extractor output widths, in test_linmult_mask_validation.FEATURES
# order -- must match exactly, unlike the mask-validation tests' toy 8-d
# checkpoint: the real ONNX extractors produce fixed-width output
# (wavlm/emotion2vec/dinov2/xlm_roberta_gt/mmbert_gt: 768-d; emotieffnet:
# 1280-d; farl: 512-d, see each export_*_to_onnx docstring), and a
# checkpoint's input layer must match that width exactly, not the tiny
# placeholder dims write_linmult_checkpoint defaults to for mask-validation
# speed.
_REAL_EXTRACTOR_DIMS = [768, 768, 768, 1280, 512, 768, 768]


def _write_avt_champion_locally(root: Path) -> tuple[Path, Path]:
    """Export a tiny (but real-width) 7-feature LinMulT checkpoint to ONNX,
    standing in for the real avt_best-ccc champion (same construction as
    test_linmult_mask_validation.py's write_linmult_checkpoint, reused via
    import rather than duplicated)."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent / "inference"))
    from test_linmult_mask_validation import write_linmult_checkpoint

    from personalitylinmult.inference.onnx_export import export_to_onnx
    from personalitylinmult.train.module import FiLightningModule

    ckpt = root / "avt.ckpt"
    write_linmult_checkpoint(ckpt, attention_type="linear", dims=_REAL_EXTRACTOR_DIMS)
    module = FiLightningModule.load_from_checkpoint(str(ckpt))
    onnx_path = export_to_onnx(module, root / "avt.onnx")
    metadata_path = root / "avt.onnx.json"
    return onnx_path, metadata_path


@unittest.skipUnless(
    RUN_HEAVY, "set RUN_HEAVY_ONNX_TESTS=1 to run (downloads YOLO/AdaFace/6 extractors)"
)
@unittest.skipUnless(HAS_FFMPEG, "ffmpeg not installed")
class TestPredictFromMediaReal(unittest.TestCase):
    def setUp(self):
        import json

        import onnxruntime as ort

        from personalitylinmult.onnx import PersonalityModelONNX

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)

        onnx_path, metadata_path = _write_avt_champion_locally(root)
        metadata = json.loads(metadata_path.read_text())
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.model = PersonalityModelONNX(
            session, feature_names=metadata["feature_names"], traits=metadata["traits"]
        )

        self.clip = root / "clip.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=1:size=64x64:rate=5",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=1",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-shortest",
                str(self.clip),
            ],
            capture_output=True,
            check=True,
        )

    def test_predict_from_media_with_explicit_transcript(self):
        scores = self.model.predict_from_media(str(self.clip), transcript="hello world")
        self.assertEqual(list(scores), TRAITS)
        for trait, value in scores.items():
            self.assertGreaterEqual(value, 0.0, trait)
            self.assertLessEqual(value, 1.0, trait)

    def test_predict_from_media_with_skip_text_mode(self):
        # transcript="" -> zero placeholder + all-False mask for both text
        # features, the mode this project's own mask-validation tests
        # (test_linmult_mask_validation.py) proved NaN-safe.
        scores = self.model.predict_from_media(str(self.clip), transcript="")
        self.assertEqual(list(scores), TRAITS)
        for trait, value in scores.items():
            self.assertGreaterEqual(value, 0.0, trait)
            self.assertLessEqual(value, 1.0, trait)

    def test_extractor_sessions_are_cached_across_calls(self):
        self.model.predict_from_media(str(self.clip), transcript="hello")
        cached_sessions = dict(self.model._extractor_sessions)
        self.model.predict_from_media(str(self.clip), transcript="world")
        # Same session objects reused, not re-downloaded/rebuilt.
        for name, session in cached_sessions.items():
            self.assertIs(self.model._extractor_sessions[name], session)


if __name__ == "__main__":
    unittest.main()
