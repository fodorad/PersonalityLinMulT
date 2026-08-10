"""Tests for the public ONNX Runtime-backed model API.

No mocks: exports a real LinT checkpoint to ONNX, then loads it through
PersonalityModelONNX exactly as from_pretrained would, with the HF Hub
download step substituted for local paths -- the same seam pattern as
test_model.py.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from inference.test_onnx_export import write_lint_checkpoint

from personalitylinmult.inference.onnx_export import export_to_onnx
from personalitylinmult.onnx import PersonalityModelONNX
from personalitylinmult.train.module import FiLightningModule

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "emotional_stability"]


class OnnxModelTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        ckpt = cls.root / "wavlm_best-ccc.ckpt"
        # Seeded so this checkpoint's weights are reproducible across runs
        # and test order -- an unseeded init previously let
        # test_all_false_mask_changes_the_output fail flakily under a full
        # `make check` run (passed reliably alone), landing on a random
        # weight init where the mask's numeric effect happened to fall
        # under the test's tolerance.
        torch.manual_seed(0)
        write_lint_checkpoint(ckpt, feature="wavlm", dim=8)

        module = FiLightningModule.load_from_checkpoint(str(ckpt))
        cls.onnx_path = export_to_onnx(module, cls.root / "wavlm_best-ccc.onnx")
        cls.metadata_path = cls.root / "wavlm_best-ccc.onnx.json"

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def features(self, t: int = 10):
        # Seeded per-call (not module-level) so every caller gets the same
        # values regardless of how much other unseeded np.random state ran
        # before it in the same test session -- see setUpClass's note.
        rng = np.random.default_rng(0)
        return {"wavlm": rng.standard_normal((t, 8)).astype("float32")}


class TestPersonalityModelONNX(OnnxModelTestCase):
    def setUp(self):
        import onnxruntime as ort

        session = ort.InferenceSession(str(self.onnx_path), providers=["CPUExecutionProvider"])
        metadata = json.loads(self.metadata_path.read_text())
        self.model = PersonalityModelONNX(
            session, feature_names=metadata["feature_names"], traits=metadata["traits"]
        )

    def test_exposes_feature_names_and_traits(self):
        self.assertEqual(self.model.feature_names, ["wavlm"])
        self.assertEqual(self.model.traits, TRAITS)

    def test_predict_returns_one_score_per_trait(self):
        scores = self.model.predict(self.features())
        self.assertEqual(list(scores), TRAITS)

    def test_predict_scores_are_clamped_to_unit_range(self):
        scores = self.model.predict(self.features())
        for trait, value in scores.items():
            self.assertGreaterEqual(value, 0.0, trait)
            self.assertLessEqual(value, 1.0, trait)


class TestFromPretrained(OnnxModelTestCase):
    def test_downloads_and_loads_by_id(self):
        import personalitylinmult.onnx as onnx_module

        original = onnx_module._download
        onnx_module._download = lambda model_id, revision=None: (
            self.onnx_path,
            self.metadata_path,
        )
        try:
            model = PersonalityModelONNX.from_pretrained("wavlm_best-ccc")
        finally:
            onnx_module._download = original

        self.assertEqual(model.feature_names, ["wavlm"])
        scores = model.predict(self.features())
        self.assertEqual(list(scores), TRAITS)

    def test_onnx_filenames_match_the_naming_convention(self):
        from personalitylinmult.onnx import _onnx_filenames

        graph, metadata = _onnx_filenames("wavlm_best-ccc")
        self.assertEqual(graph, "wavlm_best-ccc.onnx")
        self.assertEqual(metadata, "wavlm_best-ccc.onnx.json")


class TestDownloadSeams(OnnxModelTestCase):
    """Exercises _download/_download_wavlm_onnx's own bodies.

    huggingface_hub.hf_hub_download is substituted with a plain function
    returning a local file path (no unittest.mock, no network call) --
    this is the one level below TestFromPretrained's seam, covering the
    real request-building/logging code inside _download itself rather
    than treating it as an opaque, always-swapped-out dependency.
    """

    def test_download_calls_hf_hub_download_for_graph_and_metadata(self):
        import huggingface_hub

        from personalitylinmult.onnx import _download

        calls = []

        def fake_hf_hub_download(*, repo_id, filename, revision, cache_dir):
            calls.append(filename)
            return str(self.onnx_path) if filename.endswith(".onnx") else str(self.metadata_path)

        original = huggingface_hub.hf_hub_download
        huggingface_hub.hf_hub_download = fake_hf_hub_download
        try:
            onnx_path, metadata_path = _download("wavlm_best-ccc")
        finally:
            huggingface_hub.hf_hub_download = original

        self.assertEqual(calls, ["wavlm_best-ccc.onnx", "wavlm_best-ccc.onnx.json"])
        self.assertEqual(onnx_path, self.onnx_path)
        self.assertEqual(metadata_path, self.metadata_path)

    def test_download_wavlm_onnx_calls_hf_hub_download(self):
        import huggingface_hub

        from personalitylinmult.onnx import WAVLM_ONNX_FILENAME, _download_wavlm_onnx

        calls = []

        def fake_hf_hub_download(*, repo_id, filename, revision, cache_dir):
            calls.append(filename)
            return str(self.onnx_path)

        original = huggingface_hub.hf_hub_download
        huggingface_hub.hf_hub_download = fake_hf_hub_download
        try:
            path = _download_wavlm_onnx()
        finally:
            huggingface_hub.hf_hub_download = original

        self.assertEqual(calls, [WAVLM_ONNX_FILENAME])
        self.assertEqual(path, self.onnx_path)


class FakeWavlmSession:
    """A stand-in for a real onnxruntime WavLM session.

    Returns deterministic, fixed-shape features so predict_from_audio's
    plumbing (decode -> this session -> predict) can be tested without
    downloading and tracing the real ~380 MB WavLM model.
    """

    def __init__(self, t: int = 20, dim: int = 8):
        self._t = t
        self._dim = dim

    def run(self, output_names, inputs):
        del output_names, inputs
        return [np.random.randn(1, self._t, self._dim).astype("float32")]


class TestPredictFromAudio(OnnxModelTestCase):
    def setUp(self):
        import onnxruntime as ort

        session = ort.InferenceSession(str(self.onnx_path), providers=["CPUExecutionProvider"])
        metadata = json.loads(self.metadata_path.read_text())
        self.model = PersonalityModelONNX(
            session, feature_names=metadata["feature_names"], traits=metadata["traits"]
        )
        # Inject a fake WavLM session directly, bypassing the HF Hub download
        # predict_from_audio would otherwise trigger on first use.
        self.model._wavlm_session = FakeWavlmSession(dim=8)

    def test_rejects_a_non_wavlm_model(self):
        self.model.feature_names = ["emotion2vec"]
        with self.assertRaises(ValueError) as ctx:
            self.model.predict_from_audio("clip.wav")
        self.assertIn("wavlm", str(ctx.exception))

    def test_predicts_from_a_real_audio_file(self):
        import shutil
        import subprocess
        import tempfile

        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not installed")

        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "clip.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", str(clip)],
                capture_output=True,
                check=True,
            )
            scores = self.model.predict_from_audio(str(clip))

        self.assertEqual(list(scores), TRAITS)
        for trait, value in scores.items():
            self.assertGreaterEqual(value, 0.0, trait)
            self.assertLessEqual(value, 1.0, trait)

    def test_first_call_downloads_and_builds_a_wavlm_session(self):
        # No pre-injected _wavlm_session here (unlike setUp's default) --
        # this exercises the real cold-start branch: _download_wavlm_onnx +
        # ort.InferenceSession construction. Any valid ONNX graph works to
        # prove the session gets built; the already-exported LinT graph
        # from setUpClass stands in for WavLM's real ~380 MB one.
        import shutil
        import subprocess
        import tempfile

        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not installed")

        import personalitylinmult.onnx as onnx_module

        self.model._wavlm_session = None
        original = onnx_module._download_wavlm_onnx
        onnx_module._download_wavlm_onnx = lambda revision=None: self.onnx_path
        try:
            with tempfile.TemporaryDirectory() as tmp:
                clip = Path(tmp) / "clip.wav"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-f",
                        "lavfi",
                        "-i",
                        "sine=frequency=440:duration=1",
                        str(clip),
                    ],
                    capture_output=True,
                    check=True,
                )
                # The stand-in graph's real input/output names won't match
                # what extract_wavlm_onnx expects, so this call is expected
                # to fail past session construction -- that's fine, the
                # session-build line itself is what this test targets.
                with self.assertRaises(Exception):
                    self.model.predict_from_audio(str(clip))
        finally:
            onnx_module._download_wavlm_onnx = original
        self.assertIsNotNone(self.model._wavlm_session)

    def test_wavlm_session_is_cached_and_not_redownloaded(self):
        import shutil
        import subprocess
        import tempfile

        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not installed")

        # A pre-cached session (set in setUp) must be reused as-is: calling
        # predict_from_audio must not touch the download seam at all.
        import personalitylinmult.onnx as onnx_module

        original = onnx_module._download_wavlm_onnx
        onnx_module._download_wavlm_onnx = lambda revision=None: (_ for _ in ()).throw(
            AssertionError("Should not re-download once cached.")
        )
        try:
            with tempfile.TemporaryDirectory() as tmp:
                clip = Path(tmp) / "clip.wav"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-f",
                        "lavfi",
                        "-i",
                        "sine=frequency=440:duration=1",
                        str(clip),
                    ],
                    capture_output=True,
                    check=True,
                )
                scores = self.model.predict_from_audio(str(clip))
            self.assertEqual(list(scores), TRAITS)
        finally:
            onnx_module._download_wavlm_onnx = original


class TestPredictWithMasksOverride(OnnxModelTestCase):
    """predict()'s masks= parameter, the plumbing predict_from_media's
    transcript="" mode relies on."""

    def setUp(self):
        import onnxruntime as ort

        session = ort.InferenceSession(str(self.onnx_path), providers=["CPUExecutionProvider"])
        metadata = json.loads(self.metadata_path.read_text())
        self.model = PersonalityModelONNX(
            session, feature_names=metadata["feature_names"], traits=metadata["traits"]
        )

    def test_default_mask_is_all_true(self):
        # No masks= given -> identical to the pre-existing behavior. Reuses
        # the SAME features dict for both calls -- self.features() itself
        # returns fresh random values each call, so calling it twice would
        # compare two different inputs, not the same input under two mask
        # arguments.
        features = self.features()
        scores_no_mask = self.model.predict(features)
        scores_explicit_true = self.model.predict(
            features, masks={"wavlm": np.ones(10, dtype=bool)}
        )
        for trait in TRAITS:
            self.assertAlmostEqual(scores_no_mask[trait], scores_explicit_true[trait], places=5)

    def test_all_false_mask_changes_the_output(self):
        features = self.features()
        all_true = self.model.predict(features)
        all_false = self.model.predict(features, masks={"wavlm": np.zeros(10, dtype=bool)})
        differs = any(abs(all_true[trait] - all_false[trait]) > 1e-6 for trait in TRAITS)
        self.assertTrue(differs, "an all-False mask should change the prediction")


class TestPredictFromMediaGuard(OnnxModelTestCase):
    """predict_from_media is specific to the 7-feature AVT champion --
    calling it on a wavlm-only model must fail clearly, not silently do the
    wrong thing (mirrors predict_from_audio's own feature-set guard)."""

    def setUp(self):
        import onnxruntime as ort

        session = ort.InferenceSession(str(self.onnx_path), providers=["CPUExecutionProvider"])
        metadata = json.loads(self.metadata_path.read_text())
        self.model = PersonalityModelONNX(
            session, feature_names=metadata["feature_names"], traits=metadata["traits"]
        )

    def test_rejects_a_non_avt_model(self):
        with self.assertRaises(ValueError) as ctx:
            self.model.predict_from_media("clip.mp4")
        self.assertIn("AVT", str(ctx.exception))

    def test_avt_feature_names_match_the_champion_config(self):
        # Locks the exact feature order this method's docstring and
        # onnx.py's EXTRACTOR_ONNX_FILENAMES both assume.
        self.assertEqual(
            list(PersonalityModelONNX.AVT_FEATURE_NAMES),
            [
                "wavlm",
                "emotion2vec",
                "dinov2",
                "emotieffnet",
                "farl",
                "xlm_roberta_gt",
                "mmbert_gt",
            ],
        )


class TestReadMetadataSchemaCompat(unittest.TestCase):
    """A sidecar published before the LinMulT export existed uses the old
    singular feature_name/input_dim keys -- _read_metadata must keep reading
    it correctly without a re-export, since it is already live on HF Hub."""

    def test_reads_the_old_singular_schema(self):
        import tempfile

        from personalitylinmult.onnx import _read_metadata

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.onnx.json"
            path.write_text(
                json.dumps({"feature_name": "wavlm", "input_dim": 768, "traits": TRAITS})
            )
            feature_names, traits = _read_metadata(path)
        self.assertEqual(feature_names, ["wavlm"])
        self.assertEqual(traits, TRAITS)

    def test_reads_the_new_plural_schema(self):
        import tempfile

        from personalitylinmult.onnx import _read_metadata

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "new.onnx.json"
            path.write_text(
                json.dumps(
                    {
                        "feature_names": ["wavlm", "emotion2vec"],
                        "input_dims": [768, 768],
                        "traits": TRAITS,
                    }
                )
            )
            feature_names, traits = _read_metadata(path)
        self.assertEqual(feature_names, ["wavlm", "emotion2vec"])
        self.assertEqual(traits, TRAITS)


class _FakeDetection:
    """Stands in for exordium's real Detection: only `.crop()` is used by
    _densify_face_features, returning a fixed-content (3, H, W) tensor."""

    def __init__(self, frame_id: int, value: float):
        self.frame_id = frame_id
        self._value = value

    def crop(self, square: bool = True, extra_space: float = 1.5):
        import torch

        return torch.full((3, 8, 8), self._value, dtype=torch.uint8)


class _FakeTrack:
    """Stands in for exordium's real Track: only frame_ids()/get_detection()/
    __len__ are used by _densify_face_features."""

    def __init__(self, detections_by_frame: dict[int, _FakeDetection]):
        self._by_frame = detections_by_frame

    def frame_ids(self) -> list[int]:
        return sorted(self._by_frame)

    def get_detection(self, frame_id: int) -> _FakeDetection:
        return self._by_frame[frame_id]

    def __len__(self) -> int:
        return len(self._by_frame)


class _FakeVisionSession:
    """Stands in for an onnxruntime.InferenceSession: returns a fixed-shape
    feature vector, ignoring the actual pixel content."""

    def __init__(self, dim: int):
        self._dim = dim

    def run(self, output_names, inputs):
        return [np.ones((1, self._dim), dtype="float32")]


class TestDensifyFaceFeatures(unittest.TestCase):
    """Unit tests for personalitylinmult.onnx._densify_face_features --
    isolates the gap-filling logic from real face detection/tracking, since
    a genuinely gapped real video fixture is impractical to synthesize
    reliably with ffmpeg test patterns."""

    def setUp(self):
        from personalitylinmult.onnx import _densify_face_features

        self.densify = _densify_face_features
        self.emotieffnet_session = _FakeVisionSession(1280)
        self.farl_session = _FakeVisionSession(512)

    def test_no_protagonist_returns_all_zero_and_all_false_mask(self):
        emotieffnet, farl, mask = self.densify(
            None,
            num_frames=5,
            emotieffnet_session=self.emotieffnet_session,
            farl_session=self.farl_session,
        )
        self.assertEqual(emotieffnet.shape, (5, 1280))
        self.assertEqual(farl.shape, (5, 512))
        self.assertEqual(mask.shape, (5,))
        self.assertFalse(mask.any())
        self.assertFalse(emotieffnet.any())
        self.assertFalse(farl.any())

    def test_empty_track_returns_all_zero_and_all_false_mask(self):
        track = _FakeTrack({})
        emotieffnet, farl, mask = self.densify(
            track,
            num_frames=4,
            emotieffnet_session=self.emotieffnet_session,
            farl_session=self.farl_session,
        )
        self.assertFalse(mask.any())

    def test_full_track_marks_every_frame_true(self):
        track = _FakeTrack({i: _FakeDetection(i, float(i)) for i in range(4)})
        emotieffnet, farl, mask = self.densify(
            track,
            num_frames=4,
            emotieffnet_session=self.emotieffnet_session,
            farl_session=self.farl_session,
        )
        self.assertTrue(mask.all())
        self.assertTrue((emotieffnet != 0).all())
        self.assertTrue((farl != 0).all())

    def test_gapped_track_zero_fills_only_the_missing_frames(self):
        # Protagonist detected at frames 0, 1, 4 out of a 5-frame clip --
        # frames 2 and 3 are a genuine gap (occlusion/brief exit).
        track = _FakeTrack(
            {0: _FakeDetection(0, 1.0), 1: _FakeDetection(1, 1.0), 4: _FakeDetection(4, 1.0)}
        )
        emotieffnet, farl, mask = self.densify(
            track,
            num_frames=5,
            emotieffnet_session=self.emotieffnet_session,
            farl_session=self.farl_session,
        )
        np.testing.assert_array_equal(mask, [True, True, False, False, True])
        # Real rows are non-zero (fake session always returns ones)...
        for i in (0, 1, 4):
            self.assertTrue((emotieffnet[i] != 0).all())
            self.assertTrue((farl[i] != 0).all())
        # ...gap rows stay the zero placeholder.
        for i in (2, 3):
            self.assertFalse(emotieffnet[i].any())
            self.assertFalse(farl[i].any())

    def test_out_of_range_frame_id_is_ignored_not_indexerror(self):
        # A detector-reported frame_id past the decoded frame count must not
        # crash predict_from_media -- silently dropped, matching the inline
        # comment in _densify_face_features.
        track = _FakeTrack({0: _FakeDetection(0, 1.0), 10: _FakeDetection(10, 1.0)})
        emotieffnet, farl, mask = self.densify(
            track,
            num_frames=3,
            emotieffnet_session=self.emotieffnet_session,
            farl_session=self.farl_session,
        )
        self.assertEqual(mask.shape, (3,))
        np.testing.assert_array_equal(mask, [True, False, False])


class TestPublicExport(unittest.TestCase):
    def test_onnx_module_does_not_require_onnxruntime_to_import(self):
        # PersonalityModelONNX imports onnxruntime lazily inside
        # from_pretrained/predict, never at module scope -- a bare `pip
        # install personalitylinmult` (no [onnx] extra) must still be able
        # to `import personalitylinmult.onnx` without an ImportError, even
        # though calling from_pretrained on it would fail.
        import importlib
        import sys

        sys.modules.pop("personalitylinmult.onnx", None)
        module = importlib.import_module("personalitylinmult.onnx")
        self.assertTrue(hasattr(module, "PersonalityModelONNX"))


if __name__ == "__main__":
    unittest.main()
