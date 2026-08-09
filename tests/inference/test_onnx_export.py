"""Tests for ONNX export, against a real (tiny) LinT checkpoint.

No mocks: exports a genuine LinT model to a real .onnx file and runs it
through onnxruntime, comparing against the same input's PyTorch output.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from personalitylinmult.inference.predictor import FiPredictor
from personalitylinmult.train.config import ModelConfig, TrainConfig
from personalitylinmult.train.module import FiLightningModule

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "emotional_stability"]


def write_lint_checkpoint(path: Path, feature: str = "wavlm", dim: int = 8) -> None:
    """Build a tiny single-feature (LinT) module and save it as a checkpoint."""
    import lightning as L

    model_config = ModelConfig(family="lint", d_model=8, num_heads=2, cmt_num_layers=1)
    train_config = TrainConfig(task="bigfive", targets=TRAITS)
    module = FiLightningModule(
        model_config, train_config, feature_names=[feature], input_feature_dim=[dim]
    )
    trainer = L.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False)
    trainer.strategy.connect(module)
    trainer.save_checkpoint(str(path))


class TestExportToOnnx(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.ckpt = self.root / "lint.ckpt"
        write_lint_checkpoint(self.ckpt)

    def test_exports_onnx_graph_and_metadata_sidecar(self):
        from personalitylinmult.inference.onnx_export import export_to_onnx

        module = FiLightningModule.load_from_checkpoint(str(self.ckpt))
        onnx_path = export_to_onnx(module, self.root / "model.onnx")

        self.assertTrue(onnx_path.is_file())
        metadata_path = self.root / "model.onnx.json"
        self.assertTrue(metadata_path.is_file())

        metadata = json.loads(metadata_path.read_text())
        # Sidecar schema is always plural (feature_names/input_dims), even
        # for a single-feature LinT graph, so PersonalityModelONNX never has
        # to branch on which family produced a given sidecar.
        self.assertEqual(metadata["feature_names"], ["wavlm"])
        self.assertEqual(metadata["input_dims"], [8])
        self.assertEqual(metadata["traits"], TRAITS)

    def test_onnx_output_matches_pytorch(self):
        import onnxruntime as ort

        from personalitylinmult.inference.onnx_export import export_to_onnx

        module = FiLightningModule.load_from_checkpoint(str(self.ckpt))
        onnx_path = export_to_onnx(module, self.root / "model.onnx")

        rng = np.random.default_rng(0)
        x = rng.standard_normal((1, 20, 8)).astype("float32")
        mask = np.ones((1, 20), dtype=bool)

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        (onnx_out,) = session.run(["predictions"], {"x": x, "mask": mask})

        module.eval()
        with torch.no_grad():
            torch_out = module([torch.from_numpy(x)], [torch.from_numpy(mask)]).numpy()

        np.testing.assert_allclose(onnx_out, torch_out, atol=1e-4, rtol=1e-4)

    def test_onnx_output_matches_fipredictor_end_to_end(self):
        """The exported graph must agree with the exact path a real user
        takes -- FiPredictor.predict on the same checkpoint -- not just the
        raw forward call, since FiPredictor also clamps to [0, 1]."""
        import onnxruntime as ort

        from personalitylinmult.inference.onnx_export import export_to_onnx

        module = FiLightningModule.load_from_checkpoint(str(self.ckpt))
        onnx_path = export_to_onnx(module, self.root / "model.onnx")

        rng = np.random.default_rng(1)
        features = {"wavlm": rng.standard_normal((15, 8)).astype("float32")}

        predictor = FiPredictor.from_checkpoint(self.ckpt)
        torch_scores = predictor.predict(features)

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        x = features["wavlm"][np.newaxis, :, :]
        mask = np.ones(x.shape[:2], dtype=bool)
        (raw,) = session.run(["predictions"], {"x": x, "mask": mask})
        onnx_scores = dict(zip(TRAITS, np.clip(raw[0], 0.0, 1.0).tolist(), strict=True))

        for trait in TRAITS:
            self.assertAlmostEqual(torch_scores[trait], onnx_scores[trait], places=4)

    def test_dynamic_time_axis_accepts_varying_lengths(self):
        import onnxruntime as ort

        from personalitylinmult.inference.onnx_export import export_to_onnx

        module = FiLightningModule.load_from_checkpoint(str(self.ckpt))
        onnx_path = export_to_onnx(module, self.root / "model.onnx")
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        for t in (1, 10, 100):
            x = np.random.randn(1, t, 8).astype("float32")
            mask = np.ones((1, t), dtype=bool)
            (out,) = session.run(["predictions"], {"x": x, "mask": mask})
            self.assertEqual(out.shape, (1, 5))


class TestExportLinMulTToOnnx(unittest.TestCase):
    """The multi-feature (LinMulT) export path, e.g. the AVT champion."""

    def setUp(self):
        from inference.test_linmult_mask_validation import FEATURES, write_linmult_checkpoint

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.ckpt = self.root / "linmult.ckpt"
        write_linmult_checkpoint(self.ckpt, attention_type="linear")
        self.features = FEATURES

    def test_exports_onnx_graph_and_plural_metadata_sidecar(self):
        from personalitylinmult.inference.onnx_export import export_to_onnx

        module = FiLightningModule.load_from_checkpoint(str(self.ckpt))
        onnx_path = export_to_onnx(module, self.root / "model.onnx")

        self.assertTrue(onnx_path.is_file())
        metadata_path = self.root / "model.onnx.json"
        self.assertTrue(metadata_path.is_file())

        metadata = json.loads(metadata_path.read_text())
        self.assertEqual(metadata["feature_names"], self.features)
        self.assertEqual(metadata["input_dims"], [8] * len(self.features))
        self.assertEqual(metadata["traits"], TRAITS)

    def test_onnx_output_matches_pytorch(self):
        import onnxruntime as ort

        from personalitylinmult.inference.onnx_export import export_to_onnx

        module = FiLightningModule.load_from_checkpoint(str(self.ckpt))
        onnx_path = export_to_onnx(module, self.root / "model.onnx")

        rng = np.random.default_rng(0)
        n = len(self.features)
        xs = [rng.standard_normal((1, 20, 8)).astype("float32") for _ in range(n)]
        masks = [np.ones((1, 20), dtype=bool) for _ in range(n)]

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        onnx_inputs = {f"x_{i}": xs[i] for i in range(n)} | {
            f"mask_{i}": masks[i] for i in range(n)
        }
        (onnx_out,) = session.run(["predictions"], onnx_inputs)

        module.eval()
        with torch.no_grad():
            torch_inputs = [torch.from_numpy(x) for x in xs]
            torch_masks = [torch.from_numpy(m) for m in masks]
            torch_out = module(torch_inputs, torch_masks).numpy()

        np.testing.assert_allclose(onnx_out, torch_out, atol=1e-4, rtol=1e-4)

    def test_onnx_all_false_text_masks_match_pytorch(self):
        """The exported ONNX graph must preserve the all-False whole-modality
        masking behavior verified in test_linmult_mask_validation.py -- a
        dynamo=False trace could in principle bake in a fixed branch, the
        same way LinT's own mask-content branching is a documented risk for
        a genuinely padded mask. This is the real end-to-end proof, not an
        assumption carried over from the PyTorch-only Stage 0 result."""
        import onnxruntime as ort

        from inference.test_linmult_mask_validation import TEXT_FEATURES
        from personalitylinmult.inference.onnx_export import export_to_onnx

        module = FiLightningModule.load_from_checkpoint(str(self.ckpt))
        onnx_path = export_to_onnx(module, self.root / "model.onnx")

        rng = np.random.default_rng(0)
        n = len(self.features)
        xs = [rng.standard_normal((1, 20, 8)).astype("float32") for _ in range(n)]
        masks = [np.ones((1, 20), dtype=bool) for _ in range(n)]
        for i, name in enumerate(self.features):
            if name in TEXT_FEATURES:
                masks[i] = np.zeros((1, 20), dtype=bool)

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        onnx_inputs = {f"x_{i}": xs[i] for i in range(n)} | {
            f"mask_{i}": masks[i] for i in range(n)
        }
        (onnx_out,) = session.run(["predictions"], onnx_inputs)
        self.assertFalse(np.isnan(onnx_out).any())

        module.eval()
        with torch.no_grad():
            torch_inputs = [torch.from_numpy(x) for x in xs]
            torch_masks = [torch.from_numpy(m) for m in masks]
            torch_out = module(torch_inputs, torch_masks).numpy()

        np.testing.assert_allclose(onnx_out, torch_out, atol=1e-4, rtol=1e-4)

    def test_dynamic_time_axis_accepts_varying_lengths(self):
        import onnxruntime as ort

        from personalitylinmult.inference.onnx_export import export_to_onnx

        module = FiLightningModule.load_from_checkpoint(str(self.ckpt))
        onnx_path = export_to_onnx(module, self.root / "model.onnx")
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        n = len(self.features)

        for t in (1, 10, 100):
            xs = [np.random.randn(1, t, 8).astype("float32") for _ in range(n)]
            masks = [np.ones((1, t), dtype=bool) for _ in range(n)]
            onnx_inputs = {f"x_{i}": xs[i] for i in range(n)} | {
                f"mask_{i}": masks[i] for i in range(n)
            }
            (out,) = session.run(["predictions"], onnx_inputs)
            self.assertEqual(out.shape, (1, 5))

    def test_features_with_genuinely_different_lengths_are_independent(self):
        """Every current caller happened to use the same T for all 7
        features per call, which never exercised dynamic_axes' symbolic
        time-name collision: a single shared "time" name across all 14
        inputs (an earlier version of this export) makes onnxruntime treat
        every feature's T as required-equal, which is architecturally wrong
        -- personalitylinmult/data/dataset.py windows each feature from its
        own rate_hz (WavLM ~50 Hz, DINOv2 ~25 fps, text one row per token),
        so different features genuinely have different T in real data,
        confirmed by predict_from_media's real end-to-end test hitting
        exactly this failure. Each feature must accept an independent T."""
        import onnxruntime as ort

        from personalitylinmult.inference.onnx_export import export_to_onnx

        module = FiLightningModule.load_from_checkpoint(str(self.ckpt))
        onnx_path = export_to_onnx(module, self.root / "model.onnx")
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        n = len(self.features)

        # A different T per feature -- not a single shared t like every
        # other test in this file.
        lengths = [11 + i for i in range(n)]
        xs = [np.random.randn(1, lengths[i], 8).astype("float32") for i in range(n)]
        masks = [np.ones((1, lengths[i]), dtype=bool) for i in range(n)]
        onnx_inputs = {f"x_{i}": xs[i] for i in range(n)} | {
            f"mask_{i}": masks[i] for i in range(n)
        }
        (out,) = session.run(["predictions"], onnx_inputs)
        self.assertEqual(out.shape, (1, 5))
        self.assertFalse(np.isnan(out).any())


if __name__ == "__main__":
    unittest.main()
