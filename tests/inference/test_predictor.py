"""Tests for the inference predictor, against a real (tiny) checkpoint."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import lightning as L
import numpy as np

from personalitylinmult.inference.predictor import FiPredictor, InferenceError
from personalitylinmult.train.config import ModelConfig, TrainConfig
from personalitylinmult.train.module import FiLightningModule

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "emotional_stability"]


def write_checkpoint(path: Path, features: list[str], dims: list[int]) -> None:
    """Build a tiny FI module and save it as a real checkpoint.

    Args:
        path: Destination ``.ckpt``.
        features: Feature names the model consumes.
        dims: Per-feature widths.
    """
    model_config = ModelConfig(
        family="linmult", d_model=8, num_heads=2, cmt_num_layers=1, branch_sat_num_layers=1
    )
    train_config = TrainConfig(task="bigfive", targets=TRAITS)
    module = FiLightningModule(
        model_config, train_config, feature_names=features, input_feature_dim=dims
    )
    trainer = L.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False)
    trainer.strategy.connect(module)
    trainer.save_checkpoint(str(path))


class PredictorTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.ckpt = cls.root / "a.ckpt"
        write_checkpoint(cls.ckpt, ["wavlm", "emotion2vec"], [8, 8])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def features(self, t: int = 10):
        return {
            "wavlm": np.random.randn(t, 8).astype("float32"),
            "emotion2vec": np.random.randn(t, 8).astype("float32"),
        }


class TestLoad(PredictorTestCase):
    def test_from_checkpoint_reads_features_and_traits(self):
        predictor = FiPredictor.from_checkpoint(self.ckpt)
        self.assertEqual(predictor.feature_names, ["wavlm", "emotion2vec"])
        self.assertEqual(predictor.traits, TRAITS)

    def test_missing_checkpoint_raises(self):
        with self.assertRaises(FileNotFoundError):
            FiPredictor.from_checkpoint(self.root / "nope.ckpt")


class TestPredict(PredictorTestCase):
    def setUp(self):
        self.predictor = FiPredictor.from_checkpoint(self.ckpt)

    def test_returns_one_score_per_trait(self):
        scores = self.predictor.predict(self.features())
        self.assertEqual(list(scores), TRAITS)

    def test_scores_are_clamped_to_unit_range(self):
        scores = self.predictor.predict(self.features())
        for trait, value in scores.items():
            self.assertGreaterEqual(value, 0.0, trait)
            self.assertLessEqual(value, 1.0, trait)

    def test_variable_length_is_accepted(self):
        # T differs run to run at inference; the model gap-reduces over time.
        for t in (1, 5, 50):
            scores = self.predictor.predict(self.features(t))
            self.assertEqual(len(scores), len(TRAITS))

    def test_missing_feature_is_rejected(self):
        with self.assertRaises(InferenceError) as ctx:
            self.predictor.predict({"wavlm": np.random.randn(10, 8).astype("float32")})
        self.assertIn("emotion2vec", str(ctx.exception))

    def test_extra_feature_is_rejected(self):
        feats = self.features()
        feats["dinov2"] = np.random.randn(10, 8).astype("float32")
        with self.assertRaises(InferenceError) as ctx:
            self.predictor.predict(feats)
        self.assertIn("dinov2", str(ctx.exception))

    def test_wrong_rank_feature_is_rejected(self):
        feats = self.features()
        feats["wavlm"] = np.random.randn(8).astype("float32")  # 1-D, not (T, F)
        with self.assertRaises(InferenceError) as ctx:
            self.predictor.predict(feats)
        self.assertIn("2-D", str(ctx.exception))

    def test_prediction_is_deterministic_in_eval_mode(self):
        feats = self.features()
        first = self.predictor.predict(feats)
        second = self.predictor.predict(feats)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
