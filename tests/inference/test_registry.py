"""Tests for the model registry."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from inference.test_predictor import write_checkpoint
from personalitylinmult.inference.predictor import FiPredictor
from personalitylinmult.inference.registry import ModelRegistry, ModelSpec


class RegistryTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.a_ckpt = cls.root / "a.ckpt"
        cls.at_ckpt = cls.root / "at.ckpt"
        write_checkpoint(cls.a_ckpt, ["wavlm", "emotion2vec"], [8, 8])
        write_checkpoint(cls.at_ckpt, ["wavlm", "emotion2vec", "xlm_roberta_gt"], [8, 8, 8])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def registry(self):
        return ModelRegistry.from_checkpoints({"A": self.a_ckpt, "AT": self.at_ckpt})


class TestRegistry(RegistryTestCase):
    def test_names_preserve_order(self):
        self.assertEqual(self.registry().names, ["A", "AT"])

    def test_load_returns_a_predictor(self):
        predictor = self.registry().load("A")
        self.assertIsInstance(predictor, FiPredictor)
        self.assertEqual(predictor.feature_names, ["wavlm", "emotion2vec"])

    def test_each_name_loads_its_own_model(self):
        registry = self.registry()
        self.assertEqual(len(registry.load("A").feature_names), 2)
        self.assertEqual(len(registry.load("AT").feature_names), 3)

    def test_load_is_cached(self):
        registry = self.registry()
        self.assertIs(registry.load("A"), registry.load("A"))

    def test_unknown_name_raises_with_the_known_ones(self):
        with self.assertRaises(KeyError) as ctx:
            self.registry().load("XYZ")
        self.assertIn("A", str(ctx.exception))

    def test_spec_returns_the_checkpoint_path(self):
        self.assertEqual(self.registry().spec("A").checkpoint, self.a_ckpt)

    def test_spec_unknown_name_raises(self):
        with self.assertRaises(KeyError):
            self.registry().spec("nope")

    def test_from_checkpoints_defers_opening_until_load(self):
        # A missing file is not an error at construction -- only at load, so a UI
        # can list models whose checkpoints are not all present yet.
        registry = ModelRegistry.from_checkpoints({"ghost": self.root / "absent.ckpt"})
        self.assertEqual(registry.names, ["ghost"])
        with self.assertRaises(FileNotFoundError):
            registry.load("ghost")

    def test_duplicate_names_are_rejected(self):
        with self.assertRaises(ValueError):
            ModelRegistry(
                [
                    ModelSpec("A", self.a_ckpt),
                    ModelSpec("A", self.at_ckpt),
                ]
            )

    def test_a_loaded_model_predicts(self):
        scores = (
            self.registry()
            .load("A")
            .predict(
                {
                    "wavlm": np.random.randn(6, 8).astype("float32"),
                    "emotion2vec": np.random.randn(6, 8).astype("float32"),
                }
            )
        )
        self.assertEqual(len(scores), 5)


if __name__ == "__main__":
    unittest.main()
