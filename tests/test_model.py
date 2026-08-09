"""Tests for the public pip-install-only model API.

No mocks: ``from_pretrained``'s download step is a separate, module-level
function (:func:`personalitylinmult.model._download`) precisely so it can be
substituted with a real local checkpoint path here, instead of mocking the HF
Hub network call.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from inference.test_predictor import write_checkpoint

from personalitylinmult.inference.predictor import FiPredictor
from personalitylinmult.model import PersonalityModel

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "emotional_stability"]


class ModelTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.ckpt = cls.root / "wavlm-emotion2vec_best-ccc.ckpt"
        write_checkpoint(cls.ckpt, ["wavlm", "emotion2vec"], [8, 8])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def features(self, t: int = 10):
        return {
            "wavlm": np.random.randn(t, 8).astype("float32"),
            "emotion2vec": np.random.randn(t, 8).astype("float32"),
        }


class TestPersonalityModel(ModelTestCase):
    def setUp(self):
        # Bypasses the HF Hub download: constructs the same object
        # from_pretrained would, from a real local checkpoint.
        self.model = PersonalityModel(FiPredictor.from_checkpoint(self.ckpt))

    def test_exposes_feature_names_and_traits_from_the_checkpoint(self):
        self.assertEqual(self.model.feature_names, ["wavlm", "emotion2vec"])
        self.assertEqual(self.model.traits, TRAITS)

    def test_predict_returns_one_score_per_trait(self):
        scores = self.model.predict(self.features())
        self.assertEqual(list(scores), TRAITS)

    def test_predict_scores_are_clamped_to_unit_range(self):
        scores = self.model.predict(self.features())
        for trait, value in scores.items():
            self.assertGreaterEqual(value, 0.0, trait)
            self.assertLessEqual(value, 1.0, trait)


class TestFromPretrained(ModelTestCase):
    def test_downloads_and_loads_by_id(self):
        # Monkeypatches the module-level download seam for this one call,
        # rather than mocking huggingface_hub -- from_pretrained's contract
        # (call _download, then FiPredictor.from_checkpoint the result) is
        # exercised for real, only the network fetch itself is swapped out.
        import personalitylinmult.model as model_module

        original = model_module._download
        model_module._download = lambda model_id, revision=None: self.ckpt
        try:
            model = PersonalityModel.from_pretrained("wavlm-emotion2vec_best-ccc")
        finally:
            model_module._download = original

        self.assertEqual(model.feature_names, ["wavlm", "emotion2vec"])
        scores = model.predict(self.features())
        self.assertEqual(list(scores), TRAITS)

    def test_checkpoint_filename_matches_the_naming_convention(self):
        from personalitylinmult.model import _checkpoint_filename

        self.assertEqual(_checkpoint_filename("wavlm_best-ccc"), "wavlm_best-ccc.ckpt")


class TestPublicExport(unittest.TestCase):
    def test_personality_model_is_exported_at_top_level(self):
        import personalitylinmult

        self.assertIs(personalitylinmult.PersonalityModel, PersonalityModel)


if __name__ == "__main__":
    unittest.main()
