"""Tests for the MLflow pyfunc wrapper and model registration.

These exercise the real pyfunc contract against a tiny checkpoint and a local
sqlite tracking store -- no mocks, a genuine log/register/load round-trip.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from inference.test_predictor import write_checkpoint
from personalitylinmult.inference.mlflow_pyfunc import FiPyfunc, log_and_register, set_champion

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "emotional_stability"]


def _sample(t: int = 6):
    return {
        "wavlm": np.random.randn(t, 8).astype("float32"),
        "emotion2vec": np.random.randn(t, 8).astype("float32"),
    }


class TestFiPyfunc(unittest.TestCase):
    """The delegate that load_context + predict route through."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.ckpt = Path(cls._tmp.name) / "a.ckpt"
        write_checkpoint(cls.ckpt, ["wavlm", "emotion2vec"], [8, 8])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _loaded(self):
        model = FiPyfunc()
        model.load_context(SimpleNamespace(artifacts={"checkpoint": str(self.ckpt)}))
        return model

    def test_predicts_a_single_sample(self):
        out = self._loaded().predict(None, _sample())
        self.assertEqual(len(out), 1)
        self.assertEqual(list(out[0]), TRAITS)

    def test_predicts_a_batch(self):
        out = self._loaded().predict(None, [_sample(), _sample(5)])
        self.assertEqual(len(out), 2)

    def test_scores_are_in_unit_range(self):
        out = self._loaded().predict(None, _sample())
        for value in out[0].values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)


class TestRegistration(unittest.TestCase):
    """A real log_and_register into an isolated sqlite store."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.ckpt = self.root / "a.ckpt"
        write_checkpoint(self.ckpt, ["wavlm", "emotion2vec"], [8, 8])

        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{self.root / 'mlflow.db'}")

    def test_register_then_load_and_predict(self):
        import mlflow

        version = log_and_register(self.ckpt, "test-fi-a")
        self.assertEqual(version, "1")

        model = mlflow.pyfunc.load_model("models:/test-fi-a/1")
        out = model.predict([{k: v.tolist() for k, v in _sample().items()}])
        self.assertEqual(list(out[0]), TRAITS)

    def test_missing_checkpoint_raises(self):
        with self.assertRaises(FileNotFoundError):
            log_and_register(self.root / "nope.ckpt", "test-x")


class TestSetChampion(unittest.TestCase):
    """Alias-based promotion -- the replacement for deprecated stages."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.ckpt = self.root / "a.ckpt"
        write_checkpoint(self.ckpt, ["wavlm", "emotion2vec"], [8, 8])

        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{self.root / 'mlflow.db'}")
        self.mlflow = mlflow

    def test_alias_resolves_to_the_promoted_version(self):
        v1 = log_and_register(self.ckpt, "test-champion")
        set_champion("test-champion", v1, alias="champion")

        client = self.mlflow.tracking.MlflowClient()
        resolved = client.get_model_version_by_alias("test-champion", "champion")
        self.assertEqual(str(resolved.version), v1)

    def test_alias_moves_to_a_new_version(self):
        v1 = log_and_register(self.ckpt, "test-move")
        v2 = log_and_register(self.ckpt, "test-move")
        self.assertNotEqual(v1, v2)

        set_champion("test-move", v1, alias="champion")
        set_champion("test-move", v2, alias="champion")

        client = self.mlflow.tracking.MlflowClient()
        resolved = client.get_model_version_by_alias("test-move", "champion")
        self.assertEqual(str(resolved.version), v2)

    def test_default_alias_is_champion(self):
        v1 = log_and_register(self.ckpt, "test-default-alias")
        set_champion("test-default-alias", v1)

        client = self.mlflow.tracking.MlflowClient()
        resolved = client.get_model_version_by_alias("test-default-alias", "champion")
        self.assertEqual(str(resolved.version), v1)

    def test_independent_models_keep_independent_champions(self):
        # Two different registered-model names -- promoting one's @champion
        # must not affect the other's, since AVT and A each need their own.
        avt_v1 = log_and_register(self.ckpt, "test-avt")
        a_v1 = log_and_register(self.ckpt, "test-a")
        set_champion("test-avt", avt_v1, alias="champion")
        set_champion("test-a", a_v1, alias="champion")

        client = self.mlflow.tracking.MlflowClient()
        self.assertEqual(
            str(client.get_model_version_by_alias("test-avt", "champion").version), avt_v1
        )
        self.assertEqual(str(client.get_model_version_by_alias("test-a", "champion").version), a_v1)


if __name__ == "__main__":
    unittest.main()
