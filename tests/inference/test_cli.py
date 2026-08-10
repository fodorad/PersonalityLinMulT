"""Tests for the inference CLI."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from inference.test_predictor import write_checkpoint
from personalitylinmult.inference.cli import load_features, main

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "emotional_stability"]


class TestLoadFeatures(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_round_trips_an_npz(self):
        path = self.root / "feats.npz"
        wavlm = np.random.randn(10, 8).astype("float32")
        emotion2vec = np.random.randn(10, 8).astype("float32")
        np.savez(path, wavlm=wavlm, emotion2vec=emotion2vec)

        loaded = load_features(path)
        self.assertEqual(set(loaded), {"wavlm", "emotion2vec"})
        np.testing.assert_array_equal(loaded["wavlm"], wavlm)


class TestMain(unittest.TestCase):
    """End-to-end CLI: a real checkpoint + an npz -> JSON scores on stdout."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.ckpt = self.root / "a.ckpt"
        write_checkpoint(self.ckpt, ["wavlm", "emotion2vec"], [8, 8])
        self.feats = self.root / "feats.npz"
        np.savez(
            self.feats,
            wavlm=np.random.randn(10, 8).astype("float32"),
            emotion2vec=np.random.randn(10, 8).astype("float32"),
        )

    def _run(self, argv: list[str]) -> str:
        out = io.StringIO()
        old = sys.argv
        sys.argv = ["prog", *argv]
        try:
            with contextlib.redirect_stdout(out):
                main()
        finally:
            sys.argv = old
        return out.getvalue()

    def test_prints_a_score_per_trait_as_json(self):
        printed = self._run(["--checkpoint", str(self.ckpt), "--features", str(self.feats)])
        scores = json.loads(printed)
        self.assertEqual(list(scores), TRAITS)
        for value in scores.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
