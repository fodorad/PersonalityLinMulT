"""Tests for the train-mean predictor baseline, run against a real HDF5 file."""

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from data.test_builder import wavlm_spec

from data import fixtures
from personalitylinmult.data.builder import build
from personalitylinmult.data.schema import BIG_FIVE_TRAITS, DatasetSpec
from personalitylinmult.train.mean_baseline import (
    METRICS_FILE,
    evaluate_mean_baseline,
    read_split_labels,
    run_mean_baseline,
)

VIDEO_IDS = [f"clip{i}" for i in range(12)]
SUBSETS = ["train"] * 6 + ["valid"] * 3 + ["test"] * 3


class MeanBaselineTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        fixtures.write_label_csv(cls.root / "label.csv", VIDEO_IDS, SUBSETS)
        fixtures.write_wavlm(cls.root / "wavlm_baseplus", VIDEO_IDS, dim=8)
        build(
            DatasetSpec(features=[wavlm_spec()], h5_path="fi.h5", label_csv="label.csv"),
            cls.root,
            "cfg",
        )
        cls.h5_path = cls.root / "fi.h5"

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_read_split_labels_shape(self):
        train = read_split_labels(self.h5_path, "train", BIG_FIVE_TRAITS)
        test = read_split_labels(self.h5_path, "test", BIG_FIVE_TRAITS)
        self.assertEqual(tuple(train.shape), (6, 5))
        self.assertEqual(tuple(test.shape), (3, 5))

    def test_constant_prediction_has_zero_ccc_r_and_std_ratio(self):
        metrics = evaluate_mean_baseline(self.h5_path)
        for trait in BIG_FIVE_TRAITS:
            self.assertAlmostEqual(metrics[f"test/ccc/{trait}"], 0.0, places=4)
            self.assertEqual(metrics[f"test/pearson_r/{trait}"], 0.0)
            self.assertEqual(metrics[f"test/std_ratio/{trait}"], 0.0)
        self.assertAlmostEqual(metrics["test/mean_ccc"], 0.0, places=4)

    def test_one_minus_mae_matches_closed_form(self):
        with h5py.File(self.h5_path, "r") as fh:
            subset = fh["subset"][:]
            labels = np.stack([fh[f"labels/{t}"][:] for t in BIG_FIVE_TRAITS], axis=1)
        train_mean = labels[subset == 0].mean(axis=0)
        expected = 1.0 - np.abs(labels[subset == 2] - train_mean).mean(axis=0)

        metrics = evaluate_mean_baseline(self.h5_path)
        for trait, value in zip(BIG_FIVE_TRAITS, expected, strict=True):
            self.assertAlmostEqual(metrics[f"test/one_minus_mae/{trait}"], float(value), places=5)
        self.assertAlmostEqual(metrics["test/mean_one_minus_mae"], float(expected.mean()), places=5)

    def test_only_train_rows_contribute_to_the_mean(self):
        with h5py.File(self.h5_path, "r") as fh:
            subset = fh["subset"][:]
            labels = np.stack([fh[f"labels/{t}"][:] for t in BIG_FIVE_TRAITS], axis=1)
        wrong_mean = labels.mean(axis=0)  # pooled over every split
        wrong = 1.0 - np.abs(labels[subset == 2] - wrong_mean).mean(axis=0)

        metrics = evaluate_mean_baseline(self.h5_path)
        got = np.array([metrics[f"test/one_minus_mae/{t}"] for t in BIG_FIVE_TRAITS])
        self.assertFalse(np.allclose(got, wrong, atol=1e-6))

    def test_json_keys_match_a_trained_run(self):
        with tempfile.TemporaryDirectory() as out:
            run_mean_baseline(self.h5_path, Path(out) / "nested")
            written = json.loads((Path(out) / "nested" / METRICS_FILE).read_text())
        self.assertIn("test/mean_ccc", written)
        self.assertIn("test/mse", written)
        for name in ("one_minus_mae", "pearson_r", "ccc", "r2", "std_ratio"):
            self.assertIn(f"test/mean_{name}", written)
            for trait in BIG_FIVE_TRAITS:
                self.assertIn(f"test/{name}/{trait}", written)


if __name__ == "__main__":
    unittest.main()
