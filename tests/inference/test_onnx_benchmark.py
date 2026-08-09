"""Tests for the PyTorch-vs-ONNX benchmark, against a real (tiny) h5 + LinT
checkpoint. No mocks: runs both the PyTorch and ONNX Runtime paths for real
over a small built test split.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml
from data.test_builder import wavlm_spec

from data import fixtures
from inference.test_onnx_export import write_lint_checkpoint
from personalitylinmult.data.builder import build
from personalitylinmult.data.schema import DatasetSpec
from personalitylinmult.inference.onnx_benchmark import run_benchmark
from personalitylinmult.inference.onnx_export import export_to_onnx
from personalitylinmult.train.module import FiLightningModule

VIDEO_IDS = [f"clip{i}" for i in range(12)]
SUBSETS = ["train"] * 6 + ["valid"] * 3 + ["test"] * 3


class TestRunBenchmark(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

        fixtures.write_label_csv(self.root / "label.csv", VIDEO_IDS, SUBSETS)
        fixtures.write_wavlm(self.root / "wavlm_baseplus", VIDEO_IDS, dim=8)
        build(
            DatasetSpec(features=[wavlm_spec()], h5_path="fi.h5", label_csv="label.csv"),
            self.root,
            "cfg",
        )

        data_path = self.root / "data.yaml"
        data_path.write_text(
            yaml.safe_dump(
                {
                    "h5_path": str(self.root / "fi.h5"),
                    "features": ["wavlm"],
                    "batch_size": 2,
                    "num_workers": 0,
                    "pin_memory": False,
                    "persistent_workers": False,
                    "window_seconds": 1.0,
                }
            )
        )
        self.data_path = data_path

        self.ckpt = self.root / "lint.ckpt"
        write_lint_checkpoint(self.ckpt, feature="wavlm", dim=8)

        module = FiLightningModule.load_from_checkpoint(str(self.ckpt), map_location="cpu")
        self.onnx_path = export_to_onnx(module, self.root / "lint.onnx")

    def test_pytorch_and_onnx_report_the_same_metric_keys(self):
        results = run_benchmark(
            str(self.ckpt), str(self.onnx_path), str(self.data_path), n_timing_runs=2
        )
        self.assertEqual(set(results), {"pytorch", "onnx"})
        expected_keys = {
            "latency_ms",
            "mean_one_minus_mae",
            "mean_pearson_r",
            "mean_ccc",
            "mean_r2",
            "mean_std_ratio",
        }
        self.assertEqual(set(results["pytorch"]), expected_keys)
        self.assertEqual(set(results["onnx"]), expected_keys)

    def test_onnx_metrics_match_pytorch_metrics(self):
        """The export is exact (no quantization) -- accuracy metrics from
        both paths must agree on this small test split, not just be close."""
        results = run_benchmark(
            str(self.ckpt), str(self.onnx_path), str(self.data_path), n_timing_runs=2
        )
        for key in ("mean_one_minus_mae", "mean_pearson_r", "mean_ccc", "mean_r2"):
            self.assertAlmostEqual(results["pytorch"][key], results["onnx"][key], places=4)


if __name__ == "__main__":
    unittest.main()
