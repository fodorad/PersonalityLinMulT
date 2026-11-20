"""Tests for the random-input baseline, run against a real tiny HDF5 file."""

import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml
from data.test_builder import wavlm_spec

from data import fixtures
from personalitylinmult.data.builder import build
from personalitylinmult.data.dataset import FiHdf5Dataset
from personalitylinmult.data.schema import DatasetSpec
from personalitylinmult.train.cli import run
from personalitylinmult.train.config import DataConfig, ExperimentConfig
from personalitylinmult.train.noise_baseline import NoiseDataModule, NoiseHdf5Dataset, main

VIDEO_IDS = [f"clip{i}" for i in range(12)]
SUBSETS = ["train"] * 6 + ["valid"] * 3 + ["test"] * 3
TARGETS = ["openness", "extraversion"]


class NoiseTestCase(unittest.TestCase):
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

    def make(self, cls, subset):
        return cls(self.h5_path, subset, ["wavlm"], TARGETS, window_seconds=1.0)


class TestNoiseDataset(NoiseTestCase):
    def test_shape_and_dtype_match_the_real_feature(self):
        real = self.make(FiHdf5Dataset, "train")[0]
        noisy = self.make(NoiseHdf5Dataset, "train")[0]
        self.assertEqual(noisy["wavlm"].shape, real["wavlm"].shape)
        self.assertEqual(noisy["wavlm"].dtype, real["wavlm"].dtype)

    def test_keeps_real_mask_targets_and_id(self):
        for subset in ("train", "valid", "test"):
            real = self.make(FiHdf5Dataset, subset)[1]
            noisy = self.make(NoiseHdf5Dataset, subset)[1]
            self.assertTrue(torch.equal(noisy["wavlm_mask"], real["wavlm_mask"]))
            self.assertTrue(torch.equal(noisy["targets"], real["targets"]))
            self.assertEqual(noisy["video_id"], real["video_id"])

    def test_features_are_not_the_real_ones(self):
        real = self.make(FiHdf5Dataset, "train")[0]["wavlm"]
        noisy = self.make(NoiseHdf5Dataset, "train")[0]["wavlm"]
        self.assertFalse(torch.equal(noisy, real))

    def test_train_noise_is_resampled_on_every_access(self):
        dataset = self.make(NoiseHdf5Dataset, "train")
        self.assertFalse(torch.equal(dataset[0]["wavlm"], dataset[0]["wavlm"]))

    def test_eval_noise_is_identical_between_accesses(self):
        for subset in ("valid", "test"):
            dataset = self.make(NoiseHdf5Dataset, subset)
            self.assertTrue(torch.equal(dataset[0]["wavlm"], dataset[0]["wavlm"]))

    def test_eval_noise_differs_between_clips(self):
        dataset = self.make(NoiseHdf5Dataset, "valid")
        self.assertFalse(torch.equal(dataset[0]["wavlm"], dataset[1]["wavlm"]))


class TestNoiseDataModule(NoiseTestCase):
    def test_builds_noise_datasets_for_every_split(self):
        config = DataConfig(h5_path=str(self.h5_path), features=["wavlm"], window_seconds=1.0)
        datamodule = NoiseDataModule(config, target_names=TARGETS)
        datamodule.setup()
        for dataset in (
            datamodule.train_dataset,
            datamodule.valid_dataset,
            datamodule.test_dataset,
        ):
            self.assertIsInstance(dataset, NoiseHdf5Dataset)


class TestRunWithNoise(NoiseTestCase):
    def write_configs(self, out):
        (out / "data.yaml").write_text(
            yaml.safe_dump(
                {
                    "h5_path": str(self.h5_path),
                    "features": ["wavlm"],
                    "batch_size": 2,
                    "num_workers": 0,
                    "pin_memory": False,
                    "persistent_workers": False,
                    "window_seconds": 1.0,
                }
            )
        )
        (out / "model.yaml").write_text(
            yaml.safe_dump({"family": "lint", "d_model": 8, "num_heads": 2, "cmt_num_layers": 1})
        )
        (out / "train.yaml").write_text(
            yaml.safe_dump(
                {
                    "task": "bigfive",
                    "max_epochs": 1,
                    "loss": "l1",
                    "accelerator": "cpu",
                    "output_dir": str(out / "results"),
                    "early_stopping": {"enabled": False},
                    "mlflow": {
                        "tracking_uri": f"sqlite:///{out / 'mlflow.db'}",
                        "experiment_name": "noise-test",
                        "log_model": False,
                    },
                }
            )
        )

    def test_run_completes_and_writes_metrics(self):
        with tempfile.TemporaryDirectory() as out_name:
            out = Path(out_name)
            self.write_configs(out)
            config = ExperimentConfig.from_files(
                out / "data.yaml", out / "model.yaml", out / "train.yaml"
            )
            datamodule = NoiseDataModule(config.data, target_names=config.train.targets)
            metrics = run(config, datamodule=datamodule)

            self.assertIn("test/mean_ccc", metrics)
            run_dir = out / "results" / "bigfive" / "noise-test"
            self.assertTrue((run_dir / "metrics_test.json").is_file())

    def test_cli_main_runs_a_fast_dev_run(self):
        with tempfile.TemporaryDirectory() as out_name:
            out = Path(out_name)
            self.write_configs(out)
            argv = sys.argv
            sys.argv = [
                "noise_baseline",
                "--data", str(out / "data.yaml"),
                "--model", str(out / "model.yaml"),
                "--train", str(out / "train.yaml"),
                "--set", "train.loss=l1",
                "--fast-dev-run",
            ]  # fmt: skip
            self.addCleanup(setattr, sys, "argv", argv)
            main()
            self.assertTrue((out / "results" / "bigfive" / "noise-test").is_dir())


if __name__ == "__main__":
    unittest.main()
