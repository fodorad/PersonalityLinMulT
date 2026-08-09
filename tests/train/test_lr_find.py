"""Tests for the LR range test, against a real tiny HDF5 file."""

from __future__ import annotations

from personalitylinmult.data.datamodule import FiDataModule
from personalitylinmult.train.cli import read_input_dims
from personalitylinmult.train.config import ExperimentConfig
from personalitylinmult.train.lr_find import _output_path, find_lr
from personalitylinmult.train.module import FiLightningModule
from train.test_cli import CliTestCase


class TestOutputPath(CliTestCase):
    def test_path_is_under_the_runs_own_results_subdir(self):
        # Matches every other per-run artifact (checkpoints, test predictions):
        # {output_dir}/{task}/{experiment_name}/lr_find.png, not a loose file.
        data_path, model_path, train_path = self.write_configs(task="bigfive")
        config = ExperimentConfig.from_files(data_path, model_path, train_path)
        path = _output_path(config)
        self.assertEqual(path.name, "lr_find.png")
        self.assertEqual(path.parent.name, "cli-test-bigfive")
        self.assertEqual(path.parent.parent.name, "bigfive")


class TestFindLr(CliTestCase):
    def test_returns_a_positive_suggestion_and_saves_an_annotated_plot(self):
        data_path, model_path, train_path = self.write_configs(task="bigfive")
        config = ExperimentConfig.from_files(data_path, model_path, train_path)

        # The fixture h5 is tiny (6 train clips, batch_size 2 -> 3 batches per
        # epoch); num_training must span enough steps for the loss to move
        # across a real range of LRs, or Lightning has too few points to fit a
        # suggestion from.
        suggestion = find_lr(config, num_training=20, min_lr=1e-5, max_lr=1.0)

        self.assertIsInstance(suggestion, float)
        self.assertGreater(suggestion, 0.0)
        plot_path = _output_path(config)
        self.assertTrue(plot_path.is_file())
        self.assertGreater(plot_path.stat().st_size, 0)

    def test_the_lr_property_round_trips_through_the_module(self):
        # This is the exact mechanism Lightning's finder relies on: it reads
        # module.lr to find the attribute, sweeps by writing module.lr, and
        # writes the final suggestion back the same way. Confirm the property
        # added to FiLightningModule actually works, independent of Lightning's
        # internals -- a frozen dataclass field silently failing to update
        # would make the finder always report the same fixed lr.
        data_path, model_path, train_path = self.write_configs(task="bigfive")
        config = ExperimentConfig.from_files(data_path, model_path, train_path)

        datamodule = FiDataModule(config.data, target_names=config.train.targets)
        datamodule.setup("fit")
        module = FiLightningModule(
            model_config=config.model,
            train_config=config.train,
            feature_names=config.data.features,
            input_feature_dim=read_input_dims(config.data),
        )

        original = module.lr
        module.lr = original * 10
        self.assertAlmostEqual(module.lr, original * 10)
        self.assertAlmostEqual(module.train_config.optimizer.lr, original * 10)


if __name__ == "__main__":
    import unittest

    unittest.main()
