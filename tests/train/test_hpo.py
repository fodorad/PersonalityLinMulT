"""Tests for the Optuna hyperparameter search, against a real tiny HDF5 file."""

from __future__ import annotations

import optuna

from personalitylinmult.train.config import ExperimentConfig
from personalitylinmult.train.hpo import (
    build_objective,
    default_search_space,
    default_storage_path,
    run_search,
)
from train.test_cli import CliTestCase

EXPECTED_OVERRIDE_KEYS = {
    "train.optimizer.lr",
    "train.optimizer.weight_decay",
    "train.optimizer.warmup_ratio",
    "model.dropout_attention",
    "model.dropout_ffn",
    "model.head_dropout",
}


def _real_trial() -> optuna.Trial:
    """A trial from a real, throwaway in-memory study -- no mocks."""
    study = optuna.create_study()
    return study.ask()


class TestDefaultSearchSpace(CliTestCase):
    def test_returns_the_expected_override_keys(self):
        overrides = default_search_space(_real_trial())
        self.assertEqual(set(overrides), EXPECTED_OVERRIDE_KEYS)

    def test_values_are_within_their_declared_ranges(self):
        overrides = default_search_space(_real_trial())
        self.assertTrue(1e-4 <= overrides["train.optimizer.lr"] <= 3e-3)
        self.assertTrue(1e-4 <= overrides["train.optimizer.weight_decay"] <= 1e-1)
        self.assertTrue(0.05 <= overrides["train.optimizer.warmup_ratio"] <= 0.2)
        self.assertTrue(0.0 <= overrides["model.dropout_attention"] <= 0.3)
        self.assertTrue(0.1 <= overrides["model.dropout_ffn"] <= 0.4)
        self.assertTrue(0.1 <= overrides["model.head_dropout"] <= 0.4)

    def test_overrides_are_accepted_by_experiment_config(self):
        # The search space must name real, valid config keys -- this would
        # raise if a key were misspelled or a value out of a validated range.
        data_path, model_path, train_path = self.write_configs(task="bigfive")
        overrides = {
            "model.family": "lint",
            "data.features": ["wavlm"],
            **default_search_space(_real_trial()),
        }
        config = ExperimentConfig.from_files(data_path, model_path, train_path, overrides)
        self.assertEqual(config.model.family, "lint")


class TestDefaultStoragePath(CliTestCase):
    def test_path_is_under_the_studys_own_results_subdir(self):
        # Regression test: the Optuna DB used to default to an in-memory study
        # (or, when Adam ran it manually, a loose file at the repo root) --
        # neither lives next to the study's own checkpoints the way every other
        # per-experiment artifact does.
        _, _, train_path = self.write_configs(task="bigfive")
        path = default_storage_path(train_path, "my-study")
        self.assertEqual(path.name, "optuna.db")
        self.assertEqual(path.parent.name, "my-study")
        self.assertEqual(path.parent.parent.name, "bigfive")

    def test_run_search_persists_to_that_path_by_default(self):
        data_path, model_path, train_path = self.write_configs(
            task="bigfive", max_epochs=1, early_stopping={"enabled": False}
        )
        run_search(
            data_path,
            model_path,
            train_path,
            {"model.family": "lint", "data.features": ["wavlm"]},
            n_trials=1,
            min_resource_epochs=10,
            study_name="persisted-study",
        )
        db_path = default_storage_path(train_path, "persisted-study")
        self.assertTrue(db_path.is_file())

        # A second call with the same study name resumes rather than starting
        # a fresh in-memory study with no history.
        resumed = optuna.load_study(study_name="persisted-study", storage=f"sqlite:///{db_path}")
        self.assertEqual(len(resumed.trials), 1)


class TestBuildObjective(CliTestCase):
    def test_experiment_name_is_study_specific_not_the_train_config_filename(self):
        # Regression test: the experiment name used to default to
        # f"{train_path.stem}-hpo", which collapsed every search (regardless of
        # what model/data it searched) into the same "bigfive-hpo" experiment
        # and checkpoint directory, corrupting concurrent/later searches. Every
        # config here is named "bigfive" (train_path.stem), so if the old bug
        # were still present the run would land under "bigfive-hpo" instead.
        import mlflow

        data_path, model_path, train_path = self.write_configs(
            task="bigfive", max_epochs=1, early_stopping={"enabled": False}
        )
        objective = build_objective(
            data_path,
            model_path,
            train_path,
            {"model.family": "lint", "data.features": ["wavlm"]},
            "my-specific-study",
        )
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1)

        config = ExperimentConfig.from_files(data_path, model_path, train_path)
        mlflow.set_tracking_uri(config.train.mlflow.tracking_uri)
        client = mlflow.tracking.MlflowClient()
        self.assertIsNotNone(client.get_experiment_by_name("my-specific-study"))
        self.assertIsNone(client.get_experiment_by_name("bigfive-hpo"))

    def test_a_real_trial_trains_and_reports_a_score(self):
        # End-to-end: one real trial through the tiny fixture h5, no pruning
        # (min_resource well above max_epochs so it never fires), confirming
        # the objective actually trains and returns a finite score.
        data_path, model_path, train_path = self.write_configs(
            task="bigfive", max_epochs=1, early_stopping={"enabled": False}
        )
        study = run_search(
            data_path,
            model_path,
            train_path,
            {"model.family": "lint", "data.features": ["wavlm"]},
            n_trials=1,
            min_resource_epochs=10,
            study_name="test-hpo-study",
        )
        self.assertEqual(len(study.trials), 1)
        self.assertTrue(study.trials[0].state.is_finished())
        self.assertIsInstance(study.best_value, float)


if __name__ == "__main__":
    import unittest

    unittest.main()
