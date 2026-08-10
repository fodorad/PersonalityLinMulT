"""Tests for the training entry point, run against a real HDF5 file."""

import json
import tempfile
import unittest
from pathlib import Path

import yaml
from data.test_builder import dinov2_spec, wavlm_spec

from data import fixtures
from personalitylinmult.data.builder import build
from personalitylinmult.data.schema import DatasetSpec
from personalitylinmult.train.cli import (
    MONITOR,
    RUN_ID_FILE,
    build_callbacks,
    read_input_dims,
    run,
)
from personalitylinmult.train.config import DataConfig, ExperimentConfig

VIDEO_IDS = [f"clip{i}" for i in range(12)]
SUBSETS = ["train"] * 6 + ["valid"] * 3 + ["test"] * 3


class CliTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)

        fixtures.write_label_csv(cls.root / "label.csv", VIDEO_IDS, SUBSETS)
        fixtures.write_wavlm(cls.root / "wavlm_baseplus", VIDEO_IDS, dim=8)
        fixtures.write_dinov2(cls.root / "dinov2_base", VIDEO_IDS, dim=8)
        build(
            DatasetSpec(
                features=[wavlm_spec(), dinov2_spec()],
                h5_path="fi.h5",
                label_csv="label.csv",
            ),
            cls.root,
            "cfg",
        )
        cls.h5_path = cls.root / "fi.h5"

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        self._out = tempfile.TemporaryDirectory()
        self.out = Path(self._out.name)
        self.addCleanup(self._out.cleanup)

    def write_configs(self, task="bigfive", **train_overrides):
        data_path = self.out / "data.yaml"
        data_path.write_text(
            yaml.safe_dump(
                {
                    "h5_path": str(self.h5_path),
                    "features": ["wavlm", "dinov2"],
                    "batch_size": 2,
                    "num_workers": 0,
                    "pin_memory": False,
                    "persistent_workers": False,
                    "window_seconds": 1.0,
                }
            )
        )
        model_path = self.out / "model.yaml"
        model_path.write_text(
            yaml.safe_dump(
                {"d_model": 8, "num_heads": 2, "cmt_num_layers": 1, "branch_sat_num_layers": 1}
            )
        )
        train_config = {
            "task": task,
            "max_epochs": 1,
            "accelerator": "cpu",
            "output_dir": str(self.out / "results"),
            "early_stopping": {"enabled": False},
            "mlflow": {
                "tracking_uri": f"sqlite:///{self.out / 'mlflow.db'}",
                "experiment_name": f"cli-test-{task}",
                "log_model": False,
            },
        }
        train_config.update(train_overrides)
        train_path = self.out / "train.yaml"
        train_path.write_text(yaml.safe_dump(train_config))
        return data_path, model_path, train_path

    def config(self, task="bigfive", overrides=None, **train_overrides):
        paths = self.write_configs(task, **train_overrides)
        return ExperimentConfig.from_files(*paths, overrides=overrides)


class TestReadInputDims(CliTestCase):
    def test_reads_widths_from_the_dataset_not_the_config(self):
        dims = read_input_dims(DataConfig(h5_path=str(self.h5_path), features=["wavlm", "dinov2"]))
        self.assertEqual(dims, [8, 8])

    def test_order_follows_the_configured_features(self):
        a = read_input_dims(DataConfig(h5_path=str(self.h5_path), features=["wavlm", "dinov2"]))
        b = read_input_dims(DataConfig(h5_path=str(self.h5_path), features=["dinov2", "wavlm"]))
        self.assertEqual(len(a), len(b))

    def test_missing_dataset_says_how_to_build_it(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            read_input_dims(DataConfig(h5_path=str(self.out / "nope.h5"), features=["wavlm"]))
        self.assertIn("make hdf5-fi", str(ctx.exception))

    def test_unknown_feature_lists_what_is_available(self):
        with self.assertRaises(KeyError) as ctx:
            read_input_dims(DataConfig(h5_path=str(self.h5_path), features=["marlin"]))
        self.assertIn("marlin", str(ctx.exception))


class TestBuildCallbacks(CliTestCase):
    def test_checkpoint_selected_by_the_primary_metric(self):
        from lightning.pytorch.callbacks import ModelCheckpoint

        callbacks = build_callbacks(self.config(), self.out)
        monitors = {c.monitor for c in callbacks if isinstance(c, ModelCheckpoint)}
        self.assertIn(MONITOR, monitors)
        self.assertEqual(MONITOR, "valid/mean_ccc")

    def test_early_stopping_is_configurable(self):
        from lightning.pytorch.callbacks import EarlyStopping

        without = build_callbacks(self.config(), self.out)
        self.assertFalse(any(isinstance(c, EarlyStopping) for c in without))

        with_es = build_callbacks(
            self.config(early_stopping={"enabled": True, "patience": 2}), self.out
        )
        stoppers = [c for c in with_es if isinstance(c, EarlyStopping)]
        self.assertEqual(len(stoppers), 1)
        self.assertEqual(stoppers[0].patience, 2)

    def test_checkpoints_are_resumable_not_weights_only(self):
        from lightning.pytorch.callbacks import ModelCheckpoint

        callbacks = build_callbacks(self.config(), self.out)
        for callback in callbacks:
            if isinstance(callback, ModelCheckpoint):
                self.assertFalse(callback.save_weights_only)


class TestRun(CliTestCase):
    def test_bigfive_end_to_end_produces_metrics_and_artifacts(self):
        metrics = run(self.config(task="bigfive"))

        self.assertIn("test/mean_one_minus_mae", metrics)
        for trait in ("openness", "emotional_stability"):
            self.assertIn(f"test/one_minus_mae/{trait}", metrics)

        # Outputs live under a per-experiment subdir so the modality groups keep
        # separate, independently-resumable checkpoints.
        run_dir = self.out / "results" / "bigfive" / "cli-test-bigfive"
        self.assertTrue((run_dir / "metrics_test.json").is_file())
        self.assertTrue((run_dir / "test_predictions.csv").is_file())
        self.assertTrue((run_dir / "time.json").is_file())
        self.assertTrue(list((run_dir / "checkpoints").glob("*.ckpt")))

    def test_interview_end_to_end_is_a_separate_task(self):
        metrics = run(self.config(task="interview"))

        self.assertIn("test/one_minus_mae/interview", metrics)
        self.assertNotIn("test/one_minus_mae/openness", metrics)
        self.assertTrue((self.out / "results" / "interview" / "cli-test-interview").is_dir())

    def test_metrics_json_matches_the_returned_metrics(self):
        metrics = run(self.config(task="bigfive"))
        run_dir = self.out / "results" / "bigfive" / "cli-test-bigfive"
        written = json.loads((run_dir / "metrics_test.json").read_text())
        self.assertAlmostEqual(
            written["test/mean_one_minus_mae"], metrics["test/mean_one_minus_mae"], places=5
        )

    def test_one_minus_mae_is_in_range(self):
        metrics = run(self.config(task="bigfive"))
        value = metrics["test/mean_one_minus_mae"]
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_fast_dev_run_short_circuits(self):
        self.assertEqual(run(self.config(task="bigfive"), fast_dev_run=True), {})

    def test_accumulate_grad_batches_trains_end_to_end(self):
        """A large effective batch size (via accumulation) must not crash the
        step-based warmup/cosine schedule -- see the steps_per_epoch // fix
        in cli.py that divides batch count by accumulate_grad_batches."""
        metrics = run(self.config(task="bigfive", accumulate_grad_batches=2))
        self.assertIn("test/mean_one_minus_mae", metrics)

    def test_window_override_changes_the_window_without_a_rebuild(self):
        before = self.h5_path.stat().st_mtime_ns

        config = self.config(task="bigfive", overrides={"data.window_seconds": 0.5})
        self.assertEqual(config.data.window_seconds, 0.5)
        run(config)

        self.assertEqual(self.h5_path.stat().st_mtime_ns, before, "the h5 was rebuilt")

    def test_run_is_logged_to_mlflow_with_its_dataset_hash(self):
        import mlflow

        run(self.config(task="bigfive"))

        mlflow.set_tracking_uri(f"sqlite:///{self.out / 'mlflow.db'}")
        runs = mlflow.search_runs(experiment_names=["cli-test-bigfive"])
        self.assertEqual(len(runs), 1)
        self.assertIn("params.data.h5_md5", runs.columns)
        self.assertTrue(runs.iloc[0]["params.data.h5_md5"])
        self.assertIn("metrics.test/mean_one_minus_mae", runs.columns)


class TestResume(CliTestCase):
    """Stop-and-resume: continue a run from last.ckpt into the same MLflow run."""

    def _run_dir(self, task="bigfive"):
        return self.out / "results" / task / f"cli-test-{task}"

    def test_fresh_run_persists_its_mlflow_run_id(self):
        run(self.config(task="bigfive"))
        run_id_file = self._run_dir() / RUN_ID_FILE
        self.assertTrue(run_id_file.is_file())
        self.assertTrue(run_id_file.read_text().strip())

    def test_resume_continues_the_same_mlflow_run(self):
        import mlflow

        # First launch (planned for 2 epochs), records a run id and a last.ckpt.
        run(self.config(task="bigfive", max_epochs=2))
        first_id = (self._run_dir() / RUN_ID_FILE).read_text().strip()
        self.assertTrue((self._run_dir() / "checkpoints" / "last.ckpt").is_file())

        # Resume with the SAME horizon: must reuse the run, not mint a new one.
        run(self.config(task="bigfive", max_epochs=2), resume=True)

        mlflow.set_tracking_uri(f"sqlite:///{self.out / 'mlflow.db'}")
        runs = mlflow.search_runs(experiment_names=["cli-test-bigfive"])
        self.assertEqual(len(runs), 1, "resume created a second run instead of continuing")
        self.assertEqual((self._run_dir() / RUN_ID_FILE).read_text().strip(), first_id)

    def test_resume_without_a_checkpoint_raises(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            run(self.config(task="bigfive"), resume=True)
        self.assertIn("last.ckpt", str(ctx.exception))

    def test_resume_rejects_a_changed_max_epochs(self):
        # OneCycle's LR curve is fixed at first launch; resuming with a different
        # horizon is rejected rather than silently changing the schedule.
        run(self.config(task="bigfive", max_epochs=2))
        with self.assertRaises(ValueError) as ctx:
            run(self.config(task="bigfive", max_epochs=5), resume=True)
        self.assertIn("max_epochs", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
