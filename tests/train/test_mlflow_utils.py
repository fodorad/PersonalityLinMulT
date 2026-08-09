"""Tests for the MLflow wiring."""

import tempfile
import unittest
from pathlib import Path

import yaml

from personalitylinmult.train.config import ExperimentConfig
from personalitylinmult.train.mlflow_utils import (
    MAX_PARAM_LENGTH,
    SKIP_ARTIFACT_DIRS,
    build_logger,
    environment_params,
    file_md5,
    git_sha,
    log_artifacts,
    truncate_params,
)


class TestGitSha(unittest.TestCase):
    def test_returns_a_sha_or_unknown(self):
        sha = git_sha()
        self.assertTrue(sha == "unknown" or len(sha) == 40)


class TestFileMd5(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_hashes_content(self):
        path = self.root / "a.bin"
        path.write_bytes(b"hello")
        # md5("hello")
        self.assertEqual(file_md5(path), "5d41402abc4b2a76b9719d911017c592")

    def test_same_content_same_hash(self):
        (self.root / "a").write_bytes(b"x" * 100)
        (self.root / "b").write_bytes(b"x" * 100)
        self.assertEqual(file_md5(self.root / "a"), file_md5(self.root / "b"))

    def test_different_content_different_hash(self):
        (self.root / "a").write_bytes(b"x")
        (self.root / "b").write_bytes(b"y")
        self.assertNotEqual(file_md5(self.root / "a"), file_md5(self.root / "b"))

    def test_missing_file_is_reported_not_raised(self):
        self.assertEqual(file_md5(self.root / "nope"), "missing")

    def test_handles_a_file_larger_than_one_chunk(self):
        path = self.root / "big.bin"
        path.write_bytes(b"z" * (3 * (1 << 20)))
        self.assertEqual(len(file_md5(path, chunk_size=1024)), 32)


class TestTruncateParams(unittest.TestCase):
    def test_leaves_short_values_alone(self):
        params = {"a": 1, "b": "short", "c": 0.5}
        self.assertEqual(truncate_params(params), params)

    def test_truncates_long_values(self):
        value = truncate_params({"a": "x" * (MAX_PARAM_LENGTH + 50)})["a"]
        self.assertEqual(len(value), MAX_PARAM_LENGTH)
        self.assertTrue(value.endswith("..."))

    def test_preserves_types_of_short_values(self):
        result = truncate_params({"n": 42, "f": 1.5, "b": True})
        self.assertIsInstance(result["n"], int)
        self.assertIsInstance(result["f"], float)
        self.assertIsInstance(result["b"], bool)


class MLflowConfigTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

        self.h5 = self.root / "fi.h5"
        import h5py

        with h5py.File(self.h5, "w") as fh:
            fh.attrs["schema_version"] = 1
            fh.attrs["git_sha"] = "abc123"
            fh.attrs["created_utc"] = "2026-01-01T00:00:00+00:00"

        self.data_path = self.root / "data.yaml"
        self.data_path.write_text(
            yaml.safe_dump({"h5_path": str(self.h5), "features": ["wavlm", "dinov2"]})
        )
        self.model_path = self.root / "model.yaml"
        self.model_path.write_text(yaml.safe_dump({"d_model": 32}))
        self.train_path = self.root / "train.yaml"
        self.train_path.write_text(
            yaml.safe_dump(
                {
                    "task": "bigfive",
                    "mlflow": {
                        "tracking_uri": f"sqlite:///{self.root / 'mlflow.db'}",
                        "experiment_name": "unit-test",
                    },
                }
            )
        )

    def config(self):
        return ExperimentConfig.from_files(self.data_path, self.model_path, self.train_path)


class TestEnvironmentParams(MLflowConfigTestCase):
    def test_records_library_versions_and_commit(self):
        params = environment_params(self.config())
        for key in ("env.git_sha", "env.torch", "env.lightning", "env.linmult"):
            self.assertIn(key, params)
            self.assertTrue(params[key])

    def test_pins_the_run_to_its_exact_dataset(self):
        """Two runs with the same config and different data are not comparable."""
        params = environment_params(self.config())
        self.assertEqual(len(params["data.h5_md5"]), 32)
        self.assertGreater(params["data.h5_bytes"], 0)
        self.assertEqual(params["data.h5_schema_version"], 1)
        self.assertEqual(params["data.h5_git_sha"], "abc123")

    def test_missing_dataset_does_not_raise(self):
        self.h5.unlink()
        params = environment_params(self.config())
        self.assertEqual(params["data.h5_md5"], "missing")
        self.assertEqual(params["data.h5_bytes"], 0)

    def test_params_are_mlflow_loggable(self):
        config = self.config()
        params = truncate_params({**config.to_flat_dict(), **environment_params(config)})
        for key, value in params.items():
            with self.subTest(key):
                self.assertNotIsInstance(value, dict | list)
                self.assertLessEqual(len(str(value)), MAX_PARAM_LENGTH)


class TestBuildLoggerAndArtifacts(MLflowConfigTestCase):
    def test_logger_targets_the_configured_experiment(self):
        logger = build_logger(self.config())
        self.assertEqual(logger._experiment_name, "unit-test")

    def test_run_name_override(self):
        logger = build_logger(self.config(), run_name="my-run")
        self.assertEqual(logger._run_name, "my-run")

    def test_logs_files_and_skips_checkpoints(self):
        output = self.root / "out"
        (output / "checkpoints").mkdir(parents=True)
        (output / "plots").mkdir()
        (output / "metrics_test.json").write_text("{}")
        (output / "plots" / "scatter.png").write_bytes(b"png")
        (output / "checkpoints" / "best.ckpt").write_bytes(b"x" * 1000)

        logger = build_logger(self.config())
        log_artifacts(logger, output)

        client = logger.experiment
        names = {a.path for a in client.list_artifacts(logger.run_id)}
        self.assertIn("metrics_test.json", names)
        self.assertIn("plots", names)
        # Checkpoints are logged deliberately, not in bulk.
        self.assertNotIn("checkpoints", names)

    def test_missing_output_dir_is_a_noop(self):
        logger = build_logger(self.config())
        log_artifacts(logger, self.root / "does-not-exist")

    def test_skip_list_names_checkpoints(self):
        self.assertIn("checkpoints", SKIP_ARTIFACT_DIRS)


if __name__ == "__main__":
    unittest.main()
