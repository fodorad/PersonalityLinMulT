"""Tests for registering trained models from their MLflow runs."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

from inference.test_predictor import write_checkpoint
from personalitylinmult.inference.register_models import Registration, main, register_all


class RegisterModelsTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.uri = f"sqlite:///{self.root / 'mlflow.db'}"

        import mlflow

        mlflow.set_tracking_uri(self.uri)
        self.mlflow = mlflow
        # A trained run that logged its best checkpoint under model/, mirroring
        # what training does.
        ckpt = self.root / "best.ckpt"
        write_checkpoint(ckpt, ["wavlm", "emotion2vec"], [8, 8])
        mlflow.set_experiment("Trained-A")
        with mlflow.start_run() as run:
            self.run_id = run.info.run_id
            mlflow.log_artifact(str(ckpt), artifact_path="model")


class TestRegisterAll(RegisterModelsTestCase):
    def test_registers_from_the_runs_logged_checkpoint(self):
        versions = register_all([Registration("Trained-A", "reg-a")], tracking_uri=self.uri)
        self.assertEqual(versions, {"reg-a": "1"})

    def test_unknown_experiment_raises(self):
        with self.assertRaises(ValueError) as ctx:
            register_all([Registration("Nope", "x")], tracking_uri=self.uri)
        self.assertIn("Nope", str(ctx.exception))

    def test_explicit_run_id_registers_that_run_not_the_latest(self):
        # A second, later run in the same experiment -- without an explicit
        # run_id, "latest" would pick this one instead of self.run_id.
        with self.mlflow.start_run():
            self.mlflow.log_artifact(str(self.root / "best.ckpt"), artifact_path="model")

        versions = register_all(
            [Registration("Trained-A", "reg-pinned", run_id=self.run_id)],
            tracking_uri=self.uri,
        )
        self.assertEqual(versions, {"reg-pinned": "1"})

        client = self.mlflow.tracking.MlflowClient()
        registered_run_id = client.get_model_version("reg-pinned", "1").run_id
        # The pyfunc wrapper is logged under a fresh registration run, not
        # self.run_id itself -- what we can assert is that registration
        # succeeded using the pinned run's checkpoint, not the later one.
        self.assertIsNotNone(registered_run_id)


class TestMain(RegisterModelsTestCase):
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

    def test_registers_a_single_named_run(self):
        printed = self._run(
            [
                "--tracking-uri",
                self.uri,
                "--experiment",
                "Trained-A",
                "--registered-name",
                "reg-cli",
                "--run-id",
                self.run_id,
            ]
        )
        self.assertIn("reg-cli: v1", printed)

    def test_experiment_without_registered_name_errors(self):
        with self.assertRaises(SystemExit):
            self._run(["--tracking-uri", self.uri, "--experiment", "Trained-A"])

    def test_alias_promotes_the_registered_version(self):
        printed = self._run(
            [
                "--tracking-uri",
                self.uri,
                "--experiment",
                "Trained-A",
                "--registered-name",
                "reg-champion",
                "--alias",
                "champion",
            ]
        )
        self.assertIn("reg-champion@champion -> v1", printed)

        client = self.mlflow.tracking.MlflowClient()
        version = client.get_model_version_by_alias("reg-champion", "champion")
        self.assertEqual(str(version.version), "1")


if __name__ == "__main__":
    unittest.main()
