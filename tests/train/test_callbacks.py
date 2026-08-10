"""Tests for the training callbacks, driven by a real Trainer."""

import json
import tempfile
import unittest
from pathlib import Path

import lightning as L
from data.test_builder import dinov2_spec, wavlm_spec

from data import fixtures
from personalitylinmult.data.builder import build
from personalitylinmult.data.datamodule import FiDataModule
from personalitylinmult.data.schema import BIG_FIVE_TRAITS, DatasetSpec
from personalitylinmult.train.callbacks import (
    PlotCallback,
    PredictionWriter,
    TimeTrackingCallback,
)
from personalitylinmult.train.config import DataConfig, ModelConfig, TrainConfig
from personalitylinmult.train.module import FiLightningModule

VIDEO_IDS = [f"clip{i}" for i in range(9)]
SUBSETS = ["train"] * 3 + ["valid"] * 3 + ["test"] * 3
FEATURES = ["wavlm", "dinov2"]


class CallbackTestCase(unittest.TestCase):
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
        self.output_dir = Path(self._out.name)
        self.addCleanup(self._out.cleanup)

    def _fit_and_test(self, callbacks, targets=None, max_epochs=1):
        targets = targets or list(BIG_FIVE_TRAITS)
        datamodule = FiDataModule(
            DataConfig(
                h5_path=str(self.h5_path),
                features=FEATURES,
                batch_size=2,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
                window_seconds=1.0,
            ),
            target_names=targets,
        )
        module = FiLightningModule(
            model_config=ModelConfig(
                d_model=8, num_heads=2, cmt_num_layers=1, branch_sat_num_layers=1
            ),
            train_config=TrainConfig(
                task="bigfive" if len(targets) == 5 else "interview",
                targets=targets,
                max_epochs=max_epochs,
            ),
            feature_names=FEATURES,
            input_feature_dim=[8, 8],
            steps_per_epoch=1,
        )
        trainer = L.Trainer(
            max_epochs=max_epochs,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            callbacks=callbacks,
            default_root_dir=self.output_dir,
        )
        trainer.fit(module, datamodule=datamodule)
        trainer.test(module, datamodule=datamodule, verbose=False)
        return module


class TestTimeTracking(CallbackTestCase):
    def test_writes_timings_for_each_epoch(self):
        callback = TimeTrackingCallback(self.output_dir)
        self._fit_and_test([callback], max_epochs=2)

        path = self.output_dir / "time.json"
        self.assertTrue(path.is_file())

        data = json.loads(path.read_text())
        self.assertEqual(len(data["epochs"]), 2)
        self.assertGreater(data["total_seconds"], 0)
        self.assertGreater(data["mean_epoch_seconds"], 0)
        self.assertEqual([e["epoch"] for e in data["epochs"]], [0, 1])


class TestPredictionWriter(CallbackTestCase):
    def test_writes_one_row_per_test_clip_with_ids(self):
        """This artifact is what lets significance tests run without re-running."""
        self._fit_and_test([PredictionWriter(self.output_dir)])

        import pandas as pd

        frame = pd.read_csv(self.output_dir / "test_predictions.csv")

        self.assertEqual(len(frame), 3)
        self.assertEqual(set(frame["video_id"]), set(VIDEO_IDS[6:]))
        for trait in BIG_FIVE_TRAITS:
            self.assertIn(f"pred_{trait}", frame.columns)
            self.assertIn(f"true_{trait}", frame.columns)

    def test_targets_written_match_the_source_labels(self):
        self._fit_and_test([PredictionWriter(self.output_dir)])

        import pandas as pd

        written = pd.read_csv(self.output_dir / "test_predictions.csv").set_index("video_id")
        source = pd.read_csv(self.root / "label.csv").set_index("video_id")

        for video_id in written.index:
            for trait in BIG_FIVE_TRAITS:
                self.assertAlmostEqual(
                    float(written.loc[video_id, f"true_{trait}"]),
                    float(source.loc[video_id, trait]),
                    places=4,
                )

    def test_predictions_are_within_the_target_range(self):
        self._fit_and_test([PredictionWriter(self.output_dir)])

        import pandas as pd

        frame = pd.read_csv(self.output_dir / "test_predictions.csv")
        for trait in BIG_FIVE_TRAITS:
            values = frame[f"pred_{trait}"]
            self.assertTrue((values >= 0).all() and (values <= 1).all())

    def test_interview_task_writes_only_its_own_column(self):
        self._fit_and_test([PredictionWriter(self.output_dir)], targets=["interview"])

        import pandas as pd

        frame = pd.read_csv(self.output_dir / "test_predictions.csv")
        self.assertEqual(list(frame.columns), ["video_id", "pred_interview", "true_interview"])


class TestPlotCallback(CallbackTestCase):
    def test_writes_a_scatter_plot(self):
        self._fit_and_test([PlotCallback(self.output_dir)])
        path = self.output_dir / "test_scatter.png"
        self.assertTrue(path.is_file())
        self.assertGreater(path.stat().st_size, 0)


class TestCallbackOrdering(CallbackTestCase):
    def test_writer_and_plotter_coexist_in_either_order(self):
        """Both read the metric state at test end; neither may consume it."""
        for callbacks in (
            [PredictionWriter(self.output_dir), PlotCallback(self.output_dir)],
            [PlotCallback(self.output_dir), PredictionWriter(self.output_dir)],
        ):
            with self.subTest(order=[type(c).__name__ for c in callbacks]):
                for path in self.output_dir.glob("*"):
                    if path.is_file():
                        path.unlink()

                self._fit_and_test(callbacks)

                self.assertTrue((self.output_dir / "test_predictions.csv").is_file())
                self.assertTrue((self.output_dir / "test_scatter.png").is_file())

    def test_repeated_test_runs_do_not_accumulate(self):
        module = self._fit_and_test([PredictionWriter(self.output_dir)])
        preds, _ = module.test_metrics.predictions()
        self.assertEqual(preds.shape[0], 3)

        datamodule = FiDataModule(
            DataConfig(
                h5_path=str(self.h5_path),
                features=FEATURES,
                batch_size=2,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
                window_seconds=1.0,
            ),
            target_names=list(BIG_FIVE_TRAITS),
        )
        trainer = L.Trainer(
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            callbacks=[PredictionWriter(self.output_dir)],
        )
        trainer.test(module, datamodule=datamodule, verbose=False)

        preds, _ = module.test_metrics.predictions()
        self.assertEqual(preds.shape[0], 3, "test state accumulated across runs")


class TestPredictionWriterEdgeCases(CallbackTestCase):
    def test_id_count_mismatch_degrades_gracefully(self):
        module = self._fit_and_test([PredictionWriter(self.output_dir)])
        module.test_video_ids = ["only-one"]

        writer = PredictionWriter(self.output_dir)
        with self.assertLogs("personalitylinmult.train.callbacks", level="WARNING"):
            writer.on_test_epoch_end(None, module)

        import pandas as pd

        frame = pd.read_csv(self.output_dir / "test_predictions.csv", keep_default_na=False)
        self.assertEqual(len(frame), 3)

    def test_plot_with_no_predictions_is_a_noop(self):
        module = self._fit_and_test([])
        module.test_metrics.reset()

        PlotCallback(self.output_dir).on_test_epoch_end(None, module)

        self.assertFalse((self.output_dir / "test_scatter.png").exists())


if __name__ == "__main__":
    unittest.main()
