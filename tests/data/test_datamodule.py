"""Tests for the Lightning DataModule, against a real built HDF5 file."""

import tempfile
import unittest
from pathlib import Path

import torch

from personalitylinmult.data.builder import build
from personalitylinmult.data.datamodule import FiDataModule
from personalitylinmult.data.schema import BIG_FIVE_TRAITS, DatasetSpec
from personalitylinmult.train.config import DataConfig

from . import fixtures
from .test_builder import dinov2_spec, wavlm_spec

VIDEO_IDS = [f"clip{i}" for i in range(12)]
SUBSETS = ["train"] * 6 + ["valid"] * 3 + ["test"] * 3


class DataModuleTestCase(unittest.TestCase):
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

    def make(self, **kwargs):
        params = {
            "h5_path": str(self.h5_path),
            "features": ["wavlm", "dinov2"],
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "window_seconds": 1.0,
        }
        params.update(kwargs)
        return FiDataModule(DataConfig(**params), target_names=list(BIG_FIVE_TRAITS))


class TestSetup(DataModuleTestCase):
    def test_fit_builds_train_and_valid(self):
        dm = self.make()
        dm.setup("fit")
        self.assertIsNotNone(dm.train_dataset)
        self.assertIsNotNone(dm.valid_dataset)
        self.assertEqual(len(dm.train_dataset), 6)
        self.assertEqual(len(dm.valid_dataset), 3)

    def test_test_builds_test_split(self):
        dm = self.make()
        dm.setup("test")
        self.assertEqual(len(dm.test_dataset), 3)

    def test_dataloader_before_setup_raises(self):
        dm = self.make()
        with self.assertRaises(RuntimeError):
            dm.train_dataloader()
        with self.assertRaises(RuntimeError):
            dm.test_dataloader()


class TestBatches(DataModuleTestCase):
    def test_train_batch_shape_and_mask_pairing(self):
        dm = self.make()
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))

        self.assertEqual(batch["wavlm"].shape[0], 2)
        self.assertEqual(batch["dinov2"].shape[0], 2)
        self.assertEqual(batch["targets"].shape, (2, 5))
        self.assertEqual(len(batch["video_id"]), 2)
        for name in ("wavlm", "dinov2"):
            self.assertIn(f"{name}_mask", batch)
            self.assertEqual(batch[f"{name}_mask"].dtype, torch.bool)

    def test_window_is_applied_per_modality_rate(self):
        dm = self.make(window_seconds=1.0)
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))

        self.assertEqual(batch["wavlm"].shape[1], round(fixtures.WAVLM_RATE_HZ * 1.0))
        self.assertEqual(batch["dinov2"].shape[1], round(fixtures.DINOV2_RATE_HZ * 1.0))

    def test_train_drops_last_partial_batch(self):
        dm = self.make(batch_size=4)
        dm.setup("fit")
        # 6 train clips at batch 4 -> 1 full batch, remainder dropped
        self.assertEqual(len(list(dm.train_dataloader())), 1)

    def test_eval_keeps_every_clip(self):
        dm = self.make(batch_size=2)
        dm.setup(None)
        seen = sum(len(b["video_id"]) for b in dm.val_dataloader())
        self.assertEqual(seen, 3)

    def test_eval_order_is_deterministic(self):
        dm = self.make()
        dm.setup("test")
        first = [b["video_id"] for b in dm.test_dataloader()]
        second = [b["video_id"] for b in dm.test_dataloader()]
        self.assertEqual(first, second)

    def test_train_shuffles(self):
        dm = self.make(batch_size=6, seed=1)
        dm.setup("fit")
        order = [b["video_id"] for b in dm.train_dataloader()][0]
        self.assertNotEqual(order, sorted(order))


class TestWorkers(DataModuleTestCase):
    def test_multiprocess_loading_works(self):
        """Exercises the fork-safe lazy h5 handle for real."""
        dm = self.make(num_workers=2, persistent_workers=True)
        dm.setup("fit")

        batches = list(dm.train_dataloader())

        self.assertTrue(batches)
        self.assertEqual(batches[0]["wavlm"].shape[0], 2)
        self.assertFalse(torch.isnan(batches[0]["wavlm"]).any())

    def test_worker_batches_match_single_process_batches(self):
        single = self.make(num_workers=0, batch_size=3)
        single.setup("test")
        expected = {
            vid: t for b in single.test_dataloader() for vid, t in zip(b["video_id"], b["wavlm"])
        }

        parallel = self.make(num_workers=2, batch_size=3)
        parallel.setup("test")
        got = {
            vid: t for b in parallel.test_dataloader() for vid, t in zip(b["video_id"], b["wavlm"])
        }

        self.assertEqual(set(expected), set(got))
        for vid in expected:
            torch.testing.assert_close(expected[vid], got[vid])


if __name__ == "__main__":
    unittest.main()
