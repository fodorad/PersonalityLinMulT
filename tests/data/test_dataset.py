"""Tests for the HDF5-only dataset, against a real built file."""

import pickle
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from personalitylinmult.data.builder import build
from personalitylinmult.data.dataset import DatasetError, FiHdf5Dataset
from personalitylinmult.data.schema import DatasetSpec, SchemaError

from . import fixtures
from .test_builder import dinov2_spec, mmbert_spec, wavlm_spec, xlm_spec

VIDEO_IDS = [f"clip{i}" for i in range(9)]
SUBSETS = ["train", "train", "train", "valid", "valid", "valid", "test", "test", "test"]


class DatasetTestCase(unittest.TestCase):
    """Builds one real h5 from real fixture files, shared by the test methods."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)

        fixtures.write_label_csv(cls.root / "label.csv", VIDEO_IDS, SUBSETS)
        fixtures.write_wavlm(cls.root / "wavlm_baseplus", VIDEO_IDS, dim=8)
        fixtures.write_dinov2(cls.root / "dinov2_base", VIDEO_IDS, dim=8)
        fixtures.write_text_h5(cls.root / "mmbert_gt.h5", VIDEO_IDS, dim=8)
        fixtures.write_text_h5(cls.root / "xlm_roberta_gt.h5", VIDEO_IDS, dim=8)

        spec = DatasetSpec(
            features=[wavlm_spec(), dinov2_spec(), mmbert_spec(), xlm_spec()],
            h5_path="fi.h5",
            label_csv="label.csv",
        )
        build(spec, cls.root, "cfg")
        cls.h5_path = cls.root / "fi.h5"

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def make(self, **kwargs):
        params = {
            "h5_path": self.h5_path,
            "subset": "train",
            "feature_names": ["wavlm", "dinov2"],
            "target_names": ["openness", "emotional_stability"],
        }
        params.update(kwargs)
        return FiHdf5Dataset(**params)


class TestSplitting(DatasetTestCase):
    def test_each_split_sees_only_its_own_clips(self):
        for subset, expected in [("train", 3), ("valid", 3), ("test", 3)]:
            with self.subTest(subset):
                ds = self.make(subset=subset)
                self.assertEqual(len(ds), expected)

    def test_splits_are_disjoint_and_cover_everything(self):
        ids = [set(self.make(subset=s).video_ids) for s in ("train", "valid", "test")]
        self.assertEqual(ids[0] | ids[1] | ids[2], set(VIDEO_IDS))
        self.assertEqual(ids[0] & ids[1], set())
        self.assertEqual(ids[1] & ids[2], set())

    def test_unknown_subset_raises(self):
        with self.assertRaises(SchemaError):
            self.make(subset="holdout")


class TestItemContract(DatasetTestCase):
    def test_every_feature_is_paired_with_a_mask(self):
        ds = self.make(feature_names=["wavlm", "dinov2", "mmbert_gt", "xlm_roberta_gt"])
        item = ds[0]
        for name in ("wavlm", "dinov2", "mmbert_gt", "xlm_roberta_gt"):
            self.assertIn(name, item)
            self.assertIn(f"{name}_mask", item)
            self.assertEqual(item[f"{name}_mask"].dtype, torch.bool)

    def test_shapes_and_dtypes(self):
        item = self.make()[0]
        self.assertEqual(item["wavlm"].shape, (fixtures.AUDIO_T, 8))
        self.assertEqual(item["dinov2"].shape, (fixtures.VISUAL_T, 8))
        self.assertEqual(item["wavlm"].dtype, torch.float32)
        self.assertEqual(item["wavlm_mask"].shape, (fixtures.AUDIO_T,))

    def test_fp16_storage_is_cast_up_to_fp32(self):
        with h5py.File(self.h5_path, "r") as fh:
            self.assertEqual(fh["features/wavlm/data"].dtype, np.float16)
        self.assertEqual(self.make()[0]["wavlm"].dtype, torch.float32)

    def test_token_text_presented_as_padded_sequence(self):
        item = self.make(feature_names=["xlm_roberta_gt"])[0]
        # Token-level text: (max_time_dim, D) with a per-token mask.
        self.assertEqual(item["xlm_roberta_gt"].shape, (12, 8))
        self.assertEqual(item["xlm_roberta_gt_mask"].shape, (12,))
        # The mask marks the real (unpadded) tokens as valid.
        self.assertTrue(bool(item["xlm_roberta_gt_mask"].any()))

    def test_multi_layer_feature_keeps_layer_axis(self):
        multi = DatasetSpec(
            features=[wavlm_spec(layers=(6, 7, 8))], h5_path="multi.h5", label_csv="label.csv"
        )
        build(multi, self.root, "multi")
        ds = FiHdf5Dataset(
            h5_path=self.root / "multi.h5",
            subset="train",
            feature_names=["wavlm"],
            target_names=["openness"],
        )
        self.assertEqual(ds[0]["wavlm"].shape, (3, fixtures.AUDIO_T, 8))

    def test_targets_follow_requested_order(self):
        a = self.make(target_names=["openness", "interview"])[0]["targets"]
        b = self.make(target_names=["interview", "openness"])[0]["targets"]
        self.assertEqual(a.shape, (2,))
        torch.testing.assert_close(a, b.flip(0))

    def test_video_id_is_returned(self):
        ds = self.make()
        self.assertEqual(ds[0]["video_id"], ds.video_ids[0])
        self.assertIn(ds[0]["video_id"], VIDEO_IDS)


class TestLabelsMatchSource(DatasetTestCase):
    def test_targets_match_the_label_csv_for_the_right_clip(self):
        import pandas as pd

        frame = pd.read_csv(self.root / "label.csv").set_index("video_id")
        ds = self.make(subset="valid", target_names=["openness", "emotional_stability"])

        for i in range(len(ds)):
            item = ds[i]
            row = frame.loc[item["video_id"]]
            self.assertAlmostEqual(float(item["targets"][0]), float(row["openness"]), places=5)
            self.assertAlmostEqual(
                float(item["targets"][1]), float(row["emotional_stability"]), places=5
            )


class TestFeatureValuesMatchSource(DatasetTestCase):
    def test_wavlm_values_come_from_the_configured_layer(self):
        ds = self.make(feature_names=["wavlm"])
        item = ds[0]

        with (self.root / "wavlm_baseplus" / f"{item['video_id']}.pkl").open("rb") as fh:
            raw = pickle.load(fh)

        np.testing.assert_allclose(item["wavlm"].numpy(), np.asarray(raw[9]), rtol=1e-2, atol=1e-2)
        self.assertFalse(np.allclose(item["wavlm"].numpy(), np.asarray(raw[0]), atol=1e-2))


class TestWindowing(DatasetTestCase):
    def test_no_window_reads_the_whole_clip(self):
        item = self.make(window_seconds=None)[0]
        self.assertEqual(item["wavlm"].shape[0], fixtures.AUDIO_T)
        self.assertEqual(item["dinov2"].shape[0], fixtures.VISUAL_T)

    def test_window_crops_each_feature_by_its_own_rate(self):
        item = self.make(window_seconds=1.0)[0]
        self.assertEqual(item["wavlm"].shape[0], round(fixtures.WAVLM_RATE_HZ * 1.0))
        self.assertEqual(item["dinov2"].shape[0], round(fixtures.DINOV2_RATE_HZ * 1.0))
        self.assertEqual(item["wavlm_mask"].shape[0], round(fixtures.WAVLM_RATE_HZ * 1.0))

    def test_shorter_window_yields_proportionally_fewer_frames(self):
        one = self.make(window_seconds=1.0)[0]
        half = self.make(window_seconds=0.5)[0]
        self.assertLess(half["wavlm"].shape[0], one["wavlm"].shape[0])
        self.assertEqual(half["wavlm"].shape[0], round(fixtures.WAVLM_RATE_HZ * 0.5))

    def test_text_is_never_windowed(self):
        """Text time is tokens, not wall-clock, so a window must not touch it."""
        full = self.make(feature_names=["mmbert_gt"], window_seconds=None)[0]
        cropped = self.make(feature_names=["mmbert_gt"], window_seconds=0.5)[0]
        self.assertEqual(full["mmbert_gt"].shape, cropped["mmbert_gt"].shape)

    def test_window_longer_than_the_clip_raises(self):
        with self.assertRaises(DatasetError) as ctx:
            self.make(window_seconds=999.0)
        self.assertIn("Shorten window_seconds", str(ctx.exception))

    def test_head_crop_takes_the_start_of_the_clip(self):
        full = self.make(window_seconds=None, crop_mode="head")[0]
        cropped = self.make(window_seconds=1.0, crop_mode="head")[0]
        n = cropped["wavlm"].shape[0]
        torch.testing.assert_close(cropped["wavlm"], full["wavlm"][:n])

    def test_center_crop_takes_the_middle(self):
        full = self.make(window_seconds=None)[0]
        cropped = self.make(window_seconds=1.0, crop_mode="center")[0]
        n = cropped["wavlm"].shape[0]
        start = (fixtures.AUDIO_T - n) // 2
        torch.testing.assert_close(cropped["wavlm"], full["wavlm"][start : start + n])

    def test_random_crop_is_deterministic_for_evaluation_splits(self):
        first = self.make(subset="test", window_seconds=1.0, crop_mode="random")[0]
        second = self.make(subset="test", window_seconds=1.0, crop_mode="random")[0]
        torch.testing.assert_close(first["wavlm"], second["wavlm"])

        centered = self.make(subset="test", window_seconds=1.0, crop_mode="center")[0]
        torch.testing.assert_close(first["wavlm"], centered["wavlm"])

    def test_random_crop_on_train_is_seeded_and_reproducible(self):
        a = self.make(subset="train", window_seconds=1.0, crop_mode="random", seed=7)[0]
        b = self.make(subset="train", window_seconds=1.0, crop_mode="random", seed=7)[0]
        torch.testing.assert_close(a["wavlm"], b["wavlm"])

    def test_random_crop_keeps_modalities_wall_clock_aligned(self):
        """Audio and video windows must cover the same span of the clip.

        The offset is drawn once per clip as a fraction of the slack. Drawing it
        per feature in frames instead -- even from one seed -- picks a different
        integer out of each feature's own range, so the audio and video windows
        would silently describe different moments of the same clip.
        """
        ds = self.make(subset="train", window_seconds=1.0, crop_mode="random", seed=7)

        for index in range(len(ds)):
            fraction = ds._crop_fraction(index)
            starts_s = {}
            for view in ds.views:
                if view.rate_hz is None:
                    continue
                starts_s[view.name] = ds._crop_start(view, fraction) / view.rate_hz

            spread = max(starts_s.values()) - min(starts_s.values())
            # One video frame; the rounding floor when aligning a 50 Hz grid to
            # a 25 Hz one. Frame-wise drawing drifts arbitrarily past this.
            self.assertLessEqual(
                spread,
                1.0 / fixtures.DINOV2_RATE_HZ,
                f"clip {index}: modality starts {starts_s} differ by {spread * 1000:.0f} ms",
            )

    def test_unknown_crop_mode_raises(self):
        with self.assertRaises(SchemaError):
            self.make(crop_mode="diagonal")


class TestWorkerSafety(DatasetTestCase):
    def test_handle_is_not_open_after_construction(self):
        """A handle created here would be inherited by forked workers."""
        ds = self.make()
        self.assertIsNone(ds._h5)

    def test_handle_opens_lazily_on_first_read(self):
        ds = self.make()
        ds[0]
        self.assertIsNotNone(ds._h5)

    def test_pickling_drops_the_handle(self):
        ds = self.make()
        ds[0]
        self.assertIsNotNone(ds._h5)

        restored = pickle.loads(pickle.dumps(ds))

        self.assertIsNone(restored._h5)
        self.assertEqual(len(restored), len(ds))
        torch.testing.assert_close(restored[0]["wavlm"], ds[0]["wavlm"])


class TestValidation(DatasetTestCase):
    def test_missing_file_raises(self):
        with self.assertRaises(DatasetError) as ctx:
            self.make(h5_path=self.root / "nope.h5")
        self.assertIn("make hdf5-fi", str(ctx.exception))

    def test_unknown_feature_lists_what_is_available(self):
        with self.assertRaises(DatasetError) as ctx:
            self.make(feature_names=["marlin"])
        message = str(ctx.exception)
        self.assertIn("marlin", message)
        self.assertIn("dinov2", message)

    def test_unknown_target_raises(self):
        with self.assertRaises(DatasetError) as ctx:
            self.make(target_names=["charisma"])
        self.assertIn("charisma", str(ctx.exception))

    def test_schema_version_mismatch_raises(self):
        stale = self.root / "stale.h5"
        import shutil

        shutil.copy(self.h5_path, stale)
        with h5py.File(stale, "r+") as fh:
            fh.attrs["schema_version"] = 999

        with self.assertRaises(DatasetError) as ctx:
            self.make(h5_path=stale)
        self.assertIn("Rebuild", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
