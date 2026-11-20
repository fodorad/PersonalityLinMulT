"""Tests for eGeMAPS train-only z-score normalisation."""

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import yaml

from personalitylinmult.data.builder import build
from personalitylinmult.data.egemaps import (
    STD_FLOOR,
    EgemapsError,
    EgemapsStats,
    apply_zscore,
    fit_zscore_stats,
    main,
    normalize_directory,
)
from personalitylinmult.data.schema import DatasetSpec, FeatureSpec

from . import fixtures

D = 25
FEATURE_YAML = Path(__file__).resolve().parents[2] / "config/FI/features/egemaps.yaml"


def random_clips(seed: int = 0, lengths: tuple[int, ...] = (40, 55, 31)) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [(rng.standard_normal((t, D)) * 3 + 5).astype(np.float32) for t in lengths]


class TestFitZscoreStats(unittest.TestCase):
    def test_matches_concatenated_statistics(self):
        clips = random_clips()
        stats = fit_zscore_stats(clips)
        all_frames = np.concatenate(clips).astype(np.float64)
        np.testing.assert_allclose(stats.mean, all_frames.mean(axis=0), rtol=1e-10)
        np.testing.assert_allclose(stats.std, all_frames.std(axis=0), rtol=1e-10)

    def test_ignores_padded_frames(self):
        clips = random_clips()
        expected = fit_zscore_stats(clips)
        padded, masks = [], []
        for clip in clips:
            pad = np.full((10, D), 1e6, dtype=np.float32)
            padded.append(np.concatenate([clip, pad]))
            masks.append(np.arange(len(clip) + 10) < len(clip))
        stats = fit_zscore_stats(padded, masks=masks)
        np.testing.assert_allclose(stats.mean, expected.mean)
        np.testing.assert_allclose(stats.std, expected.std)

    def test_extra_clip_changes_statistics(self):
        """Documents why callers must pass train clips only."""
        clips = random_clips()
        outlier = np.full((20, D), 1e3, dtype=np.float32)
        self.assertFalse(
            np.allclose(fit_zscore_stats(clips).mean, fit_zscore_stats([*clips, outlier]).mean)
        )

    def test_zero_variance_lld_is_finite(self):
        clips = random_clips()
        for clip in clips:
            clip[:, 3] = 7.0
        stats = fit_zscore_stats(clips)
        self.assertGreaterEqual(stats.std[3], STD_FLOOR)
        out = apply_zscore(clips[0], stats)
        self.assertTrue(np.isfinite(out).all())
        np.testing.assert_allclose(out[:, 3], 0.0, atol=1e-5)

    def test_default_and_custom_names(self):
        clips = random_clips()
        self.assertEqual(fit_zscore_stats(clips).names[:2], ("lld_00", "lld_01"))
        names = [f"n{i}" for i in range(D)]
        self.assertEqual(fit_zscore_stats(clips, names=names).names, tuple(names))

    def test_empty_input_raises(self):
        with self.assertRaises(EgemapsError):
            fit_zscore_stats([])

    def test_all_frames_masked_raises(self):
        clip = random_clips()[0]
        with self.assertRaises(EgemapsError):
            fit_zscore_stats([clip], masks=[np.zeros(len(clip), dtype=bool)])

    def test_inconsistent_inputs_raise(self):
        clips = random_clips()
        with self.assertRaises(EgemapsError):
            fit_zscore_stats([clips[0], np.zeros((5, D + 1), dtype=np.float32)])
        with self.assertRaises(EgemapsError):
            fit_zscore_stats([np.zeros(5)])
        with self.assertRaises(EgemapsError):
            fit_zscore_stats(clips, masks=[np.ones(len(clips[0]), dtype=bool)])
        with self.assertRaises(EgemapsError):
            fit_zscore_stats(clips[:1], masks=[np.ones(3, dtype=bool)])
        with self.assertRaises(EgemapsError):
            fit_zscore_stats(
                clips[:1], masks=[np.ones(len(clips[0]), dtype=bool), np.ones(2, dtype=bool)]
            )


class TestApplyZscore(unittest.TestCase):
    def test_train_data_has_zero_mean_unit_std(self):
        clips = random_clips()
        stats = fit_zscore_stats(clips)
        z = np.concatenate([apply_zscore(c, stats) for c in clips])
        self.assertEqual(z.dtype, np.float32)
        np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-4)
        np.testing.assert_allclose(z.std(axis=0), 1.0, atol=1e-4)

    def test_wrong_width_raises(self):
        stats = fit_zscore_stats(random_clips())
        with self.assertRaises(EgemapsError):
            apply_zscore(np.zeros((4, D - 1), dtype=np.float32), stats)


class TestStatsSerialisation(unittest.TestCase):
    def test_json_round_trip(self):
        names = [f"n{i}" for i in range(D)]
        stats = fit_zscore_stats(random_clips(), names=names)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "stats.json"
            stats.save(path)
            loaded = EgemapsStats.load(path)
        self.assertEqual(loaded.names, stats.names)
        np.testing.assert_array_equal(loaded.mean, stats.mean)
        np.testing.assert_array_equal(loaded.std, stats.std)

    def test_length_mismatch_raises(self):
        with self.assertRaises(EgemapsError):
            EgemapsStats(names=("a",), mean=np.zeros(2), std=np.ones(2))


class TestNormalizeDirectoryAndBuilder(unittest.TestCase):
    """Raw npy -> z-scored npy -> real h5 built from the shipped feature yaml."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.video_ids = [f"clip{i}" for i in range(6)]
        subsets = ["train", "train", "valid", "valid", "test", "test"]
        fixtures.write_label_csv(self.root / "label.csv", self.video_ids, subsets)
        self.raw_dir = self.root / "egemaps"
        self.raw_dir.mkdir()
        rng = np.random.default_rng(0)
        self.lengths = {}
        for i, video_id in enumerate(self.video_ids):
            self.lengths[video_id] = 40 + 7 * i
            scale = 1000.0 if subsets[i] == "test" else 1.0  # test outlier must not leak
            raw = rng.standard_normal((self.lengths[video_id], D)).astype(np.float32) * scale
            np.save(self.raw_dir / f"{video_id}.npy", raw)
        self.train_ids = self.video_ids[:2]

    def normalize(self) -> EgemapsStats:
        return normalize_directory(
            self.video_ids,
            self.train_ids,
            raw_dir=self.raw_dir,
            out_dir=self.root / "egemaps_z",
            stats_path=self.root / "egemaps_zscore.json",
        )

    def test_stats_come_from_train_clips_only(self):
        stats = self.normalize()
        train = np.concatenate([np.load(self.raw_dir / f"{v}.npy") for v in self.train_ids])
        np.testing.assert_allclose(stats.mean, train.astype(np.float64).mean(axis=0))
        self.assertTrue((self.root / "egemaps_zscore.json").is_file())
        for video_id in self.video_ids:
            self.assertEqual(
                np.load(self.root / "egemaps_z" / f"{video_id}.npy").shape,
                (self.lengths[video_id], D),
            )

    def test_builder_pads_with_exact_zero_and_marks_mask(self):
        self.normalize()
        config = yaml.safe_load(FEATURE_YAML.read_text())
        config["source_dir"] = "egemaps_z"
        config["max_time_dim"] = 80
        spec = FeatureSpec.from_dict(config)
        self.assertEqual((spec.dim, spec.source_format, spec.rate_hz), (D, "npy", 100.0))

        build(DatasetSpec(features=[spec], h5_path="exp.h5", label_csv="label.csv"), self.root, "x")

        with h5py.File(self.root / "exp.h5", "r") as fh:
            data = fh["features/egemaps/data"][:]
            mask = fh["features/egemaps/mask"][:]
        self.assertEqual(data.shape, (6, 80, D))
        for row, video_id in enumerate(self.video_ids):
            length = self.lengths[video_id]
            np.testing.assert_array_equal(mask[row], np.arange(80) < length)
            self.assertTrue((data[row, length:] == 0).all())


class TestCli(TestNormalizeDirectoryAndBuilder):
    def test_main_writes_stats_and_normalised_clips(self):
        processed = Path("proc")
        (self.root / processed).mkdir()
        (self.root / "label.csv").rename(self.root / processed / "label.csv")
        self.raw_dir.rename(self.root / processed / "egemaps")
        names_file = self.root / "names.json"
        names_file.write_text(json.dumps([f"n{i}" for i in range(D)]))

        main(
            [
                "--root",
                str(self.root),
                "--processed",
                str(processed),
                "--names-file",
                str(names_file),
            ]
        )

        stats = EgemapsStats.load(self.root / processed / "egemaps_zscore.json")
        self.assertEqual(stats.names[0], "n0")
        self.assertEqual(len(list((self.root / processed / "egemaps_z").glob("*.npy"))), 6)


class TestFeatureYaml(unittest.TestCase):
    def test_loads_through_feature_spec(self):
        spec = FeatureSpec.from_dict(yaml.safe_load(FEATURE_YAML.read_text()))
        self.assertEqual(spec.name, "egemaps")
        self.assertEqual(spec.source_format, "npy")
        self.assertEqual(spec.dim, 25)
        self.assertEqual(spec.rate_hz, 100.0)
        self.assertEqual(spec.max_time_dim, 1500)


if __name__ == "__main__":
    unittest.main()
