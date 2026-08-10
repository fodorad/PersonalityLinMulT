"""Tests for the config-driven HDF5 builder, against real fixture files."""

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import yaml

from personalitylinmult.data.builder import (
    BuildError,
    build,
    dataset_spec_from_config,
    load_labels,
)
from personalitylinmult.data.schema import SCHEMA_VERSION, DatasetSpec, FeatureSpec
from personalitylinmult.data.sources import FeatureReader

from . import fixtures

VIDEO_IDS = ["clip0", "clip1", "clip2", "clip3", "clip4", "clip5"]


def wavlm_spec(layers=(9,)) -> FeatureSpec:
    return FeatureSpec(
        name="wavlm",
        source_dir="wavlm_baseplus",
        source_format="pickle_layers",
        dim=8,
        rate_hz=fixtures.WAVLM_RATE_HZ,
        layers=layers,
    )


def dinov2_spec(**overrides) -> FeatureSpec:
    return FeatureSpec(
        name="dinov2",
        source_dir="dinov2_base",
        source_format="safetensors",
        dim=8,
        rate_hz=fixtures.DINOV2_RATE_HZ,
        **overrides,
    )


def mmbert_spec() -> FeatureSpec:
    return FeatureSpec(
        name="mmbert_gt",
        source_dir=".",
        source_format="hdf5_per_id",
        source_file="mmbert_gt.h5",
        dim=8,
        rate_hz=None,
    )


def xlm_spec() -> FeatureSpec:
    return FeatureSpec(
        name="xlm_roberta_gt",
        source_dir=".",
        source_format="hdf5_per_id",
        source_file="xlm_roberta_gt.h5",
        dim=8,
        rate_hz=None,
        max_time_dim=12,
    )


class BuilderTestCase(unittest.TestCase):
    """Writes a full set of real fixture features into a temp root."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

        fixtures.write_label_csv(self.root / "label.csv", VIDEO_IDS)
        fixtures.write_wavlm(self.root / "wavlm_baseplus", VIDEO_IDS, dim=8)
        fixtures.write_emotion2vec(self.root / "emotion2vecplus", VIDEO_IDS, dim=8)
        fixtures.write_dinov2(self.root / "dinov2_base", VIDEO_IDS, dim=8)
        fixtures.write_masked_visual(self.root / "farl_ep64", VIDEO_IDS, dim=8, n_invalid=3)
        fixtures.write_text_h5(self.root / "mmbert_gt.h5", VIDEO_IDS, dim=8)
        fixtures.write_text_h5(self.root / "xlm_roberta_gt.h5", VIDEO_IDS, dim=8)

    def build(self, features, config_yaml="cfg: v1", **kwargs):
        spec = DatasetSpec(features=features, h5_path="fi.h5", label_csv="label.csv")
        stats = build(spec, self.root, config_yaml, **kwargs)
        return self.root / "fi.h5", stats


class TestConfigDrivenSelection(BuilderTestCase):
    """The core requirement: only configured features enter the file."""

    def test_unlisted_features_are_absent(self):
        h5_path, _ = self.build([wavlm_spec(), dinov2_spec()])

        with h5py.File(h5_path, "r") as fh:
            names = set(fh["features"].keys())

        self.assertEqual(names, {"wavlm", "dinov2"})
        self.assertNotIn("emotion2vec", names)
        self.assertNotIn("farl", names)
        self.assertNotIn("mmbert_gt", names)

    def test_single_feature_dataset(self):
        h5_path, stats = self.build([dinov2_spec()])
        with h5py.File(h5_path, "r") as fh:
            self.assertEqual(set(fh["features"].keys()), {"dinov2"})
        self.assertEqual(stats.n_features, 1)

    def test_every_configured_feature_is_present(self):
        h5_path, _ = self.build([wavlm_spec(), dinov2_spec(), mmbert_spec(), xlm_spec()])
        with h5py.File(h5_path, "r") as fh:
            self.assertEqual(
                set(fh["features"].keys()),
                {"wavlm", "dinov2", "mmbert_gt", "xlm_roberta_gt"},
            )


class TestStorageLayout(BuilderTestCase):
    def test_shapes_dtype_and_one_sample_per_chunk(self):
        h5_path, _ = self.build([dinov2_spec()])

        with h5py.File(h5_path, "r") as fh:
            data = fh["features/dinov2/data"]
            self.assertEqual(data.shape, (len(VIDEO_IDS), fixtures.VISUAL_T, 8))
            self.assertEqual(data.dtype, np.float16)
            # One chunk == one sample: random reads touch exactly one chunk.
            self.assertEqual(data.chunks, (1, fixtures.VISUAL_T, 8))

    def test_multi_layer_gets_layer_axis(self):
        h5_path, _ = self.build([wavlm_spec(layers=(6, 7, 8))])

        with h5py.File(h5_path, "r") as fh:
            data = fh["features/wavlm/data"]
            self.assertEqual(data.shape, (len(VIDEO_IDS), 3, fixtures.AUDIO_T, 8))
            self.assertEqual(data.chunks, (1, 3, fixtures.AUDIO_T, 8))
            np.testing.assert_array_equal(fh["features/wavlm"].attrs["layers"], np.array([6, 7, 8]))

    def test_token_text_stored_as_padded_sequence(self):
        h5_path, _ = self.build([xlm_spec()])

        with h5py.File(h5_path, "r") as fh:
            group = fh["features/xlm_roberta_gt"]
            # Token-level text: (N, max_time_dim, D), padded and length-recorded.
            self.assertEqual(group["data"].shape, (len(VIDEO_IDS), 12, 8))
            self.assertEqual(group["mask"].shape, (len(VIDEO_IDS), 12))
            # fixtures write 3..8 tokens; each masked to its true length.
            np.testing.assert_array_equal(group["length"][:], np.arange(3, 9))
            self.assertEqual(int(group["mask"][0].sum()), 3)

    def test_ragged_text_is_padded_to_max_and_length_recorded(self):
        h5_path, _ = self.build([mmbert_spec()])

        with h5py.File(h5_path, "r") as fh:
            group = fh["features/mmbert_gt"]
            # fixtures write 3..8 tokens for 6 clips
            self.assertEqual(group["data"].shape, (len(VIDEO_IDS), 8, 8))
            np.testing.assert_array_equal(group["length"][:], np.arange(3, 9))
            # Padding is masked out, real tokens are valid.
            self.assertEqual(int(group["mask"][0].sum()), 3)
            self.assertTrue(group["mask"][0, :3].all())
            self.assertFalse(group["mask"][0, 3:].any())

    def test_mask_is_written_for_dense_features_too(self):
        """The network contract is feature + mask per feature, uniformly."""
        h5_path, _ = self.build([dinov2_spec()])
        with h5py.File(h5_path, "r") as fh:
            mask = fh["features/dinov2/mask"]
            self.assertEqual(mask.shape, (len(VIDEO_IDS), fixtures.VISUAL_T))
            self.assertTrue(mask[:].all())

    def test_real_mask_is_preserved(self):
        farl = FeatureSpec(
            name="farl",
            source_dir="farl_ep64",
            source_format="safetensors",
            dim=8,
            rate_hz=fixtures.DINOV2_RATE_HZ,
        )
        h5_path, _ = self.build([farl])
        with h5py.File(h5_path, "r") as fh:
            mask = fh["features/farl/mask"][0]
        self.assertEqual(int((~mask).sum()), 3)

    def test_values_round_trip_through_fp16(self):
        h5_path, _ = self.build([dinov2_spec()])
        source = fixtures.VISUAL_T

        with h5py.File(h5_path, "r") as fh:
            stored = fh["features/dinov2/data"][0]

        self.assertEqual(stored.shape, (source, 8))
        self.assertTrue(np.isfinite(stored).all())
        self.assertFalse(np.allclose(stored, 0.0))


class TestProbeEfficiency(BuilderTestCase):
    """The probe must not re-read every clip: wavlm costs ~0.34s per read."""

    def test_fixed_length_feature_probes_one_clip_not_all(self):
        reads = []
        original = FeatureReader.read

        def counting_read(self, video_id):
            reads.append((self.spec.name, video_id))
            return original(self, video_id)

        FeatureReader.read = counting_read
        self.addCleanup(setattr, FeatureReader, "read", original)

        self.build([dinov2_spec()])

        # 1 coverage pass (no reads) + 1 hint read + 6 write reads == 7.
        # A full probe pass would make it 13.
        self.assertEqual(len(reads), 1 + len(VIDEO_IDS))

    def test_ragged_feature_still_probes_every_clip(self):
        reads = []
        original = FeatureReader.read

        def counting_read(self, video_id):
            reads.append((self.spec.name, video_id))
            return original(self, video_id)

        FeatureReader.read = counting_read
        self.addCleanup(setattr, FeatureReader, "read", original)

        self.build([mmbert_spec()])

        # Text has no rate_hz, so its padding target genuinely needs a full pass:
        # 6 probe reads + 6 write reads.
        self.assertEqual(len(reads), 2 * len(VIDEO_IDS))

    def test_configured_max_time_dim_skips_the_probe_read_entirely(self):
        """A declared length is the answer; there is nothing left to measure."""
        reads = []
        original = FeatureReader.read

        def counting_read(self, video_id):
            reads.append((self.spec.name, video_id))
            return original(self, video_id)

        FeatureReader.read = counting_read
        self.addCleanup(setattr, FeatureReader, "read", original)

        self.build([dinov2_spec(max_time_dim=fixtures.VISUAL_T)])

        # Write reads only -- no hint read, unlike the measured case above.
        self.assertEqual(len(reads), len(VIDEO_IDS))

    def test_configured_max_time_dim_sets_the_stored_length(self):
        """The config pins T, so windows land on round numbers by construction."""
        pinned = fixtures.VISUAL_T - 3
        h5_path, _ = self.build([dinov2_spec(max_time_dim=pinned)])

        with h5py.File(h5_path, "r") as fh:
            self.assertEqual(fh["features/dinov2/data"].shape[1], pinned)
            self.assertEqual(int(fh["features/dinov2"].attrs["max_time_dim"]), pinned)
            # Truncated frames are cut, not silently marked valid.
            self.assertTrue(fh["features/dinov2/mask"][0].all())
            self.assertEqual(int(fh["features/dinov2/length"][0]), pinned)

    def test_width_mismatch_still_caught_on_a_later_clip(self):
        """The probe samples clip0, so a bad clip3 must fail during the write."""
        import numpy as np
        from safetensors.numpy import save_file

        save_file(
            {
                "frame_ids": np.arange(fixtures.VISUAL_T, dtype=np.int64),
                "features": np.zeros((fixtures.VISUAL_T, 4), dtype=np.float32),
            },
            str(self.root / "dinov2_base" / "clip3.st"),
        )

        with self.assertRaises(BuildError) as ctx:
            self.build([dinov2_spec()])
        self.assertIn("clip3", str(ctx.exception))


class TestVariableClipLength(BuilderTestCase):
    """FI clips vary 15.32-15.36s, so a few frames of slack is normal."""

    def test_longer_clip_is_truncated_to_stored_length(self):
        import numpy as np
        from safetensors.numpy import save_file

        longer = fixtures.VISUAL_T + 4
        save_file(
            {
                "frame_ids": np.arange(longer, dtype=np.int64),
                "features": np.ones((longer, 8), dtype=np.float32),
            },
            str(self.root / "dinov2_base" / "clip3.st"),
        )

        h5_path, _ = self.build([dinov2_spec()])

        with h5py.File(h5_path, "r") as fh:
            self.assertEqual(fh["features/dinov2/data"].shape[1], fixtures.VISUAL_T)
            self.assertEqual(int(fh["features/dinov2/length"][3]), fixtures.VISUAL_T)
            self.assertTrue(fh["features/dinov2/mask"][3].all())

    def test_shorter_clip_is_padded_and_masked(self):
        import numpy as np
        from safetensors.numpy import save_file

        shorter = fixtures.VISUAL_T - 5
        save_file(
            {
                "frame_ids": np.arange(shorter, dtype=np.int64),
                "features": np.ones((shorter, 8), dtype=np.float32),
            },
            str(self.root / "dinov2_base" / "clip3.st"),
        )

        h5_path, _ = self.build([dinov2_spec()])

        with h5py.File(h5_path, "r") as fh:
            self.assertEqual(int(fh["features/dinov2/length"][3]), shorter)
            mask = fh["features/dinov2/mask"][3]
            data = fh["features/dinov2/data"][3]
        self.assertTrue(mask[:shorter].all())
        self.assertFalse(mask[shorter:].any())
        np.testing.assert_allclose(data[shorter:], 0.0)

        # Post-padding: the clip sits at the FRONT and the zeros at the back.
        # Asserting only that the tail is zero would also pass if the array were
        # pre-padded and the source happened to be zeros -- the source is written
        # as ones above precisely so this distinguishes the two.
        np.testing.assert_allclose(data[:shorter], 1.0)


class TestAttributes(BuilderTestCase):
    def test_root_attrs_make_the_file_traceable(self):
        h5_path, _ = self.build([dinov2_spec()], config_yaml="features: [dinov2]\n")

        with h5py.File(h5_path, "r") as fh:
            self.assertEqual(int(fh.attrs["schema_version"]), SCHEMA_VERSION)
            self.assertEqual(fh.attrs["builder_config_yaml"], "features: [dinov2]\n")
            self.assertTrue(fh.attrs["created_utc"])
            self.assertTrue(fh.attrs["git_sha"])
            self.assertTrue(fh.attrs["source_label_csv_sha256"])

    def test_feature_attrs_round_trip(self):
        h5_path, _ = self.build([wavlm_spec(), mmbert_spec()])

        with h5py.File(h5_path, "r") as fh:
            wavlm = fh["features/wavlm"].attrs
            self.assertEqual(int(wavlm["dim"]), 8)
            self.assertEqual(int(wavlm["max_time_dim"]), fixtures.AUDIO_T)
            self.assertAlmostEqual(float(wavlm["rate_hz"]), fixtures.WAVLM_RATE_HZ, places=3)

            # Text has no wall-clock rate: sentinel -1.0 means "never windowed".
            self.assertEqual(float(fh["features/mmbert_gt"].attrs["rate_hz"]), -1.0)

    def test_git_sha_marks_a_dirty_tree(self):
        import subprocess
        import tempfile

        from personalitylinmult.data.builder import git_sha

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = lambda *a: subprocess.run(  # noqa: E731
                ["git", *a], cwd=root, check=True, capture_output=True
            )
            run("init")
            run("config", "user.email", "t@t")
            run("config", "user.name", "t")
            (root / "a.txt").write_text("1")
            run("add", "a.txt")
            run("commit", "-m", "init")
            self.assertFalse(git_sha(root).endswith("-dirty"))

            (root / "a.txt").write_text("2")
            self.assertTrue(git_sha(root).endswith("-dirty"))

    def test_git_sha_is_unknown_outside_a_checkout(self):
        import tempfile

        from personalitylinmult.data.builder import git_sha

        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(git_sha(Path(tmp)), "unknown")

    def test_labels_include_all_traits(self):
        h5_path, _ = self.build([dinov2_spec()])
        with h5py.File(h5_path, "r") as fh:
            self.assertIn("emotional_stability", fh["labels"])
            self.assertIn("neuroticism", fh["labels"])
            self.assertIn("interview", fh["labels"])
            es = fh["labels/emotional_stability"][:]
            n = fh["labels/neuroticism"][:]
        np.testing.assert_allclose(es, 1.0 - n, atol=1e-5)

    def test_subset_codes_and_ids(self):
        h5_path, _ = self.build([dinov2_spec()])
        with h5py.File(h5_path, "r") as fh:
            ids = [v.decode() for v in fh["video_id"][:]]
            subsets = fh["subset"][:]
        self.assertEqual(ids, VIDEO_IDS)
        # fixtures assign round-robin train/valid/test
        np.testing.assert_array_equal(subsets, np.array([0, 1, 2, 0, 1, 2], dtype=np.uint8))


class TestCoverageEnforcement(BuilderTestCase):
    def test_missing_features_hard_fail_and_name_the_ids(self):
        """A partially-extracted feature must never be silently zero-filled."""
        (self.root / "dinov2_base" / "clip3.st").unlink()
        (self.root / "dinov2_base" / "clip5.st").unlink()

        with self.assertRaises(BuildError) as ctx:
            self.build([dinov2_spec()])

        message = str(ctx.exception)
        self.assertIn("2/6", message)
        self.assertIn("clip3", message)
        self.assertIn("will not zero-fill", message)

    def test_no_file_is_left_behind_on_failure(self):
        (self.root / "dinov2_base" / "clip0.st").unlink()
        with self.assertRaises(BuildError):
            self.build([dinov2_spec()])
        self.assertFalse((self.root / "fi.h5").exists())

    def test_declared_dim_mismatch_raises(self):
        wrong = FeatureSpec(
            name="dinov2",
            source_dir="dinov2_base",
            source_format="safetensors",
            dim=999,
            rate_hz=fixtures.DINOV2_RATE_HZ,
        )
        with self.assertRaises(BuildError) as ctx:
            self.build([wrong])
        self.assertIn("999", str(ctx.exception))


class TestLimitAndIdempotency(BuilderTestCase):
    def test_limit_truncates(self):
        h5_path, stats = self.build([dinov2_spec()], limit=3)
        self.assertEqual(stats.n_samples, 3)
        with h5py.File(h5_path, "r") as fh:
            self.assertEqual(fh["video_id"].shape[0], 3)
            self.assertEqual(fh["labels/openness"].shape[0], 3)
            self.assertEqual(fh["features/dinov2/data"].shape[0], 3)

    def test_limit_keeps_every_split(self):
        """label.csv is grouped by split: a naive head() would be train-only."""
        grouped = ["train", "train", "valid", "valid", "test", "test"]
        fixtures.write_label_csv(self.root / "label.csv", VIDEO_IDS, grouped)

        h5_path, _ = self.build([dinov2_spec()], limit=3)

        with h5py.File(h5_path, "r") as fh:
            subsets = set(fh["subset"][:].tolist())
        self.assertEqual(subsets, {0, 1, 2})

    def test_limit_labels_stay_aligned_with_their_clips(self):
        """A stratified limit must not shuffle labels away from their rows."""
        import pandas as pd

        frame = pd.read_csv(self.root / "label.csv")
        h5_path, _ = self.build([dinov2_spec()], limit=3)

        with h5py.File(h5_path, "r") as fh:
            ids = [v.decode() for v in fh["video_id"][:]]
            openness = fh["labels/openness"][:]

        for i, video_id in enumerate(ids):
            expected = float(frame.loc[frame.video_id == video_id, "openness"].iloc[0])
            self.assertAlmostEqual(float(openness[i]), expected, places=5)

    def test_rebuild_with_same_config_is_a_noop(self):
        h5_path, _ = self.build([dinov2_spec()], config_yaml="v1")
        first_mtime = h5_path.stat().st_mtime_ns

        self.build([dinov2_spec()], config_yaml="v1")

        self.assertEqual(h5_path.stat().st_mtime_ns, first_mtime)

    def test_changed_config_triggers_rebuild(self):
        h5_path, _ = self.build([dinov2_spec()], config_yaml="v1")

        self.build([wavlm_spec()], config_yaml="v2")

        with h5py.File(h5_path, "r") as fh:
            self.assertEqual(set(fh["features"].keys()), {"wavlm"})

    def test_overwrite_forces_rebuild(self):
        h5_path, _ = self.build([dinov2_spec()], config_yaml="v1")
        with h5py.File(h5_path, "r+") as fh:
            fh["features/dinov2/data"][0] = 0.0

        self.build([dinov2_spec()], config_yaml="v1", overwrite=True)

        with h5py.File(h5_path, "r") as fh:
            self.assertFalse(np.allclose(fh["features/dinov2/data"][0], 0.0))


class TestLoadLabels(BuilderTestCase):
    def test_missing_csv_raises(self):
        with self.assertRaises(BuildError):
            load_labels(self.root / "nope.csv")

    def test_missing_column_raises(self):
        path = self.root / "bad.csv"
        path.write_text("video_id,subset\na,train\n")
        with self.assertRaises(BuildError) as ctx:
            load_labels(path)
        self.assertIn("missing required columns", str(ctx.exception))

    def test_unknown_subset_raises(self):
        fixtures.write_label_csv(self.root / "odd.csv", ["a"], subsets=["holdout"])
        with self.assertRaises(BuildError) as ctx:
            load_labels(self.root / "odd.csv")
        self.assertIn("holdout", str(ctx.exception))

    def test_empty_dataset_raises(self):
        fixtures.write_label_csv(self.root / "label.csv", [])
        with self.assertRaises(BuildError):
            self.build([dinov2_spec()])


class TestConfigLoading(BuilderTestCase):
    def _write_configs(self, feature_names, extra=None):
        config_dir = self.root / "config" / "FI"
        (config_dir / "features").mkdir(parents=True, exist_ok=True)
        (config_dir / "data").mkdir(parents=True, exist_ok=True)

        specs = {
            "dinov2": {
                "name": "dinov2",
                "source_dir": "dinov2_base",
                "source_format": "safetensors",
                "dim": 8,
                "rate_hz": fixtures.DINOV2_RATE_HZ,
            },
            "wavlm": {
                "name": "wavlm",
                "source_dir": "wavlm_baseplus",
                "source_format": "pickle_layers",
                "dim": 8,
                "rate_hz": fixtures.WAVLM_RATE_HZ,
                "layers": [9],
            },
        }
        for name in specs:
            (config_dir / "features" / f"{name}.yaml").write_text(yaml.safe_dump(specs[name]))

        data_config = {"h5_path": "fi.h5", "label_csv": "label.csv", "features": feature_names}
        data_config.update(extra or {})
        path = config_dir / "data" / "set.yaml"
        path.write_text(yaml.safe_dump(data_config))
        return path

    def test_loads_only_named_features(self):
        path = self._write_configs(["dinov2"])
        spec, config_yaml = dataset_spec_from_config(path)
        self.assertEqual(spec.feature_names, ["dinov2"])
        self.assertIn("dinov2", config_yaml)

    def test_feature_order_is_preserved(self):
        path = self._write_configs(["wavlm", "dinov2"])
        spec, _ = dataset_spec_from_config(path)
        self.assertEqual(spec.feature_names, ["wavlm", "dinov2"])

    def test_end_to_end_from_config(self):
        path = self._write_configs(["dinov2"])
        spec, config_yaml = dataset_spec_from_config(path)
        build(spec, self.root, config_yaml)
        with h5py.File(self.root / "fi.h5", "r") as fh:
            self.assertEqual(set(fh["features"].keys()), {"dinov2"})

    def test_missing_config_raises(self):
        with self.assertRaises(BuildError):
            dataset_spec_from_config(self.root / "nope.yaml")

    def test_missing_feature_config_raises(self):
        path = self._write_configs(["ghost"])
        with self.assertRaises(BuildError) as ctx:
            dataset_spec_from_config(path)
        self.assertIn("ghost", str(ctx.exception))

    def test_missing_required_key_raises(self):
        path = self.root / "bare.yaml"
        path.write_text(yaml.safe_dump({"features": ["dinov2"]}))
        with self.assertRaises(BuildError) as ctx:
            dataset_spec_from_config(path)
        self.assertIn("h5_path", str(ctx.exception))

    def test_invalid_feature_config_raises(self):
        config_dir = self.root / "config" / "FI"
        (config_dir / "features").mkdir(parents=True, exist_ok=True)
        (config_dir / "data").mkdir(parents=True, exist_ok=True)
        (config_dir / "features" / "broken.yaml").write_text(
            yaml.safe_dump(
                {"name": "broken", "source_dir": "d", "source_format": "telepathy", "dim": 8}
            )
        )
        path = config_dir / "data" / "set.yaml"
        path.write_text(yaml.safe_dump({"h5_path": "fi.h5", "features": ["broken"]}))

        with self.assertRaises(BuildError) as ctx:
            dataset_spec_from_config(path)
        self.assertIn("telepathy", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
