"""Tests for the per-format feature readers, against real fixture files."""

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from personalitylinmult.data.schema import FeatureSpec
from personalitylinmult.data.sources import (
    FeatureReader,
    SourceError,
    is_hidden_sidecar,
    iter_feature_files,
    read_npy,
    read_pickle_layers,
    read_safetensors,
)

from . import fixtures


class TestAppleDoubleFiltering(unittest.TestCase):
    """The ._ sidecars on this volume double every glob and are unreadable."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_is_hidden_sidecar(self):
        self.assertTrue(is_hidden_sidecar(Path("._abc.st")))
        self.assertFalse(is_hidden_sidecar(Path("abc.st")))

    def test_iter_skips_sidecars(self):
        fixtures.write_dinov2(self.root, ["a", "b"])
        fixtures.write_appledouble(self.root / "a.st")
        fixtures.write_appledouble(self.root / "b.st")

        found = iter_feature_files(self.root, ".st")

        self.assertEqual([p.stem for p in found], ["a", "b"])

    def test_missing_directory_raises(self):
        with self.assertRaises(SourceError):
            iter_feature_files(self.root / "nope", ".st")

    def test_available_ids_excludes_sidecars(self):
        fixtures.write_dinov2(self.root, ["a"])
        fixtures.write_appledouble(self.root / "a.st")
        spec = FeatureSpec(
            name="dinov2",
            source_dir=self.root.name,
            source_format="safetensors",
            dim=8,
            rate_hz=25.02,
        )
        with FeatureReader(spec, self.root.parent) as reader:
            self.assertEqual(reader.available_ids(), {"a"})


class TestPickleLayers(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        fixtures.write_wavlm(self.root, ["a"], dim=8)
        self.path = self.root / "a.pkl"

    def test_single_layer_gives_2d(self):
        sample = read_pickle_layers(self.path, (9,))
        self.assertEqual(sample.features.shape, (fixtures.AUDIO_T, 8))
        self.assertEqual(sample.time_dim, fixtures.AUDIO_T)
        self.assertTrue(sample.mask.all())

    def test_multi_layer_gives_3d_in_requested_order(self):
        sample = read_pickle_layers(self.path, (2, 5))
        self.assertEqual(sample.features.shape, (2, fixtures.AUDIO_T, 8))

        with self.path.open("rb") as fh:
            raw = pickle.load(fh)
        np.testing.assert_allclose(sample.features[0], raw[2])
        np.testing.assert_allclose(sample.features[1], raw[5])

    def test_layer_selection_actually_selects(self):
        """A wrong layer index must not silently pass as layer 0."""
        with self.path.open("rb") as fh:
            raw = pickle.load(fh)
        sample = read_pickle_layers(self.path, (7,))
        np.testing.assert_allclose(sample.features, raw[7])
        self.assertFalse(np.allclose(sample.features, raw[0]))

    def test_absent_layer_raises(self):
        with self.assertRaises(SourceError) as ctx:
            read_pickle_layers(self.path, (99,))
        self.assertIn("not in this file", str(ctx.exception))

    def test_unreadable_file_raises(self):
        bad = self.root / "bad.pkl"
        bad.write_bytes(b"not a pickle")
        with self.assertRaises(SourceError):
            read_pickle_layers(bad, (0,))

    def test_pickle_of_the_wrong_shape_raises(self):
        bad = self.root / "scalar.pkl"
        with bad.open("wb") as fh:
            pickle.dump({"not": "a layer payload"}, fh)
        with self.assertRaises(SourceError):
            read_pickle_layers(bad, (0,))


class TestPickleLayersSelfDescribing(unittest.TestCase):
    """The extractor stores only the layers it was asked for, tagged by id.

    A file holding layer 9 alone has that array at position 0, so position can no
    longer stand in for the layer id -- the ids must travel with the data.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        fixtures.write_wavlm(self.root, ["a"], dim=8)
        self.legacy = self.root / "a.pkl"
        with self.legacy.open("rb") as fh:
            self.all_layers = pickle.load(fh)

        self.thin = self.root / "thin.pkl"
        with self.thin.open("wb") as fh:
            pickle.dump({"layers": [9], "features": [self.all_layers[9].astype("float16")]}, fh)

    def test_layer_id_means_the_same_in_both_formats(self):
        """`layers: [9]` must read layer 9 whichever format is on disk."""
        legacy = read_pickle_layers(self.legacy, (9,))
        thin = read_pickle_layers(self.thin, (9,))
        np.testing.assert_allclose(thin.features, legacy.features, rtol=1e-2, atol=1e-2)

    def test_requesting_an_unextracted_layer_fails_loudly(self):
        """Position 0 holds layer 9 -- reading it as layer 0 would be silent corruption."""
        with self.assertRaises(SourceError) as ctx:
            read_pickle_layers(self.thin, (0,))
        message = str(ctx.exception)
        self.assertIn("[9]", message)
        self.assertIn("acoustic-wavlm-fi", message)

    def test_multi_layer_file_keeps_requested_order(self):
        path = self.root / "two.pkl"
        with path.open("wb") as fh:
            pickle.dump(
                {"layers": [6, 9], "features": [self.all_layers[6], self.all_layers[9]]},
                fh,
            )
        sample = read_pickle_layers(path, (9, 6))
        self.assertEqual(sample.features.shape, (2, fixtures.AUDIO_T, 8))
        np.testing.assert_allclose(sample.features[0], self.all_layers[9])
        np.testing.assert_allclose(sample.features[1], self.all_layers[6])

    def test_mismatched_ids_and_arrays_raise(self):
        path = self.root / "broken.pkl"
        with path.open("wb") as fh:
            pickle.dump({"layers": [6, 9], "features": [self.all_layers[6]]}, fh)
        with self.assertRaises(SourceError):
            read_pickle_layers(path, (9,))


class TestNpy(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_reads_2d_with_dense_mask(self):
        fixtures.write_emotion2vec(self.root, ["a"], dim=8)
        sample = read_npy(self.root / "a.npy")
        self.assertEqual(sample.features.shape, (fixtures.AUDIO_T, 8))
        self.assertTrue(sample.mask.all())

    def test_rejects_wrong_rank(self):
        np.save(self.root / "bad.npy", np.zeros((3, 4, 5), dtype=np.float32))
        with self.assertRaises(SourceError):
            read_npy(self.root / "bad.npy")

    def test_missing_file_raises(self):
        with self.assertRaises(SourceError):
            read_npy(self.root / "nope.npy")


class TestSafetensors(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_dinov2_without_mask_gets_dense_mask(self):
        fixtures.write_dinov2(self.root, ["a"], dim=8)
        sample = read_safetensors(self.root / "a.st")
        self.assertEqual(sample.features.shape, (fixtures.VISUAL_T, 8))
        self.assertEqual(sample.mask.shape, (fixtures.VISUAL_T,))
        self.assertTrue(sample.mask.all())

    def test_stored_mask_is_honoured(self):
        fixtures.write_masked_visual(self.root, ["a"], dim=8, n_invalid=3)
        sample = read_safetensors(self.root / "a.st")
        self.assertEqual(int((~sample.mask).sum()), 3)
        self.assertTrue(sample.mask[0])
        self.assertFalse(sample.mask[-1])

    def test_missing_features_key_raises(self):
        save_file({"frame_ids": np.arange(3)}, str(self.root / "bad.st"))
        with self.assertRaises(SourceError):
            read_safetensors(self.root / "bad.st")

    def test_mask_length_mismatch_raises(self):
        save_file(
            {
                "features": np.zeros((5, 8), dtype=np.float32),
                "mask": np.ones((3,), dtype=bool),
            },
            str(self.root / "bad.st"),
        )
        with self.assertRaises(SourceError) as ctx:
            read_safetensors(self.root / "bad.st")
        self.assertIn("does not match", str(ctx.exception))

    def test_appledouble_content_raises_not_silently_passes(self):
        fixtures.write_dinov2(self.root, ["a"], dim=8)
        fixtures.write_appledouble(self.root / "a.st")
        with self.assertRaises(SourceError):
            read_safetensors(self.root / "._a.st")


class TestHdf5PerId(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.ids = ["a", "b", "c"]
        fixtures.write_text_h5(self.root / "feat" / "mmbert_gt.h5", self.ids, dim=8)
        fixtures.write_text_h5(self.root / "feat" / "xlm_roberta_gt.h5", self.ids, dim=8)

    def _spec(self, source_file: str) -> FeatureSpec:
        return FeatureSpec(
            name="text",
            source_dir="feat",
            source_format="hdf5_per_id",
            source_file=source_file,
            dim=8,
            rate_hz=None,
        )

    def test_sequence_entries_are_ragged(self):
        spec = self._spec("mmbert_gt.h5")
        with FeatureReader(spec, self.root) as reader:
            a = reader.read("a")
            c = reader.read("c")
        self.assertEqual(a.features.shape, (3, 8))
        self.assertEqual(c.features.shape, (5, 8))
        self.assertTrue(a.mask.all())

    def test_available_ids(self):
        spec = self._spec("mmbert_gt.h5")
        with FeatureReader(spec, self.root) as reader:
            self.assertEqual(reader.available_ids(), set(self.ids))

    def test_available_ids_without_open_handle(self):
        spec = self._spec("mmbert_gt.h5")
        reader = FeatureReader(spec, self.root)
        self.assertEqual(reader.available_ids(), set(self.ids))

    def test_unknown_id_raises(self):
        spec = self._spec("mmbert_gt.h5")
        with FeatureReader(spec, self.root) as reader, self.assertRaises(SourceError):
            reader.read("missing")

    def test_wrong_rank_raises(self):
        # A 1-D (D,) entry is no longer a valid text feature: text is token-level
        # (T, D), so a rank-1 entry must be rejected.
        import h5py

        bad = self.root / "feat" / "bad_rank.h5"
        with h5py.File(bad, "w") as fh:
            fh.create_dataset("a", data=np.zeros((8,), dtype=np.float32))
        spec = self._spec("bad_rank.h5")
        with FeatureReader(spec, self.root) as reader, self.assertRaises(SourceError):
            reader.read("a")

    def test_use_outside_context_manager_raises(self):
        spec = self._spec("mmbert_gt.h5")
        reader = FeatureReader(spec, self.root)
        with self.assertRaises(SourceError) as ctx:
            reader.read("a")
        self.assertIn("context manager", str(ctx.exception))

    def test_missing_file_raises(self):
        spec = self._spec("nope.h5")
        with self.assertRaises(SourceError), FeatureReader(spec, self.root):
            pass


class TestFeatureReaderDispatch(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_dispatches_each_per_clip_format(self):
        fixtures.write_wavlm(self.root / "wavlm", ["a"], dim=8)
        fixtures.write_emotion2vec(self.root / "e2v", ["a"], dim=8)
        fixtures.write_dinov2(self.root / "dino", ["a"], dim=8)

        cases = [
            (
                FeatureSpec(
                    name="wavlm",
                    source_dir="wavlm",
                    source_format="pickle_layers",
                    dim=8,
                    rate_hz=49.79,
                    layers=(9,),
                ),
                (fixtures.AUDIO_T, 8),
            ),
            (
                FeatureSpec(
                    name="e2v",
                    source_dir="e2v",
                    source_format="npy",
                    dim=8,
                    rate_hz=49.79,
                ),
                (fixtures.AUDIO_T, 8),
            ),
            (
                FeatureSpec(
                    name="dino",
                    source_dir="dino",
                    source_format="safetensors",
                    dim=8,
                    rate_hz=25.02,
                ),
                (fixtures.VISUAL_T, 8),
            ),
        ]
        for spec, expected in cases:
            with self.subTest(spec.name), FeatureReader(spec, self.root) as reader:
                self.assertEqual(reader.read("a").features.shape, expected)
                self.assertEqual(reader.available_ids(), {"a"})


if __name__ == "__main__":
    unittest.main()
