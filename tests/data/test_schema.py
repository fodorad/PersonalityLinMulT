"""Tests for the shared feature/dataset schema."""

import unittest

from personalitylinmult.data.schema import (
    BIG_FIVE_TRAITS,
    SUBSET_CODES,
    DatasetSpec,
    FeatureSpec,
    SchemaError,
)


def make_spec(**overrides) -> FeatureSpec:
    base = {
        "name": "dinov2",
        "source_dir": "data/processed/FI/dinov2_base",
        "source_format": "safetensors",
        "dim": 768,
        "rate_hz": 25.02,
    }
    base.update(overrides)
    return FeatureSpec(**base)


class TestTraits(unittest.TestCase):
    def test_big_five_uses_emotional_stability_not_neuroticism(self):
        self.assertIn("emotional_stability", BIG_FIVE_TRAITS)
        self.assertNotIn("neuroticism", BIG_FIVE_TRAITS)
        self.assertEqual(len(BIG_FIVE_TRAITS), 5)

    def test_subset_codes_are_stable(self):
        self.assertEqual(SUBSET_CODES, {"train": 0, "valid": 1, "test": 2})


class TestFeatureSpecValidation(unittest.TestCase):
    def test_valid_spec(self):
        spec = make_spec()
        self.assertEqual(spec.name, "dinov2")
        self.assertFalse(spec.is_multi_layer)

    def test_rejects_unknown_source_format(self):
        with self.assertRaises(SchemaError):
            make_spec(source_format="parquet")

    def test_rejects_non_positive_dim(self):
        with self.assertRaises(SchemaError):
            make_spec(dim=0)

    def test_rejects_non_positive_rate(self):
        with self.assertRaises(SchemaError):
            make_spec(rate_hz=0.0)

    def test_rejects_non_positive_max_time_dim(self):
        with self.assertRaises(SchemaError):
            make_spec(max_time_dim=0)

    def test_max_time_dim_defaults_to_measuring_from_the_data(self):
        self.assertIsNone(make_spec().max_time_dim)

    def test_rejects_empty_name(self):
        with self.assertRaises(SchemaError):
            make_spec(name="")

    def test_layers_require_pickle_layers_format(self):
        with self.assertRaises(SchemaError):
            make_spec(layers=(9,))

    def test_pickle_layers_requires_explicit_layers(self):
        with self.assertRaises(SchemaError):
            FeatureSpec(
                name="wavlm",
                source_dir="d",
                source_format="pickle_layers",
                dim=768,
                rate_hz=49.79,
            )

    def test_hdf5_per_id_requires_source_file(self):
        with self.assertRaises(SchemaError):
            FeatureSpec(
                name="mmbert_gt",
                source_dir="data/processed/FI",
                source_format="hdf5_per_id",
                dim=768,
            )


class TestMultiLayer(unittest.TestCase):
    def test_single_layer_is_not_multi_layer(self):
        spec = FeatureSpec(
            name="wavlm",
            source_dir="d",
            source_format="pickle_layers",
            dim=768,
            rate_hz=49.79,
            layers=(9,),
        )
        self.assertFalse(spec.is_multi_layer)

    def test_several_layers_is_multi_layer(self):
        spec = FeatureSpec(
            name="wavlm",
            source_dir="d",
            source_format="pickle_layers",
            dim=768,
            rate_hz=49.79,
            layers=(6, 7, 8),
        )
        self.assertTrue(spec.is_multi_layer)


class TestWindowFrames(unittest.TestCase):
    def test_dinov2_15s_is_375_frames(self):
        self.assertEqual(make_spec(rate_hz=25.02).window_frames(15.0), 375)

    def test_wavlm_15s_is_747_frames(self):
        spec = FeatureSpec(
            name="wavlm",
            source_dir="d",
            source_format="pickle_layers",
            dim=768,
            rate_hz=49.79,
            layers=(9,),
        )
        self.assertEqual(spec.window_frames(15.0), 747)

    def test_shorter_windows_scale(self):
        spec = make_spec(rate_hz=25.02)
        self.assertEqual(spec.window_frames(10.0), 250)
        self.assertEqual(spec.window_frames(5.0), 125)

    def test_no_window_returns_none(self):
        self.assertIsNone(make_spec().window_frames(None))

    def test_feature_without_rate_is_never_windowed(self):
        text = FeatureSpec(
            name="mmbert_gt",
            source_dir="data/processed/FI",
            source_format="hdf5_per_id",
            source_file="mmbert_gt.h5",
            dim=768,
            rate_hz=None,
        )
        self.assertIsNone(text.window_frames(15.0))

    def test_window_is_at_least_one_frame(self):
        self.assertEqual(make_spec(rate_hz=1.0).window_frames(0.01), 1)


class TestFeatureSpecFromDict(unittest.TestCase):
    def test_round_trip(self):
        spec = FeatureSpec.from_dict(
            {
                "name": "wavlm",
                "source_dir": "data/processed/FI/wavlm_baseplus",
                "source_format": "pickle_layers",
                "dim": 768,
                "rate_hz": 49.79,
                "layers": [9],
            }
        )
        self.assertEqual(spec.layers, (9,))
        self.assertEqual(spec.dtype, "float16")

    def test_rejects_unknown_keys(self):
        with self.assertRaises(SchemaError) as ctx:
            FeatureSpec.from_dict(
                {
                    "name": "x",
                    "source_dir": "d",
                    "source_format": "npy",
                    "dim": 4,
                    "recursive": True,
                }
            )
        self.assertIn("recursive", str(ctx.exception))

    def test_rejects_missing_keys(self):
        with self.assertRaises(SchemaError):
            FeatureSpec.from_dict({"name": "x"})


class TestDatasetSpec(unittest.TestCase):
    def test_feature_lookup_and_order(self):
        a = make_spec(name="a")
        b = make_spec(name="b")
        ds = DatasetSpec(features=[a, b], h5_path="out.h5", label_csv="label.csv")
        self.assertEqual(ds.feature_names, ["a", "b"])
        self.assertIs(ds.feature("b"), b)

    def test_unknown_feature_raises(self):
        ds = DatasetSpec(features=[make_spec()], h5_path="o.h5", label_csv="l.csv")
        with self.assertRaises(KeyError):
            ds.feature("nope")

    def test_rejects_empty_feature_list(self):
        with self.assertRaises(SchemaError):
            DatasetSpec(features=[], h5_path="o.h5", label_csv="l.csv")

    def test_rejects_duplicate_names(self):
        with self.assertRaises(SchemaError):
            DatasetSpec(features=[make_spec(), make_spec()], h5_path="o.h5", label_csv="l.csv")


if __name__ == "__main__":
    unittest.main()
