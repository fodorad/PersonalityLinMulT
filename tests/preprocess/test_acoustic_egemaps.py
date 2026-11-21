"""Tests for the eGeMAPS extraction helpers and the data configs.

The extraction itself needs openSMILE and real audio, so it is not exercised
here; the resume logic, atomic writes and name file are pure and are covered,
together with the data configs that consume the extractor's output.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from personalitylinmult.data.builder import dataset_spec_from_config
from personalitylinmult.preprocess.fi.acoustic_egemaps import (
    is_pending,
    load_video_ids,
    save_array,
    save_names,
    select_shard,
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config/FI/data"


class TestLoadVideoIds(unittest.TestCase):
    def test_keeps_order_and_drops_appledouble(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "label.csv"
            csv.write_text("video_id,subset\nb,train\n._b,train\na,test\n")
            self.assertEqual(load_video_ids(csv), ["b", "a"])


class TestIsPending(unittest.TestCase):
    def test_missing_output_is_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(is_pending(Path(tmp) / "a.npy", overwrite=False))

    def test_existing_output_is_skipped_unless_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.npy"
            path.touch()
            self.assertFalse(is_pending(path, overwrite=False))
            self.assertTrue(is_pending(path, overwrite=True))


class TestSaveArray(unittest.TestCase):
    def test_roundtrip_is_float32_and_leaves_no_temp_file(self):
        array = np.random.default_rng(0).normal(size=(30, 25))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.npy"
            save_array(array, path)
            loaded = np.load(path)
            self.assertEqual(loaded.dtype, np.float32)
            np.testing.assert_allclose(loaded, array, rtol=1e-6)
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["a.npy"])


class TestSelectShard(unittest.TestCase):
    def test_shards_partition_without_overlap(self):
        ids = [f"clip{i}" for i in range(11)]
        shards = [select_shard(ids, i, 4) for i in range(4)]
        self.assertEqual(sorted(sum(shards, [])), sorted(ids))
        self.assertLessEqual(max(map(len, shards)) - min(map(len, shards)), 1)

    def test_single_shard_is_identity(self):
        self.assertEqual(select_shard(["a", "b"], 0, 1), ["a", "b"])

    def test_invalid_shard_raises(self):
        with self.assertRaises(ValueError):
            select_shard(["a"], 2, 2)
        with self.assertRaises(ValueError):
            select_shard(["a"], 0, 0)


class TestSaveNames(unittest.TestCase):
    def test_writes_json_list_creating_parents(self):
        names = [f"lld{i}" for i in range(25)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "names.json"
            save_names(names, path)
            self.assertEqual(json.loads(path.read_text()), names)


class TestDataConfigs(unittest.TestCase):
    def test_both_configs_parse_and_share_the_experiment_h5(self):
        egemaps, _ = dataset_spec_from_config(CONFIG_DIR / "egemaps.yaml")
        fusion, _ = dataset_spec_from_config(CONFIG_DIR / "wavlm_egemaps.yaml")
        self.assertEqual([f.name for f in egemaps.features], ["egemaps"])
        self.assertEqual([f.name for f in fusion.features], ["wavlm", "egemaps"])
        self.assertEqual(egemaps.h5_path, fusion.h5_path)
        self.assertEqual(Path(egemaps.h5_path).name, "exp_egemaps.h5")


if __name__ == "__main__":
    unittest.main()
