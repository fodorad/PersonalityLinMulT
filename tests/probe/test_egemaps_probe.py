"""Tests for :mod:`personalitylinmult.probe.egemaps_probe`."""

from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from data.fixtures import write_label_csv

from personalitylinmult.probe.egemaps_probe import (
    align,
    clips_available,
    iter_chunks,
    load_clip,
    main,
    pool_time,
    result_to_dict,
    run_all,
    run_mean_control,
    run_probe,
    select_subset,
    write_results,
    write_subset_csv,
)

D_W, D_E, T_W = 8, 5, 40


def _write_clips(root: Path, ids: list[str], seed: int = 0, plant: bool = True) -> None:
    """Write WavLM pickles (10 layers) and 100 Hz LLDs, linear in layer 9 if planted."""
    rng = np.random.default_rng(seed)
    weight = np.random.default_rng(99).standard_normal((D_W, D_E))
    (root / "wavlm").mkdir(parents=True, exist_ok=True)
    (root / "egemaps").mkdir(parents=True, exist_ok=True)
    for video_id in ids:
        layers = [rng.standard_normal((T_W, D_W)).astype(np.float32) for _ in range(10)]
        with (root / "wavlm" / f"{video_id}.pkl").open("wb") as fh:
            pickle.dump(layers, fh)
        lld = layers[9] @ weight if plant else rng.standard_normal((T_W, D_E))
        lld = np.repeat(lld, 2, axis=0) + 0.01 * rng.standard_normal((2 * T_W, D_E))
        np.save(root / "egemaps" / f"{video_id}.npy", lld.astype(np.float32))


class PoolAlignTest(unittest.TestCase):
    def test_pool_shape_and_values(self) -> None:
        x = np.arange(12, dtype=np.float32).reshape(6, 2)
        pooled = pool_time(x)
        self.assertEqual(pooled.shape, (3, 2))
        np.testing.assert_allclose(pooled[0], [1.0, 2.0])

    def test_pool_drops_odd_trailing_frame(self) -> None:
        self.assertEqual(pool_time(np.zeros((7, 4))).shape, (3, 4))

    def test_align_trims_to_shortest(self) -> None:
        w, e = align(np.zeros((50, 8)), np.zeros((60, 5)))
        self.assertEqual((w.shape, e.shape), ((30, 8), (30, 5)))
        w, e = align(np.zeros((20, 8)), np.zeros((100, 5)))
        self.assertEqual((w.shape[0], e.shape[0]), (20, 20))

    def test_align_caps_at_max_frames(self) -> None:
        w, e = align(np.zeros((900, 8)), np.zeros((1800, 5)))
        self.assertEqual((w.shape[0], e.shape[0]), (750, 750))
        self.assertEqual(w.dtype, torch.float32)


class SubsetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ids = [f"c{i:03d}" for i in range(60)]
        self.subsets = ["train"] * 30 + ["valid"] * 15 + ["test"] * 15

    def test_deterministic_and_stratified(self) -> None:
        a = select_subset(self.ids, self.subsets, 10, 5, seed=1)
        b = select_subset(self.ids, self.subsets, 10, 5, seed=1)
        self.assertEqual(a, b)
        self.assertEqual(sum(i < "c030" for i in a), 10)
        self.assertEqual(sum("c030" <= i < "c045" for i in a), 5)
        self.assertFalse(any(i >= "c045" for i in a))
        self.assertNotEqual(a, select_subset(self.ids, self.subsets, 10, 5, seed=2))

    def test_caps_at_availability(self) -> None:
        self.assertEqual(len(select_subset(self.ids, self.subsets, 999, 999)), 45)

    def test_csv_keeps_subset_and_all_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            label = Path(tmp) / "label.csv"
            write_label_csv(label, self.ids, self.subsets)
            chosen = select_subset(self.ids, self.subsets, 4, 2)
            out = Path(tmp) / "sub.csv"
            write_subset_csv(label, out, chosen)
            frame = pd.read_csv(out, dtype={"video_id": str})
            self.assertEqual(list(frame.columns), list(pd.read_csv(label).columns))
            self.assertEqual(len(frame), 4 + 2 + 15)
            self.assertEqual(int((frame["subset"] == "test").sum()), 15)


class ProbeTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.ids = [f"c{i:02d}" for i in range(18)]
        self.subsets = ["train"] * 10 + ["valid"] * 4 + ["test"] * 4
        _write_clips(self.root, self.ids)
        self.splits = {
            s: [v for v, k in zip(self.ids, self.subsets, strict=True) if k == s]
            for s in ("train", "valid", "test")
        }
        self.wavlm = self.root / "wavlm"
        self.egemaps = self.root / "egemaps"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_load_clip_shapes(self) -> None:
        w, e = load_clip("c00", self.wavlm, self.egemaps)
        self.assertEqual((w.shape, e.shape), ((T_W, D_W), (T_W, D_E)))

    def test_iter_chunks_reverse_swaps(self) -> None:
        x, y = next(iter_chunks(["c00"], self.wavlm, self.egemaps, reverse=True))
        self.assertEqual((x.shape[1], y.shape[1]), (D_E, D_W))

    def test_forward_recovers_planted_relation(self) -> None:
        res = run_probe(self.splits, self.wavlm, self.egemaps)
        self.assertEqual(res.test.r2.shape, (D_E,))
        self.assertEqual(res.test.pearson_r.shape, (D_E,))
        self.assertGreater(res.test.mean_r2, 0.9)

    def test_reverse_shapes(self) -> None:
        res = run_probe(self.splits, self.wavlm, self.egemaps, reverse=True)
        self.assertEqual(res.test.r2.shape, (D_W,))
        self.assertLessEqual(res.test.weighted_r2, 1.0)

    def test_mean_control_is_about_zero(self) -> None:
        res = run_mean_control(self.splits, self.wavlm, self.egemaps)
        self.assertLess(abs(res.test.mean_r2), 0.1)
        self.assertEqual(res.alpha, float("inf"))

    def test_unrelated_targets_score_low(self) -> None:
        _write_clips(self.root, self.ids, plant=False)
        res = run_probe(self.splits, self.wavlm, self.egemaps)
        self.assertLess(res.test.mean_r2, 0.2)

    def test_no_train_clips_raises(self) -> None:
        with self.assertRaises(ValueError):
            run_probe({**self.splits, "train": []}, self.wavlm, self.egemaps)

    def test_result_dict_and_writer(self) -> None:
        res = run_probe(self.splits, self.wavlm, self.egemaps)
        names = [f"lld{i}" for i in range(D_E)]
        d = result_to_dict(res, names)
        self.assertEqual(set(d), {"alpha", "valid_mean_r2", "test"})
        self.assertEqual(
            set(d["test"]), {"mean_r2", "mean_pearson_r", "weighted_r2", "r2", "pearson_r"}
        )
        self.assertEqual(list(d["test"]["r2"]), names)
        self.assertNotIn("r2", result_to_dict(res)["test"])
        out = self.root / "out"
        write_results({"forward": d}, out)
        loaded = json.loads((out / "probe_results.json").read_text())
        self.assertEqual(set(loaded["forward"]), {"alpha", "valid_mean_r2", "test"})
        self.assertIn("| forward |", (out / "probe_results.md").read_text())

    def _run_all(self, random_dir: Path) -> dict[str, dict]:
        label = self.root / "label.csv"
        write_label_csv(label, self.ids, self.subsets)
        return run_all(
            label_csv=label,
            wavlm_dir=self.wavlm,
            egemaps_dir=self.egemaps,
            out_dir=self.root / "res",
            subset_csv=self.root / "subset.csv",
            random_wavlm_dir=random_dir,
            names=[f"lld{i}" for i in range(D_E)],
            n_train=6,
            n_valid=2,
        )

    def test_run_all_end_to_end(self) -> None:
        _write_clips(self.root / "rand", self.ids, seed=5, plant=False)
        results = self._run_all(self.root / "rand" / "wavlm")
        self.assertEqual(
            set(results),
            {
                "train_mean",
                "forward_full",
                "forward_subset",
                "forward_subset_random_init",
                "reverse_full",
            },
        )
        self.assertTrue((self.root / "subset.csv").is_file())
        self.assertTrue((self.root / "res" / "probe_results.json").is_file())
        self.assertNotIn("r2", results["reverse_full"]["test"])

    def test_run_all_skips_missing_random_dir(self) -> None:
        results = self._run_all(self.root / "nope")
        self.assertNotIn("forward_subset_random_init", results)

    def test_run_all_skips_incomplete_random_dir(self) -> None:
        # A random-init extraction still in progress: the directory exists but a
        # test clip is missing. The control must be skipped, not crash the run.
        _write_clips(self.root / "rand", self.ids, seed=5, plant=False)
        (self.root / "rand" / "wavlm" / f"{self.ids[-1]}.pkl").unlink()
        results = self._run_all(self.root / "rand" / "wavlm")
        self.assertNotIn("forward_subset_random_init", results)
        self.assertIn("reverse_full", results)
        self.assertTrue((self.root / "res" / "probe_results.json").is_file())


class ClipsAvailableTest(unittest.TestCase):
    def test_reports_missing_clips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "a.pkl").write_bytes(b"")
            self.assertTrue(clips_available(directory, ["a"]))
            self.assertFalse(clips_available(directory, ["a", "b"]))
            self.assertFalse(clips_available(directory / "nope", ["a"]))
            self.assertTrue(clips_available(directory, []))


class MainTest(unittest.TestCase):
    def test_main_runs_on_a_processed_tree(self) -> None:
        ids = [f"c{i:02d}" for i in range(18)]
        subsets = ["train"] * 10 + ["valid"] * 4 + ["test"] * 4
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_clips(root, ids)
            (root / "wavlm").rename(root / "wavlm_baseplus_l9")
            (root / "egemaps").rename(root / "egemaps_z")
            write_label_csv(root / "label.csv", ids, subsets)
            main(
                [
                    "--root", str(root),
                    "--processed", ".",
                    "--out-dir", "res",
                    "--n-train", "6",
                    "--n-valid", "2",
                ]
            )  # fmt: skip
            results = json.loads((root / "res" / "probe_results.json").read_text())
            self.assertIn("forward_full", results)
            self.assertNotIn("forward_subset_random_init", results)
            self.assertTrue((root / "label_probe_subset.csv").is_file())


if __name__ == "__main__":
    unittest.main()
