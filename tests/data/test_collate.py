"""Tests for batching and the model-input unpacking contract."""

import unittest

import torch

from personalitylinmult.data.collate import collate_fi, unpack_batch


def sample(video_id: str, t: int = 4, dim: int = 3, valid: int | None = None) -> dict:
    mask = torch.ones((t,), dtype=torch.bool)
    if valid is not None:
        mask[valid:] = False
    return {
        "video_id": video_id,
        "wavlm": torch.randn(t, dim),
        "wavlm_mask": mask,
        "targets": torch.randn(5),
    }


class TestCollate(unittest.TestCase):
    def test_stacks_tensors_and_collects_ids(self):
        batch = collate_fi([sample("a"), sample("b"), sample("c")])

        self.assertEqual(batch["wavlm"].shape, (3, 4, 3))
        self.assertEqual(batch["wavlm_mask"].shape, (3, 4))
        self.assertEqual(batch["targets"].shape, (3, 5))
        self.assertEqual(batch["video_id"], ["a", "b", "c"])

    def test_preserves_dtypes(self):
        batch = collate_fi([sample("a"), sample("b")])
        self.assertEqual(batch["wavlm"].dtype, torch.float32)
        self.assertEqual(batch["wavlm_mask"].dtype, torch.bool)

    def test_preserves_values_per_row(self):
        first, second = sample("a"), sample("b")
        batch = collate_fi([first, second])
        torch.testing.assert_close(batch["wavlm"][0], first["wavlm"])
        torch.testing.assert_close(batch["wavlm"][1], second["wavlm"])

    def test_mask_semantics_survive_collation(self):
        batch = collate_fi([sample("a", valid=2), sample("b", valid=3)])
        self.assertEqual(int(batch["wavlm_mask"][0].sum()), 2)
        self.assertEqual(int(batch["wavlm_mask"][1].sum()), 3)

    def test_multi_layer_features_stack_on_the_batch_axis(self):
        samples = [
            {
                "video_id": v,
                "wavlm": torch.randn(3, 4, 8),
                "wavlm_mask": torch.ones(4, dtype=torch.bool),
                "targets": torch.randn(1),
            }
            for v in ("a", "b")
        ]
        batch = collate_fi(samples)
        self.assertEqual(batch["wavlm"].shape, (2, 3, 4, 8))

    def test_empty_batch_raises(self):
        with self.assertRaises(ValueError):
            collate_fi([])

    def test_single_sample_batch(self):
        batch = collate_fi([sample("a")])
        self.assertEqual(batch["wavlm"].shape, (1, 4, 3))
        self.assertEqual(batch["video_id"], ["a"])


class TestUnpackBatch(unittest.TestCase):
    def _batch(self):
        return {
            "wavlm": torch.randn(2, 4, 3),
            "wavlm_mask": torch.ones(2, 4, dtype=torch.bool),
            "dinov2": torch.randn(2, 6, 3),
            "dinov2_mask": torch.zeros(2, 6, dtype=torch.bool),
            "targets": torch.randn(2, 5),
            "video_id": ["a", "b"],
        }

    def test_returns_feature_mask_and_name_triples_in_order(self):
        inputs, masks, names = unpack_batch(self._batch(), ["wavlm", "dinov2"])

        self.assertEqual(names, ["wavlm", "dinov2"])
        self.assertEqual(len(inputs), 2)
        self.assertEqual(len(masks), 2)
        self.assertEqual(inputs[0].shape, (2, 4, 3))
        self.assertEqual(inputs[1].shape, (2, 6, 3))
        self.assertEqual(masks[0].shape, (2, 4))
        self.assertEqual(masks[1].shape, (2, 6))

    def test_order_follows_the_requested_names_not_the_dict(self):
        inputs, _, names = unpack_batch(self._batch(), ["dinov2", "wavlm"])
        self.assertEqual(names, ["dinov2", "wavlm"])
        self.assertEqual(inputs[0].shape, (2, 6, 3))

    def test_every_feature_gets_a_mask(self):
        _, masks, _ = unpack_batch(self._batch(), ["wavlm", "dinov2"])
        self.assertTrue(all(m.dtype == torch.bool for m in masks))
        self.assertEqual(len(masks), 2)

    def test_subset_of_features_can_be_requested(self):
        inputs, masks, names = unpack_batch(self._batch(), ["wavlm"])
        self.assertEqual(names, ["wavlm"])
        self.assertEqual(len(inputs), 1)
        self.assertEqual(len(masks), 1)

    def test_missing_feature_raises(self):
        with self.assertRaises(KeyError):
            unpack_batch(self._batch(), ["marlin"])

    def test_missing_mask_raises(self):
        batch = self._batch()
        del batch["wavlm_mask"]
        with self.assertRaises(KeyError) as ctx:
            unpack_batch(batch, ["wavlm"])
        self.assertIn("mask", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
