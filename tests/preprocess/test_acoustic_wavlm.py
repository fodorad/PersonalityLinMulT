"""Tests for WavLM layer selection.

The extraction itself needs a GPU and the real model, so it is not exercised
here; the layer/dtype selection that decides what reaches the disk is pure and
is what these tests cover.
"""

import unittest

import numpy as np

from personalitylinmult.preprocess.fi.acoustic_wavlm import (
    DEFAULT_DTYPE,
    DEFAULT_LAYERS,
    select_layers,
)


def fake_layers(n: int = 12, time_dim: int = 20, dim: int = 8) -> list[np.ndarray]:
    """Build a stand-in for the extractor's per-layer output.

    Each layer is filled with its own index, so a selection bug shows up as the
    wrong values rather than merely the wrong shape.

    Args:
        n (int): Number of layers.
        time_dim (int): Frames per layer.
        dim (int): Feature width.

    Returns:
        list[np.ndarray]: ``n`` arrays of ``(time_dim, dim)`` float32.
    """
    return [np.full((time_dim, dim), float(i), dtype=np.float32) for i in range(n)]


class TestSelectLayers(unittest.TestCase):
    def test_keeps_only_the_requested_layer(self):
        selected = select_layers(fake_layers(), (9,), "float16")
        self.assertEqual(len(selected), 1)
        self.assertTrue(np.all(selected[0] == 9.0))

    def test_keeps_requested_order_not_sorted_order(self):
        selected = select_layers(fake_layers(), (9, 2), "float16")
        self.assertTrue(np.all(selected[0] == 9.0))
        self.assertTrue(np.all(selected[1] == 2.0))

    def test_casts_to_the_requested_dtype(self):
        self.assertEqual(select_layers(fake_layers(), (9,), "float16")[0].dtype, np.float16)
        self.assertEqual(select_layers(fake_layers(), (9,), "float32")[0].dtype, np.float32)

    def test_fp16_preserves_wavlm_activation_magnitudes(self):
        """Real activations peak near ~31, far inside fp16's 65504 ceiling."""
        rng = np.random.default_rng(0)
        layers = [rng.standard_normal((20, 8)).astype(np.float32) * 10 for _ in range(12)]
        selected = select_layers(layers, (9,), "float16")
        np.testing.assert_allclose(selected[0], layers[9], rtol=1e-2, atol=1e-2)

    def test_out_of_range_layer_raises(self):
        with self.assertRaises(IndexError):
            select_layers(fake_layers(), (12,), "float16")

    def test_negative_layer_raises(self):
        """Negative indices would silently wrap to a different layer."""
        with self.assertRaises(IndexError):
            select_layers(fake_layers(), (-1,), "float16")

    def test_defaults_match_the_h5_feature_config(self):
        """config/FI/features/wavlm.yaml selects layer 9 and stores fp16."""
        self.assertEqual(DEFAULT_LAYERS, (9,))
        self.assertEqual(DEFAULT_DTYPE, "float16")


if __name__ == "__main__":
    unittest.main()
