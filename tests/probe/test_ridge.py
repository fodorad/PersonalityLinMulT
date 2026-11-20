"""Tests for :mod:`personalitylinmult.probe.ridge`."""

from __future__ import annotations

import unittest

import torch

from personalitylinmult.probe.ridge import (
    ALPHA_GRID,
    RidgeAccumulator,
    evaluate,
    select_alpha,
)


def _planted(n: int, d_in: int = 6, d_out: int = 3, noise: float = 0.1, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(d_in, d_out, generator=torch.Generator().manual_seed(123))
    x = torch.randn(n, d_in, generator=g) * 3 + 1
    y = x @ w + 2.0 + noise * torch.randn(n, d_out, generator=g)
    return x, y


def _fit(x, y, alpha=1e-3, chunk=None):
    acc = RidgeAccumulator(x.shape[1], y.shape[1])
    step = chunk or x.shape[0]
    for i in range(0, x.shape[0], step):
        acc.update(x[i : i + step], y[i : i + step])
    return acc.solve([alpha])[alpha]


class TestRidge(unittest.TestCase):
    def test_recovers_planted_map(self) -> None:
        x, y = _planted(4000)
        xt, yt = _planted(2000, seed=7)
        m = evaluate(_fit(x, y), [(xt, yt)])
        self.assertEqual(m.r2.shape, (3,))
        self.assertGreater(m.mean_r2, 0.99)
        self.assertGreater(m.mean_pearson_r, 0.99)
        self.assertGreater(m.weighted_r2, 0.99)

    def test_chunked_equals_one_shot(self) -> None:
        x, y = _planted(1000)
        a, b = _fit(x, y), _fit(x, y, chunk=137)
        self.assertTrue(torch.allclose(a.weight, b.weight, atol=1e-9))
        self.assertTrue(torch.allclose(a.y_mean, b.y_mean))

    def test_huge_alpha_gives_zero_r2(self) -> None:
        x, y = _planted(2000)
        xt, yt = _planted(1000, seed=5)
        probe = _fit(x, y, alpha=1e12)
        self.assertAlmostEqual(evaluate(probe, [(xt, yt)]).mean_r2, 0.0, delta=0.05)

    def test_select_alpha_prefers_regularisation_for_noise(self) -> None:
        g = torch.Generator().manual_seed(1)
        x, y = torch.randn(60, 40, generator=g), torch.randn(60, 2, generator=g)
        xv, yv = torch.randn(500, 40, generator=g), torch.randn(500, 2, generator=g)
        acc = RidgeAccumulator(40, 2)
        acc.update(x, y)
        _, alpha = select_alpha(acc, ALPHA_GRID, [(xv, yv)])
        self.assertGreaterEqual(alpha, 100.0)

    def test_select_alpha_prefers_small_alpha_for_signal(self) -> None:
        x, y = _planted(3000)
        xv, yv = _planted(1000, seed=3)
        acc = RidgeAccumulator(6, 3)
        acc.update(x, y)
        _, alpha = select_alpha(acc, ALPHA_GRID, [(xv, yv)])
        self.assertLessEqual(alpha, 10.0)

    def test_constant_output_is_finite(self) -> None:
        x, y = _planted(500)
        y[:, 1] = 4.0
        m = evaluate(_fit(x, y), [(x, y)])
        self.assertEqual(float(m.r2[1]), 0.0)
        self.assertEqual(float(m.pearson_r[1]), 0.0)
        self.assertTrue(torch.isfinite(m.r2).all() and torch.isfinite(m.pearson_r).all())

    def test_constant_input_is_finite(self) -> None:
        x, y = _planted(500)
        x[:, 0] = 7.0
        m = evaluate(_fit(x, y), [(x, y)])
        self.assertTrue(torch.isfinite(m.r2).all())

    def test_weighted_r2_matches_definition(self) -> None:
        x, y = _planted(800, noise=2.0)
        xt, yt = _planted(600, noise=2.0, seed=9)
        probe = _fit(x, y, alpha=10.0)
        m = evaluate(probe, [(xt[:250], yt[:250]), (xt[250:], yt[250:])])
        yd = yt.double()
        res = (yd - probe.predict(xt)) ** 2
        tot = (yd - yd.mean(0)) ** 2
        self.assertAlmostEqual(m.weighted_r2, float(1 - res.sum() / tot.sum()), places=9)
        self.assertTrue(torch.allclose(m.r2, 1 - res.sum(0) / tot.sum(0)))

    def test_bad_shapes_raise(self) -> None:
        acc = RidgeAccumulator(4, 2)
        with self.assertRaises(ValueError):
            acc.update(torch.zeros(5, 4), torch.zeros(6, 2))
        with self.assertRaises(ValueError):
            acc.update(torch.zeros(5, 3), torch.zeros(5, 2))

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            RidgeAccumulator(4, 2).solve([1.0])
        probe = _fit(*_planted(50))
        with self.assertRaises(ValueError):
            evaluate(probe, [])


if __name__ == "__main__":
    unittest.main()
