"""Tests for the FI regression metrics, pinned to hand-computed values."""

import math
import unittest

import torch

from personalitylinmult.train.metrics import (
    PRIMARY_METRIC,
    TraitRegressionMetrics,
    clamp_predictions,
    concordance_correlation,
    one_minus_mae,
    pearson_r,
    r2_score,
    std_ratio,
)


class TestOneMinusMae(unittest.TestCase):
    def test_hand_computed(self):
        preds = torch.tensor([[0.5, 0.2], [0.7, 0.4]])
        target = torch.tensor([[0.4, 0.2], [0.5, 0.9]])
        # trait 0: |0.5-0.4|=0.1, |0.7-0.5|=0.2 -> MAE 0.15 -> 0.85
        # trait 1: |0.2-0.2|=0.0, |0.4-0.9|=0.5 -> MAE 0.25 -> 0.75
        torch.testing.assert_close(one_minus_mae(preds, target), torch.tensor([0.85, 0.75]))

    def test_perfect_prediction_scores_one(self):
        x = torch.rand(16, 5)
        torch.testing.assert_close(one_minus_mae(x, x), torch.ones(5))

    def test_worst_case_scores_zero(self):
        preds = torch.zeros(4, 2)
        target = torch.ones(4, 2)
        torch.testing.assert_close(one_minus_mae(preds, target), torch.zeros(2))

    def test_is_per_trait_not_pooled(self):
        preds = torch.tensor([[0.0, 1.0]])
        target = torch.tensor([[0.0, 0.0]])
        result = one_minus_mae(preds, target)
        self.assertEqual(result.shape, (2,))
        torch.testing.assert_close(result, torch.tensor([1.0, 0.0]))


class TestPearson(unittest.TestCase):
    def test_perfect_positive_correlation(self):
        target = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        preds = 2 * target + 5  # affine: r == 1 despite bias and scale
        torch.testing.assert_close(pearson_r(preds, target), torch.tensor([1.0]))

    def test_perfect_negative_correlation(self):
        target = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        torch.testing.assert_close(pearson_r(-target, target), torch.tensor([-1.0]))

    def test_constant_prediction_is_zero_not_nan(self):
        target = torch.tensor([[1.0], [2.0], [3.0]])
        preds = torch.full((3, 1), 0.5)
        result = pearson_r(preds, target)
        self.assertFalse(torch.isnan(result).any())
        torch.testing.assert_close(result, torch.tensor([0.0]))

    def test_matches_known_value(self):
        preds = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
        target = torch.tensor([[2.0], [1.0], [4.0], [3.0], [5.0]])
        # r = 0.8, confirmed against scipy.stats.pearsonr and numpy.corrcoef.
        torch.testing.assert_close(
            pearson_r(preds, target), torch.tensor([0.8]), atol=1e-4, rtol=1e-4
        )


class TestConcordance(unittest.TestCase):
    def test_identical_series_gives_one(self):
        x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        torch.testing.assert_close(
            concordance_correlation(x, x), torch.tensor([1.0]), atol=1e-5, rtol=1e-5
        )

    def test_biased_prediction_scores_below_pearson(self):
        """CCC punishes bias; Pearson does not. That is why both are reported."""
        target = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        preds = target + 10.0

        self.assertAlmostEqual(float(pearson_r(preds, target)), 1.0, places=4)
        self.assertLess(float(concordance_correlation(preds, target)), 0.1)

    def test_hand_computed(self):
        preds = torch.tensor([[0.0], [1.0], [2.0]])
        target = torch.tensor([[0.0], [1.0], [4.0]])
        # means 1, 5/3; vars 2/3, 26/9; cov = 4/3
        # ccc = 2*(4/3) / (2/3 + 26/9 + (1-5/3)^2)
        expected = 2 * (4 / 3) / (2 / 3 + 26 / 9 + (1 - 5 / 3) ** 2)
        torch.testing.assert_close(
            concordance_correlation(preds, target),
            torch.tensor([expected], dtype=torch.float32),
            atol=1e-5,
            rtol=1e-5,
        )


class TestR2(unittest.TestCase):
    def test_perfect_prediction_is_one(self):
        target = torch.tensor([[1.0], [2.0], [3.0]])
        torch.testing.assert_close(
            r2_score(target, target), torch.tensor([1.0]), atol=1e-5, rtol=1e-5
        )

    def test_mean_prediction_is_zero(self):
        target = torch.tensor([[1.0], [2.0], [3.0]])
        preds = torch.full((3, 1), 2.0)
        torch.testing.assert_close(
            r2_score(preds, target), torch.tensor([0.0]), atol=1e-5, rtol=1e-5
        )

    def test_worse_than_mean_is_negative(self):
        target = torch.tensor([[1.0], [2.0], [3.0]])
        preds = torch.tensor([[10.0], [-5.0], [7.0]])
        self.assertLess(float(r2_score(preds, target)), 0.0)

    def test_constant_target_is_zero_not_nan(self):
        target = torch.full((4, 1), 3.0)
        preds = torch.rand(4, 1)
        result = r2_score(preds, target)
        self.assertFalse(torch.isnan(result).any())
        torch.testing.assert_close(result, torch.tensor([0.0]))

    def test_is_per_trait_not_flattened(self):
        """The old code pooled R2 over the whole matrix; this must not."""
        target = torch.tensor([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
        preds = torch.tensor([[1.0, 250.0], [2.0, 150.0], [3.0, 200.0]])
        result = r2_score(preds, target)
        self.assertEqual(result.shape, (2,))
        self.assertAlmostEqual(float(result[0]), 1.0, places=4)
        self.assertLess(float(result[1]), 0.5)


class TestStdRatio(unittest.TestCase):
    def test_identical_spread_is_one(self):
        target = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        preds = target + 1.0  # same spread, shifted -- ratio is scale-only
        torch.testing.assert_close(
            std_ratio(preds, target), torch.tensor([1.0]), atol=1e-5, rtol=1e-5
        )

    def test_collapsed_prediction_is_near_zero(self):
        target = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        preds = torch.full((4, 1), target.mean())
        torch.testing.assert_close(
            std_ratio(preds, target), torch.tensor([0.0]), atol=1e-5, rtol=1e-5
        )

    def test_overconfident_prediction_exceeds_one(self):
        target = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        preds = (target - target.mean()) * 2 + target.mean()
        self.assertGreater(float(std_ratio(preds, target)), 1.0)

    def test_constant_target_is_zero_not_nan(self):
        target = torch.full((4, 1), 3.0)
        preds = torch.rand(4, 1)
        result = std_ratio(preds, target)
        self.assertFalse(torch.isnan(result).any())
        torch.testing.assert_close(result, torch.tensor([0.0]))

    def test_is_per_trait_not_flattened(self):
        target = torch.tensor([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
        preds = torch.tensor([[1.5, 150.0], [2.0, 200.0], [2.5, 250.0]])
        result = std_ratio(preds, target)
        self.assertEqual(result.shape, (2,))


class TestClamping(unittest.TestCase):
    def test_clamps_into_target_range(self):
        preds = torch.tensor([[-0.5, 0.5, 1.5]])
        torch.testing.assert_close(clamp_predictions(preds), torch.tensor([[0.0, 0.5, 1.0]]))

    def test_clamping_improves_a_wild_prediction(self):
        target = torch.tensor([[1.0]])
        wild = torch.tensor([[5.0]])
        self.assertGreater(
            float(one_minus_mae(clamp_predictions(wild), target)),
            float(one_minus_mae(wild, target)),
        )


class TestTraitRegressionMetrics(unittest.TestCase):
    TRAITS = [
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "emotional_stability",
    ]

    def test_reports_every_trait_and_summary(self):
        metric = TraitRegressionMetrics(self.TRAITS)
        metric.update(torch.rand(8, 5), torch.rand(8, 5))
        result = metric.compute()

        for trait in self.TRAITS:
            self.assertIn(f"one_minus_mae/{trait}", result)
            self.assertIn(f"pearson_r/{trait}", result)
            self.assertIn(f"ccc/{trait}", result)
            self.assertIn(f"r2/{trait}", result)
            self.assertIn(f"std_ratio/{trait}", result)
        for name in (
            "mean_one_minus_mae",
            "mean_pearson_r",
            "mean_ccc",
            "mean_r2",
            "mean_std_ratio",
            "mse",
        ):
            self.assertIn(name, result)

    def test_no_metric_is_called_racc(self):
        """`racc` meant 1-MAE in training and Pearson r in experiments."""
        metric = TraitRegressionMetrics(self.TRAITS)
        metric.update(torch.rand(4, 5), torch.rand(4, 5))
        self.assertNotIn("racc", " ".join(metric.compute().keys()))

    def test_mean_is_the_mean_of_the_five_trait_values(self):
        metric = TraitRegressionMetrics(self.TRAITS)
        metric.update(torch.rand(16, 5), torch.rand(16, 5))
        result = metric.compute()

        per_trait = torch.stack([result[f"ccc/{t}"] for t in self.TRAITS])
        torch.testing.assert_close(result[PRIMARY_METRIC], per_trait.mean())

    def test_accumulates_across_batches(self):
        preds, target = torch.rand(12, 5), torch.rand(12, 5)

        batched = TraitRegressionMetrics(self.TRAITS)
        for i in range(0, 12, 4):
            batched.update(preds[i : i + 4], target[i : i + 4])

        single = TraitRegressionMetrics(self.TRAITS)
        single.update(preds, target)

        torch.testing.assert_close(
            batched.compute()[PRIMARY_METRIC], single.compute()[PRIMARY_METRIC]
        )

    def test_predictions_are_clamped_before_scoring(self):
        metric = TraitRegressionMetrics(["a"])
        metric.update(torch.tensor([[5.0]]), torch.tensor([[1.0]]))
        self.assertAlmostEqual(float(metric.compute()["one_minus_mae/a"]), 1.0, places=5)

    def test_perfect_predictions_score_one(self):
        target = torch.rand(10, 5)
        metric = TraitRegressionMetrics(self.TRAITS)
        metric.update(target, target)
        self.assertAlmostEqual(float(metric.compute()[PRIMARY_METRIC]), 1.0, places=5)

    def test_returns_accumulated_predictions_for_artifact_dumping(self):
        metric = TraitRegressionMetrics(["a"])
        metric.update(torch.tensor([[0.5]]), torch.tensor([[0.25]]))
        metric.update(torch.tensor([[0.75]]), torch.tensor([[1.0]]))
        preds, target = metric.predictions()
        self.assertEqual(preds.shape, (2, 1))
        self.assertEqual(target.shape, (2, 1))

    def test_single_target_interview_task(self):
        metric = TraitRegressionMetrics(["interview"])
        metric.update(torch.rand(6, 1), torch.rand(6, 1))
        result = metric.compute()
        self.assertIn("one_minus_mae/interview", result)
        torch.testing.assert_close(result[PRIMARY_METRIC], result["ccc/interview"])

    def test_shape_mismatch_raises(self):
        metric = TraitRegressionMetrics(self.TRAITS)
        with self.assertRaises(ValueError):
            metric.update(torch.rand(4, 5), torch.rand(4, 3))

    def test_wrong_trait_count_raises(self):
        metric = TraitRegressionMetrics(self.TRAITS)
        with self.assertRaises(ValueError) as ctx:
            metric.update(torch.rand(4, 3), torch.rand(4, 3))
        self.assertIn("Expected 5", str(ctx.exception))

    def test_reset_clears_state(self):
        metric = TraitRegressionMetrics(["a"])
        metric.update(torch.tensor([[0.0]]), torch.tensor([[1.0]]))
        metric.reset()
        metric.update(torch.tensor([[1.0]]), torch.tensor([[1.0]]))
        self.assertAlmostEqual(float(metric.compute()["one_minus_mae/a"]), 1.0, places=5)

    def test_predictions_on_empty_state_is_empty_not_a_crash(self):
        """PlotCallback checks for "no predictions"; it must be able to."""
        metric = TraitRegressionMetrics(self.TRAITS)
        preds, target = metric.predictions()
        self.assertEqual(preds.shape, (0, 5))
        self.assertEqual(target.shape, (0, 5))
        self.assertEqual(preds.numel(), 0)

    def test_predictions_after_reset_is_empty(self):
        metric = TraitRegressionMetrics(["a"])
        metric.update(torch.rand(3, 1), torch.rand(3, 1))
        metric.reset()
        preds, _ = metric.predictions()
        self.assertEqual(preds.shape[0], 0)

    def test_compute_on_empty_state_raises_rather_than_logging_nans(self):
        metric = TraitRegressionMetrics(self.TRAITS)
        with self.assertRaises(RuntimeError) as ctx:
            metric.compute()
        self.assertIn("No predictions", str(ctx.exception))

    def test_no_metric_is_nan_on_a_realistic_epoch(self):
        metric = TraitRegressionMetrics(self.TRAITS)
        metric.update(torch.rand(64, 5), torch.rand(64, 5))
        for name, value in metric.compute().items():
            with self.subTest(name):
                self.assertFalse(math.isnan(float(value)))


if __name__ == "__main__":
    unittest.main()
