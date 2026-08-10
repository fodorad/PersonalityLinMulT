"""Tests for the FI losses.

The bell loss was ported out of ``exordium`` so that training does not depend on
the full extraction stack. These tests pin the port against hand-computed values
and, when it is installed, against the original implementation itself.
"""

import math
import unittest

import torch
from torch import nn

from personalitylinmult.train.losses import (
    BELL_AMPLITUDE,
    BELL_SCALE,
    BellL2L1Loss,
    CCCLoss,
    UncertaintyWeightedLoss,
    bell_l2_l1_loss,
    build_loss,
    ccc_loss,
)


class TestBellL2L1Loss(unittest.TestCase):
    def test_hand_computed(self):
        preds = torch.tensor([[0.5]])
        target = torch.tensor([[0.9]])
        diff = 0.4
        expected = BELL_AMPLITUDE * (1 - math.exp(-(diff**2) / BELL_SCALE)) + diff**2 + diff
        self.assertAlmostEqual(float(bell_l2_l1_loss(preds, target)), expected, places=5)

    def test_perfect_prediction_is_zero(self):
        x = torch.rand(8, 5)
        self.assertAlmostEqual(float(bell_l2_l1_loss(x, x)), 0.0, places=6)

    def test_is_non_negative(self):
        for _ in range(20):
            loss = bell_l2_l1_loss(torch.rand(4, 5), torch.rand(4, 5))
            self.assertGreaterEqual(float(loss), 0.0)

    def test_is_symmetric_in_the_sign_of_the_error(self):
        target = torch.tensor([[0.5]])
        over = bell_l2_l1_loss(torch.tensor([[0.7]]), target)
        under = bell_l2_l1_loss(torch.tensor([[0.3]]), target)
        torch.testing.assert_close(over, under)

    def test_increases_with_error(self):
        target = torch.zeros(1, 1)
        losses = [
            float(bell_l2_l1_loss(torch.full((1, 1), e), target)) for e in (0.1, 0.3, 0.6, 0.9)
        ]
        self.assertEqual(losses, sorted(losses))

    def test_returns_a_scalar(self):
        self.assertEqual(bell_l2_l1_loss(torch.rand(4, 5), torch.rand(4, 5)).shape, ())

    def test_is_differentiable(self):
        preds = torch.rand(4, 5, requires_grad=True)
        bell_l2_l1_loss(preds, torch.rand(4, 5)).backward()
        self.assertIsNotNone(preds.grad)
        self.assertFalse(torch.isnan(preds.grad).any())

    def test_matches_the_original_exordium_implementation(self):
        """The port must be numerically identical to the code it replaced."""
        try:
            from exordium.utils.loss import bell_l2_l1_loss as original
        except ImportError:
            self.skipTest("exordium is not installed")

        for _ in range(10):
            preds, target = torch.rand(8, 5), torch.rand(8, 5)
            torch.testing.assert_close(bell_l2_l1_loss(preds, target), original(preds, target))


class TestBellL2L1Module(unittest.TestCase):
    def test_module_matches_the_function(self):
        preds, target = torch.rand(4, 5), torch.rand(4, 5)
        torch.testing.assert_close(BellL2L1Loss()(preds, target), bell_l2_l1_loss(preds, target))


class TestCCCLoss(unittest.TestCase):
    def test_perfect_prediction_is_zero(self):
        x = torch.rand(8, 5)
        self.assertAlmostEqual(float(ccc_loss(x, x)), 0.0, places=5)

    def test_is_non_negative(self):
        for _ in range(20):
            loss = ccc_loss(torch.rand(8, 5), torch.rand(8, 5))
            self.assertGreaterEqual(float(loss), 0.0)

    def test_returns_a_scalar(self):
        self.assertEqual(ccc_loss(torch.rand(8, 5), torch.rand(8, 5)).shape, ())

    def test_is_differentiable(self):
        preds = torch.rand(8, 5, requires_grad=True)
        ccc_loss(preds, torch.rand(8, 5)).backward()
        self.assertIsNotNone(preds.grad)
        self.assertFalse(torch.isnan(preds.grad).any())

    def test_worse_concordance_is_a_larger_loss(self):
        target = torch.linspace(0, 1, 8).unsqueeze(1)
        # A scaled/shifted copy of target still ranks perfectly (Pearson r=1)
        # but disagrees on scale and bias, so CCC -- and this loss -- must
        # penalise it more than the identical (perfect-concordance) prediction.
        good = target.clone()
        worse = target * 0.5 + 0.3
        self.assertLess(float(ccc_loss(good, target)), float(ccc_loss(worse, target)))

    def test_matches_the_metrics_concordance_correlation(self):
        # The loss must agree with the project's evaluation metric it is meant
        # to align training with -- see the module docstring.
        from personalitylinmult.train.metrics import concordance_correlation

        preds, target = torch.rand(8, 5), torch.rand(8, 5)
        expected = torch.mean(1.0 - concordance_correlation(preds, target))
        torch.testing.assert_close(ccc_loss(preds, target), expected)


class TestCCCLossModule(unittest.TestCase):
    def test_module_matches_the_function(self):
        preds, target = torch.rand(8, 5), torch.rand(8, 5)
        torch.testing.assert_close(CCCLoss()(preds, target), ccc_loss(preds, target))


class TestUncertaintyWeightedLoss(unittest.TestCase):
    def test_requires_at_least_two_losses(self):
        with self.assertRaises(ValueError):
            UncertaintyWeightedLoss([CCCLoss()])

    def test_returns_a_scalar(self):
        combined = UncertaintyWeightedLoss([CCCLoss(), nn.L1Loss()])
        loss = combined(torch.rand(8, 5), torch.rand(8, 5))
        self.assertEqual(loss.shape, ())

    def test_equal_initial_weights_match_hand_computed_sum(self):
        # log_sigma starts at 0 for every term, so precision = exp(0) = 1 and
        # each term contributes 0.5 * L_i + 0 -- pin that starting point down.
        preds, target = torch.rand(8, 5), torch.rand(8, 5)
        combined = UncertaintyWeightedLoss([CCCLoss(), nn.L1Loss()])
        expected = 0.5 * ccc_loss(preds, target) + 0.5 * nn.L1Loss()(preds, target)
        torch.testing.assert_close(combined(preds, target), expected)

    def test_is_differentiable_including_log_sigma(self):
        combined = UncertaintyWeightedLoss([CCCLoss(), nn.L1Loss()])
        preds = torch.rand(8, 5, requires_grad=True)
        loss = combined(preds, torch.rand(8, 5))
        loss.backward()
        self.assertIsNotNone(preds.grad)
        self.assertIsNotNone(combined.log_sigma.grad)
        self.assertFalse(torch.isnan(combined.log_sigma.grad).any())

    def test_log_sigma_is_a_trainable_parameter(self):
        combined = UncertaintyWeightedLoss([CCCLoss(), nn.L1Loss()])
        names = [name for name, _ in combined.named_parameters()]
        self.assertIn("log_sigma", names)

    def test_large_log_sigma_drives_precision_toward_zero(self):
        # This is the mechanism the whole module relies on: as log_sigma_i
        # grows, exp(-2*log_sigma_i) -> 0, so that term's weighted loss
        # contribution vanishes and only its "+log_sigma_i" regulariser term
        # remains -- i.e. the model can learn to ignore a term it fits badly.
        preds, target = torch.rand(8, 5), torch.rand(8, 5)
        combined = UncertaintyWeightedLoss([CCCLoss(), nn.L1Loss()])
        with torch.no_grad():
            combined.log_sigma[0] = 20.0
        expected = 0.5 * nn.L1Loss()(preds, target) + combined.log_sigma[1] + combined.log_sigma[0]
        torch.testing.assert_close(combined(preds, target), expected, atol=1e-3, rtol=1e-3)


class TestBuildLoss(unittest.TestCase):
    def test_builds_each_supported_loss(self):
        for name, expected in [
            ("bell_l2_l1", BellL2L1Loss),
            ("l1", nn.L1Loss),
            ("mse", nn.MSELoss),
            ("huber", nn.HuberLoss),
            ("ccc", CCCLoss),
            ("ccc+l1", UncertaintyWeightedLoss),
        ]:
            with self.subTest(name):
                self.assertIsInstance(build_loss(name), expected)

    def test_every_built_loss_runs(self):
        preds, target = torch.rand(4, 5), torch.rand(4, 5)
        for name in ("bell_l2_l1", "l1", "mse", "huber", "ccc", "ccc+l1"):
            with self.subTest(name):
                loss = build_loss(name)(preds, target)
                self.assertEqual(loss.shape, ())
                self.assertGreaterEqual(float(loss), 0.0)

    def test_unknown_loss_raises(self):
        with self.assertRaises(ValueError) as ctx:
            build_loss("cross_entropy")
        self.assertIn("cross_entropy", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
