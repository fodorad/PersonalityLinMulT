"""Tests for optimizer and LR-scheduler construction."""

import unittest

import torch
from torch.optim import Adam, AdamW, RAdam
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau

from personalitylinmult.train.config import OptimConfig
from personalitylinmult.train.optim import build_optimizer, build_scheduler


def parameters():
    return list(torch.nn.Linear(4, 2).parameters())


class TestBuildOptimizer(unittest.TestCase):
    def test_builds_each_supported_optimizer(self):
        for name, expected in [("adam", Adam), ("adamw", AdamW), ("radam", RAdam)]:
            with self.subTest(name):
                optimizer = build_optimizer(OptimConfig(name=name), parameters())
                self.assertIsInstance(optimizer, expected)

    def test_applies_lr_and_weight_decay(self):
        optimizer = build_optimizer(
            OptimConfig(name="adamw", lr=0.003, weight_decay=0.05), parameters()
        )
        group = optimizer.param_groups[0]
        self.assertAlmostEqual(group["lr"], 0.003)
        self.assertAlmostEqual(group["weight_decay"], 0.05)

    def test_unknown_optimizer_raises(self):
        config = OptimConfig.__new__(OptimConfig)
        object.__setattr__(config, "name", "lbfgs")
        object.__setattr__(config, "lr", 0.001)
        object.__setattr__(config, "weight_decay", 0.0)
        with self.assertRaises(ValueError):
            build_optimizer(config, parameters())


class TestBuildScheduler(unittest.TestCase):
    def _optimizer(self):
        return build_optimizer(OptimConfig(), parameters())

    def test_none_returns_nothing(self):
        self.assertIsNone(build_scheduler(OptimConfig(scheduler="none"), self._optimizer(), 10, 5))

    def test_onecycle_steps_per_batch(self):
        config = build_scheduler(OptimConfig(scheduler="onecycle"), self._optimizer(), 10, 5)
        self.assertIsInstance(config["scheduler"], OneCycleLR)
        self.assertEqual(config["interval"], "step")

    def test_plateau_watches_the_primary_metric_per_epoch(self):
        config = build_scheduler(OptimConfig(scheduler="plateau"), self._optimizer(), 10, 5)
        self.assertIsInstance(config["scheduler"], ReduceLROnPlateau)
        self.assertEqual(config["interval"], "epoch")
        self.assertEqual(config["monitor"], "valid/mean_ccc")

    def test_cosine_with_warmup_chains_schedules(self):
        config = build_scheduler(
            OptimConfig(scheduler="cosine", warmup_ratio=0.1), self._optimizer(), 10, 5
        )
        self.assertEqual(config["interval"], "step")

    def test_cosine_without_warmup_is_plain(self):
        config = build_scheduler(
            OptimConfig(scheduler="cosine", warmup_ratio=0.0), self._optimizer(), 10, 5
        )
        self.assertIsNotNone(config["scheduler"])

    def test_every_scheduler_actually_steps(self):
        """A schedule that reaches lr=0 before the end stops training silently."""
        for name in ("onecycle", "cosine"):
            with self.subTest(name):
                optimizer = self._optimizer()
                config = build_scheduler(
                    OptimConfig(scheduler=name), optimizer, steps_per_epoch=4, max_epochs=2
                )
                scheduler = config["scheduler"]
                for _ in range(8):
                    optimizer.step()
                    scheduler.step()
                self.assertGreater(optimizer.param_groups[0]["lr"], 0.0)

    def test_cosine_never_bottoms_out_at_zero_mid_run(self):
        optimizer = self._optimizer()
        config = build_scheduler(
            OptimConfig(scheduler="cosine", lr=0.01, warmup_ratio=0.2),
            optimizer,
            steps_per_epoch=10,
            max_epochs=5,
        )
        seen = []
        for _ in range(50):
            optimizer.step()
            config["scheduler"].step()
            seen.append(optimizer.param_groups[0]["lr"])

        self.assertTrue(all(lr > 0.0 for lr in seen), "lr hit zero during the run")

    def test_warmup_ramps_up_to_the_peak_lr(self):
        settings = OptimConfig(scheduler="cosine", lr=0.01, warmup_ratio=0.5)
        optimizer = build_optimizer(settings, parameters())
        config = build_scheduler(settings, optimizer, steps_per_epoch=10, max_epochs=2)
        start = optimizer.param_groups[0]["lr"]
        self.assertLess(start, 0.01)

        peak = start
        for _ in range(10):
            optimizer.step()
            config["scheduler"].step()
            peak = max(peak, optimizer.param_groups[0]["lr"])

        self.assertGreater(peak, start)
        self.assertAlmostEqual(peak, 0.01, places=4)

    def test_onecycle_survives_a_very_short_run(self):
        """A tiny dataset gives OneCycleLR a zero-width phase -> ZeroDivisionError."""
        for total_steps, (steps_per_epoch, max_epochs) in {
            1: (1, 1),
            2: (1, 2),
            3: (1, 3),
            4: (2, 2),
        }.items():
            with self.subTest(total_steps=total_steps):
                optimizer = self._optimizer()
                config = build_scheduler(
                    OptimConfig(scheduler="onecycle"),
                    optimizer,
                    steps_per_epoch=steps_per_epoch,
                    max_epochs=max_epochs,
                )
                if config is None:
                    continue  # too short to schedule; constant lr is correct
                for _ in range(total_steps):
                    optimizer.step()
                    config["scheduler"].step()

    def test_onecycle_falls_back_below_the_minimum_step_count(self):
        self.assertIsNone(
            build_scheduler(OptimConfig(scheduler="onecycle"), self._optimizer(), 1, 1)
        )

    def test_onecycle_peaks_near_the_configured_warmup_point(self):
        settings = OptimConfig(scheduler="onecycle", warmup_ratio=0.3, lr=0.01)
        optimizer = build_optimizer(settings, parameters())
        total = 200
        config = build_scheduler(settings, optimizer, steps_per_epoch=total, max_epochs=1)

        lrs = []
        for _ in range(total):
            optimizer.step()
            config["scheduler"].step()
            lrs.append(optimizer.param_groups[0]["lr"])

        peak_step = lrs.index(max(lrs))
        # The peak should land near 30% of the run, and the LR should rise to it
        # and fall away afterwards.
        self.assertAlmostEqual(peak_step / total, 0.3, delta=0.05)
        self.assertAlmostEqual(max(lrs), 0.01, places=4)
        self.assertLess(lrs[-1], max(lrs))

    def test_unknown_scheduler_raises(self):
        config = OptimConfig.__new__(OptimConfig)
        for name, value in [
            ("name", "adamw"),
            ("lr", 0.001),
            ("weight_decay", 0.0),
            ("scheduler", "exponential"),
            ("warmup_ratio", 0.0),
            ("gradient_clip_val", 1.0),
        ]:
            object.__setattr__(config, name, value)
        with self.assertRaises(ValueError):
            build_scheduler(config, self._optimizer(), 10, 5)


if __name__ == "__main__":
    unittest.main()
