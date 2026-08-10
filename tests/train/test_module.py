"""Tests for the Lightning module, exercised end-to-end on a real HDF5 file."""

import tempfile
import unittest
from pathlib import Path

import lightning as L
import torch
from data.test_builder import dinov2_spec, wavlm_spec, xlm_spec

from data import fixtures
from personalitylinmult.data.builder import build
from personalitylinmult.data.datamodule import FiDataModule
from personalitylinmult.data.schema import BIG_FIVE_TRAITS, DatasetSpec
from personalitylinmult.train.config import DataConfig, ModelConfig, TrainConfig
from personalitylinmult.train.module import HEAD_NAME, FiLightningModule, build_model

VIDEO_IDS = [f"clip{i}" for i in range(12)]
SUBSETS = ["train"] * 6 + ["valid"] * 3 + ["test"] * 3
FEATURES = ["wavlm", "dinov2", "xlm_roberta_gt"]
DIMS = [8, 8, 8]


class ModuleTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)

        fixtures.write_label_csv(cls.root / "label.csv", VIDEO_IDS, SUBSETS)
        fixtures.write_wavlm(cls.root / "wavlm_baseplus", VIDEO_IDS, dim=8)
        fixtures.write_dinov2(cls.root / "dinov2_base", VIDEO_IDS, dim=8)
        fixtures.write_text_h5(cls.root / "xlm_roberta_gt.h5", VIDEO_IDS, dim=8)

        build(
            DatasetSpec(
                features=[wavlm_spec(), dinov2_spec(), xlm_spec()],
                h5_path="fi.h5",
                label_csv="label.csv",
            ),
            cls.root,
            "cfg",
        )
        cls.h5_path = cls.root / "fi.h5"

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def datamodule(self, targets):
        return FiDataModule(
            DataConfig(
                h5_path=str(self.h5_path),
                features=FEATURES,
                batch_size=2,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
                window_seconds=1.0,
            ),
            target_names=targets,
        )

    def module(self, train_config):
        return FiLightningModule(
            model_config=ModelConfig(
                d_model=8, num_heads=2, cmt_num_layers=1, branch_sat_num_layers=1
            ),
            train_config=train_config,
            feature_names=FEATURES,
            input_feature_dim=DIMS,
            steps_per_epoch=3,
        )


class TestBuildLinMulT(unittest.TestCase):
    def test_forward_returns_a_dict_keyed_by_head_name(self):
        """linmult 2.x returns a dict; the old code's [0] index was a KeyError."""
        model = build_model(
            ModelConfig(d_model=8, num_heads=2, cmt_num_layers=1, branch_sat_num_layers=1),
            input_feature_dim=[4, 6],
            output_dim=5,
        )
        inputs = [torch.randn(2, 10, 4), torch.randn(2, 7, 6)]
        masks = [torch.ones(2, 10, dtype=torch.bool), torch.ones(2, 7, dtype=torch.bool)]

        out = model(inputs, masks, names=["a", "b"])

        self.assertIsInstance(out, dict)
        self.assertIn(HEAD_NAME, out)
        self.assertEqual(out[HEAD_NAME].shape, (2, 5))
        with self.assertRaises(KeyError):
            _ = out[0]

    def test_output_dim_follows_the_task(self):
        for output_dim in (5, 1):
            with self.subTest(output_dim):
                model = build_model(
                    ModelConfig(d_model=8, num_heads=2, cmt_num_layers=1, branch_sat_num_layers=1),
                    input_feature_dim=[4, 4],
                    output_dim=output_dim,
                )
                out = model(
                    [torch.randn(2, 5, 4), torch.randn(2, 5, 4)],
                    [torch.ones(2, 5, dtype=torch.bool)] * 2,
                    names=["a", "b"],
                )
                self.assertEqual(out[HEAD_NAME].shape, (2, output_dim))

    def test_handles_differing_time_dims_per_modality(self):
        """Real features are 747/375/1 frames; no forced alignment is needed."""
        model = build_model(
            ModelConfig(d_model=8, num_heads=2, cmt_num_layers=1, branch_sat_num_layers=1),
            input_feature_dim=[4, 4, 4],
            output_dim=5,
        )
        out = model(
            [torch.randn(2, 40, 4), torch.randn(2, 20, 4), torch.randn(2, 1, 4)],
            [
                torch.ones(2, 40, dtype=torch.bool),
                torch.ones(2, 20, dtype=torch.bool),
                torch.ones(2, 1, dtype=torch.bool),
            ],
            names=["a", "b", "c"],
        )
        self.assertEqual(out[HEAD_NAME].shape, (2, 5))


class TestForward(ModuleTestCase):
    def test_forward_shape_for_each_task(self):
        for task, targets, width in [
            ("bigfive", list(BIG_FIVE_TRAITS), 5),
            ("interview", ["interview"], 1),
        ]:
            with self.subTest(task):
                module = self.module(TrainConfig(task=task, targets=targets))
                dm = self.datamodule(targets)
                dm.setup("fit")
                batch = next(iter(dm.train_dataloader()))

                from personalitylinmult.data.collate import unpack_batch

                inputs, masks, _ = unpack_batch(batch, FEATURES)
                preds = module(inputs, masks)

                self.assertEqual(preds.shape, (2, width))

    def test_every_feature_is_passed_with_its_mask(self):
        dm = self.datamodule(list(BIG_FIVE_TRAITS))
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))

        from personalitylinmult.data.collate import unpack_batch

        inputs, masks, names = unpack_batch(batch, FEATURES)

        self.assertEqual(len(inputs), len(FEATURES))
        self.assertEqual(len(masks), len(FEATURES))
        self.assertEqual(names, FEATURES)
        for tensor, mask in zip(inputs, masks, strict=True):
            self.assertEqual(mask.dtype, torch.bool)
            self.assertEqual(tensor.shape[0], mask.shape[0])
            self.assertEqual(tensor.shape[-2], mask.shape[-1])


class TestFastDevRun(ModuleTestCase):
    """One batch through every stage: catches wiring bugs a unit test cannot."""

    def _run(self, task, targets):
        module = self.module(TrainConfig(task=task, targets=targets, max_epochs=1))
        dm = self.datamodule(targets)
        trainer = L.Trainer(
            fast_dev_run=True,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        trainer.fit(module, datamodule=dm)
        trainer.test(module, datamodule=dm, verbose=False)
        return trainer

    def test_bigfive_trains_and_tests(self):
        trainer = self._run("bigfive", list(BIG_FIVE_TRAITS))
        self.assertEqual(trainer.state.finished, True)

    def test_interview_trains_and_tests(self):
        trainer = self._run("interview", ["interview"])
        self.assertEqual(trainer.state.finished, True)

    def test_loss_is_finite_and_backpropagates(self):
        module = self.module(TrainConfig(task="bigfive"))
        dm = self.datamodule(list(BIG_FIVE_TRAITS))
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))

        loss = module.training_step(batch, 0)

        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        grads = [p.grad is not None for p in module.model.parameters() if p.requires_grad]
        self.assertTrue(all(grads))

    def test_combined_loss_trains_and_its_weights_get_gradients(self):
        """``ccc+l1``'s log_sigma must be a real, optimised parameter.

        It is registered only by virtue of ``self.loss_fn`` being an
        ``nn.Module`` attribute of the LightningModule -- if that wiring ever
        broke, this is what would catch it (a passing fast-dev-run alone
        would not, since a frozen log_sigma still produces a finite loss).
        """
        module = self.module(TrainConfig(task="bigfive", loss="ccc+l1"))
        dm = self.datamodule(list(BIG_FIVE_TRAITS))
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))

        self.assertIn("loss_fn.log_sigma", dict(module.named_parameters()))

        loss = module.training_step(batch, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(module.loss_fn.log_sigma.grad)
        self.assertFalse(torch.isnan(module.loss_fn.log_sigma.grad).any())

        optimizer = module.configure_optimizers()["optimizer"]
        optimizer_params = {id(p) for group in optimizer.param_groups for p in group["params"]}
        self.assertIn(id(module.loss_fn.log_sigma), optimizer_params)

    def test_combined_loss_log_sigma_is_logged_per_term(self):
        """log_sigma must be visible in MLflow, not just inside the checkpoint --
        otherwise there is no way to confirm after a real run whether a term's
        learned weight actually receded once it stopped being needed."""
        module = self.module(TrainConfig(task="bigfive", loss="ccc+l1", max_epochs=1))
        dm = self.datamodule(list(BIG_FIVE_TRAITS))
        trainer = L.Trainer(
            fast_dev_run=True,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        trainer.fit(module, datamodule=dm)

        logged = trainer.callback_metrics
        self.assertIn("train/loss_log_sigma_0", logged)
        self.assertIn("train/loss_log_sigma_1", logged)

    def test_plain_loss_logs_no_log_sigma(self):
        """A loss without learnable weights (e.g. bell_l2_l1) has no log_sigma
        to log -- confirms the getattr guard skips cleanly rather than erroring."""
        module = self.module(TrainConfig(task="bigfive", loss="bell_l2_l1", max_epochs=1))
        dm = self.datamodule(list(BIG_FIVE_TRAITS))
        trainer = L.Trainer(
            fast_dev_run=True,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        trainer.fit(module, datamodule=dm)

        logged = trainer.callback_metrics
        self.assertNotIn("train/loss_log_sigma_0", logged)


class TestCheckpointing(ModuleTestCase):
    def test_load_from_checkpoint_needs_no_model_argument(self):
        """The module builds its own model, so a checkpoint is self-contained."""
        targets = list(BIG_FIVE_TRAITS)
        module = self.module(TrainConfig(task="bigfive", targets=targets, max_epochs=1))
        dm = self.datamodule(targets)

        with tempfile.TemporaryDirectory() as tmp:
            trainer = L.Trainer(
                max_epochs=1,
                accelerator="cpu",
                logger=False,
                enable_progress_bar=False,
                default_root_dir=tmp,
                limit_train_batches=1,
                limit_val_batches=1,
            )
            trainer.fit(module, datamodule=dm)
            path = Path(tmp) / "ckpt.ckpt"
            trainer.save_checkpoint(path)

            restored = FiLightningModule.load_from_checkpoint(path, map_location="cpu")

        self.assertEqual(restored.feature_names, FEATURES)
        self.assertEqual(restored.train_config.task, "bigfive")

    def test_restored_model_predicts_identically(self):
        targets = list(BIG_FIVE_TRAITS)
        module = self.module(TrainConfig(task="bigfive", targets=targets, max_epochs=1))
        dm = self.datamodule(targets)
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))

        from personalitylinmult.data.collate import unpack_batch

        inputs, masks, _ = unpack_batch(batch, FEATURES)

        module.eval()
        with torch.no_grad():
            before = module(inputs, masks)

        with tempfile.TemporaryDirectory() as tmp:
            trainer = L.Trainer(
                max_epochs=1,
                accelerator="cpu",
                logger=False,
                enable_progress_bar=False,
                default_root_dir=tmp,
                limit_train_batches=0,
                limit_val_batches=0,
            )
            trainer.strategy.connect(module)
            path = Path(tmp) / "ckpt.ckpt"
            trainer.save_checkpoint(path)
            restored = FiLightningModule.load_from_checkpoint(path, map_location="cpu")

        restored.eval()
        with torch.no_grad():
            after = restored(inputs, masks)

        torch.testing.assert_close(before, after)


class TestOptimizerConfiguration(ModuleTestCase):
    def test_configure_optimizers_returns_optimizer_and_schedule(self):
        module = self.module(TrainConfig(task="bigfive"))
        config = module.configure_optimizers()
        self.assertIn("optimizer", config)
        self.assertIn("lr_scheduler", config)

    def test_scheduler_none_returns_only_the_optimizer(self):
        from personalitylinmult.train.config import OptimConfig

        module = self.module(TrainConfig(task="bigfive", optimizer=OptimConfig(scheduler="none")))
        config = module.configure_optimizers()
        self.assertIn("optimizer", config)
        self.assertNotIn("lr_scheduler", config)


if __name__ == "__main__":
    unittest.main()
