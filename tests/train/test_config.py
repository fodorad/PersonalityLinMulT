"""Tests for the typed training configuration."""

import tempfile
import unittest
from pathlib import Path

import yaml

from personalitylinmult.train.config import (
    ConfigError,
    DataConfig,
    EarlyStoppingConfig,
    ExperimentConfig,
    MLflowConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
    apply_override,
    load_yaml,
    parse_override,
)


class TestDataConfig(unittest.TestCase):
    def test_defaults(self):
        config = DataConfig(features=["wavlm"])
        self.assertEqual(config.window_seconds, 15.0)
        self.assertEqual(config.crop_mode, "head")
        self.assertTrue(config.persistent_workers)

    def test_requires_features(self):
        with self.assertRaises(ConfigError):
            DataConfig(features=[])

    def test_rejects_duplicate_features(self):
        with self.assertRaises(ConfigError) as ctx:
            DataConfig(features=["wavlm", "wavlm"])
        self.assertIn("duplicates", str(ctx.exception))

    def test_rejects_unknown_crop_mode(self):
        with self.assertRaises(ConfigError):
            DataConfig(features=["a"], crop_mode="sideways")

    def test_rejects_bad_batch_size(self):
        with self.assertRaises(ConfigError):
            DataConfig(features=["a"], batch_size=0)

    def test_rejects_negative_workers(self):
        with self.assertRaises(ConfigError):
            DataConfig(features=["a"], num_workers=-1)

    def test_rejects_non_positive_window(self):
        with self.assertRaises(ConfigError):
            DataConfig(features=["a"], window_seconds=0.0)

    def test_null_window_is_allowed(self):
        self.assertIsNone(DataConfig(features=["a"], window_seconds=None).window_seconds)

    def test_rejects_unknown_keys(self):
        with self.assertRaises(ConfigError) as ctx:
            DataConfig.from_dict({"features": ["a"], "batch_sise": 4})
        self.assertIn("batch_sise", str(ctx.exception))


class TestTrainConfigTasks(unittest.TestCase):
    def test_bigfive_defaults_to_the_five_traits_with_emotional_stability(self):
        config = TrainConfig(task="bigfive")
        self.assertEqual(len(config.targets), 5)
        self.assertIn("emotional_stability", config.targets)
        self.assertNotIn("neuroticism", config.targets)
        self.assertEqual(config.output_dim, 5)

    def test_bigfive_rejects_neuroticism(self):
        """Big five trains on emotional_stability so traits point the same way."""
        with self.assertRaises(ConfigError) as ctx:
            TrainConfig(
                task="bigfive",
                targets=[
                    "openness",
                    "conscientiousness",
                    "extraversion",
                    "agreeableness",
                    "neuroticism",
                ],
            )
        message = str(ctx.exception)
        self.assertIn("emotional_stability", message)
        self.assertIn("neuroticism", message)

    def test_interview_defaults_to_the_interview_variable(self):
        config = TrainConfig(task="interview")
        self.assertEqual(config.targets, ["interview"])
        self.assertEqual(config.output_dim, 1)

    def test_interview_rejects_personality_targets(self):
        """Interview is a separate task, not a sixth trait."""
        with self.assertRaises(ConfigError) as ctx:
            TrainConfig(task="interview", targets=["interview", "openness"])
        self.assertIn("separate task", str(ctx.exception))

    def test_output_dim_is_derived_never_hardcoded(self):
        self.assertEqual(TrainConfig(task="bigfive").output_dim, 5)
        self.assertEqual(TrainConfig(task="interview").output_dim, 1)

    def test_unknown_task_raises(self):
        with self.assertRaises(ConfigError):
            TrainConfig(task="sentiment")

    def test_unknown_target_raises(self):
        with self.assertRaises(ConfigError) as ctx:
            TrainConfig(task="bigfive", targets=["charisma"])
        self.assertIn("charisma", str(ctx.exception))

    def test_unknown_loss_raises(self):
        with self.assertRaises(ConfigError):
            TrainConfig(loss="triplet")

    def test_accumulate_grad_batches_defaults_to_one(self):
        self.assertEqual(TrainConfig(task="bigfive").accumulate_grad_batches, 1)

    def test_accumulate_grad_batches_below_one_raises(self):
        with self.assertRaises(ConfigError) as ctx:
            TrainConfig(task="bigfive", accumulate_grad_batches=0)
        self.assertIn("accumulate_grad_batches", str(ctx.exception))

    def test_nested_sections_are_parsed(self):
        config = TrainConfig.from_dict(
            {
                "task": "bigfive",
                "optimizer": {"name": "adam", "lr": 0.005},
                "mlflow": {"experiment_name": "FI-Test"},
                "early_stopping": {"patience": 3},
            }
        )
        self.assertIsInstance(config.optimizer, OptimConfig)
        self.assertEqual(config.optimizer.lr, 0.005)
        self.assertEqual(config.mlflow.experiment_name, "FI-Test")
        self.assertEqual(config.early_stopping.patience, 3)


class TestOptimConfig(unittest.TestCase):
    def test_rejects_unknown_optimizer(self):
        with self.assertRaises(ConfigError):
            OptimConfig(name="lbfgs")

    def test_rejects_unknown_scheduler(self):
        with self.assertRaises(ConfigError):
            OptimConfig(scheduler="exponential")

    def test_rejects_non_positive_lr(self):
        with self.assertRaises(ConfigError):
            OptimConfig(lr=0.0)

    def test_rejects_out_of_range_warmup(self):
        for ratio in (-0.1, 1.0, 1.5):
            with self.subTest(ratio), self.assertRaises(ConfigError):
                OptimConfig(warmup_ratio=ratio)

    def test_scheduler_none_is_allowed(self):
        self.assertEqual(OptimConfig(scheduler="none").scheduler, "none")


class TestModelConfig(unittest.TestCase):
    def test_defaults_match_the_published_architecture(self):
        config = ModelConfig()
        self.assertEqual(config.d_model, 32)
        self.assertEqual(config.attention_type, "linear")
        self.assertEqual(config.time_dim_reducer, "gap")

    def test_tcn_defaults_off_despite_linmult_defaulting_it_on(self):
        """linmult 2.x flips add_module_tcn to True; 1.5.2 had no TCN at all."""
        self.assertFalse(ModelConfig().add_module_tcn)

    def test_mult_baseline_differs_only_by_attention(self):
        linmult = ModelConfig(attention_type="linear")
        mult = ModelConfig(name="MulT", attention_type="softmax")
        self.assertEqual(linmult.d_model, mult.d_model)
        self.assertEqual(linmult.cmt_num_layers, mult.cmt_num_layers)
        self.assertNotEqual(linmult.attention_type, mult.attention_type)

    def test_rejects_d_model_indivisible_by_heads(self):
        with self.assertRaises(ConfigError) as ctx:
            ModelConfig(d_model=30, num_heads=8)
        self.assertIn("divisible", str(ctx.exception))

    def test_legacy_linmult_1x_keys_are_rejected_with_guidance(self):
        cases = {
            "n_heads": 8,
            "n_layers": 5,
            "ffn_fusion": True,
            "output_dim": [5],
            "dropout_qkv": 0.0,
            "dropout_relu": 0.2,
            "dropout_residual": 0.2,
            "version": "1.5.2",
        }
        for key, value in cases.items():
            with self.subTest(key), self.assertRaises(ConfigError) as ctx:
                ModelConfig.from_dict({key: value})
            self.assertIn(key, str(ctx.exception))
            self.assertIn("linmult 1.x", str(ctx.exception))

    def test_input_feature_dim_is_not_hand_written(self):
        """It is derived from the h5 so the model cannot disagree with its data."""
        with self.assertRaises(ConfigError):
            ModelConfig.from_dict({"input_feature_dim": [768, 768]})

    def test_rejects_unknown_keys(self):
        with self.assertRaises(ConfigError):
            ModelConfig.from_dict({"d_modell": 32})


class TestEarlyStoppingConfig(unittest.TestCase):
    def test_defaults_track_the_primary_metric(self):
        config = EarlyStoppingConfig()
        self.assertEqual(config.monitor, "valid/mean_ccc")
        self.assertEqual(config.mode, "max")

    def test_rejects_bad_mode(self):
        with self.assertRaises(ConfigError):
            EarlyStoppingConfig(mode="maximize")


class TestMLflowConfig(unittest.TestCase):
    def test_tracking_uri_is_relative_by_default(self):
        """An absolute local path in a committed config would leak the machine."""
        uri = MLflowConfig().tracking_uri
        self.assertFalse(Path(uri.split("///")[-1]).is_absolute())

    def test_defaults_to_a_database_backend(self):
        """MLflow 3.x retired the file store and raises when given one."""
        self.assertIn("://", MLflowConfig().tracking_uri)

    def test_filesystem_store_is_rejected_with_guidance(self):
        with self.assertRaises(ConfigError) as ctx:
            MLflowConfig(tracking_uri="./mlruns")
        self.assertIn("sqlite", str(ctx.exception))

    def test_database_uris_are_accepted(self):
        for uri in ("sqlite:///mlflow.db", "postgresql://host/db", "http://localhost:5000"):
            with self.subTest(uri):
                self.assertEqual(MLflowConfig(tracking_uri=uri).tracking_uri, uri)


class TestOverrides(unittest.TestCase):
    def test_parse_typed_values(self):
        self.assertEqual(parse_override("data.window_seconds=10"), ("data.window_seconds", 10))
        self.assertEqual(parse_override("data.batch_size=32"), ("data.batch_size", 32))
        self.assertEqual(parse_override("train.optimizer.lr=1e-4"), ("train.optimizer.lr", 1e-4))
        self.assertEqual(parse_override("data.pin_memory=false"), ("data.pin_memory", False))
        self.assertEqual(parse_override("train.task=interview"), ("train.task", "interview"))

    def test_scientific_notation_parses_as_a_float(self):
        """YAML 1.1 reads '1e-4' as a string; an lr must never arrive as one."""
        for text, expected in [
            ("train.optimizer.lr=1e-4", 1e-4),
            ("train.optimizer.lr=1.0e-4", 1e-4),
            ("train.optimizer.lr=5E-5", 5e-5),
        ]:
            with self.subTest(text):
                _, value = parse_override(text)
                self.assertIsInstance(value, float)
                self.assertAlmostEqual(value, expected)

    def test_genuine_strings_are_left_alone(self):
        for text, expected in [
            ("train.task=interview", "interview"),
            ("model.attention_type=softmax", "softmax"),
            ("data.crop_mode=head", "head"),
        ]:
            with self.subTest(text):
                _, value = parse_override(text)
                self.assertEqual(value, expected)

    def test_parse_requires_equals(self):
        with self.assertRaises(ConfigError):
            parse_override("data.batch_size")

    def test_apply_nested_override(self):
        raw = {"train": {"optimizer": {"lr": 0.001}}}
        apply_override(raw, "train.optimizer.lr", 0.1)
        self.assertEqual(raw["train"]["optimizer"]["lr"], 0.1)

    def test_apply_creates_missing_levels(self):
        raw = {}
        apply_override(raw, "a.b.c", 1)
        self.assertEqual(raw, {"a": {"b": {"c": 1}}})


class TestExperimentConfig(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

        self.data_path = self.root / "data.yaml"
        self.data_path.write_text(
            yaml.safe_dump(
                {
                    "h5_path": "data/processed/FI/fi.h5",
                    "label_csv": "data/processed/FI/label.csv",
                    "features": ["wavlm", "dinov2"],
                    "batch_size": 8,
                }
            )
        )
        self.model_path = self.root / "model.yaml"
        self.model_path.write_text(yaml.safe_dump({"d_model": 32, "attention_type": "linear"}))
        self.train_path = self.root / "train.yaml"
        self.train_path.write_text(yaml.safe_dump({"task": "bigfive", "max_epochs": 3}))

    def test_loads_three_files(self):
        config = ExperimentConfig.from_files(self.data_path, self.model_path, self.train_path)
        self.assertEqual(config.data.features, ["wavlm", "dinov2"])
        self.assertEqual(config.model.d_model, 32)
        self.assertEqual(config.train.max_epochs, 3)

    def test_builder_only_keys_are_ignored_by_the_dataloader_config(self):
        """label_csv drives the builder; DataConfig must not choke on it."""
        config = ExperimentConfig.from_files(self.data_path, self.model_path, self.train_path)
        self.assertFalse(hasattr(config.data, "label_csv"))

    def test_overrides_apply(self):
        config = ExperimentConfig.from_files(
            self.data_path,
            self.model_path,
            self.train_path,
            overrides={"data.window_seconds": 5.0, "train.optimizer.lr": 0.01},
        )
        self.assertEqual(config.data.window_seconds, 5.0)
        self.assertEqual(config.train.optimizer.lr, 0.01)

    def test_lint_with_multiple_features_is_rejected_at_config_load(self):
        # setUp lists two features; a lint family must fail here, not deep in the
        # model builder.
        with self.assertRaises(ConfigError):
            ExperimentConfig.from_files(
                self.data_path,
                self.model_path,
                self.train_path,
                overrides={"model.family": "lint"},
            )

    def test_lint_with_one_feature_is_accepted(self):
        config = ExperimentConfig.from_files(
            self.data_path,
            self.model_path,
            self.train_path,
            overrides={"model.family": "lint", "data.features": ["wavlm"]},
        )
        self.assertEqual(config.model.family, "lint")
        self.assertEqual(config.data.features, ["wavlm"])

    def test_window_experiment_is_a_one_key_change(self):
        for seconds in (15.0, 10.0, 5.0):
            with self.subTest(seconds):
                config = ExperimentConfig.from_files(
                    self.data_path,
                    self.model_path,
                    self.train_path,
                    overrides={"data.window_seconds": seconds},
                )
                self.assertEqual(config.data.window_seconds, seconds)
                self.assertEqual(config.data.h5_path, "data/processed/FI/fi.h5")

    def test_flat_dict_is_mlflow_ready(self):
        config = ExperimentConfig.from_files(self.data_path, self.model_path, self.train_path)
        flat = config.to_flat_dict()

        self.assertEqual(flat["data.batch_size"], 8)
        self.assertEqual(flat["model.d_model"], 32)
        self.assertEqual(flat["train.optimizer.lr"], 0.001)
        self.assertEqual(flat["data.features"], "wavlm,dinov2")
        self.assertTrue(all(not isinstance(v, dict | list) for v in flat.values()))

    def test_missing_file_raises(self):
        with self.assertRaises(ConfigError):
            ExperimentConfig.from_files(self.root / "nope.yaml", self.model_path, self.train_path)

    def test_non_mapping_yaml_raises(self):
        bad = self.root / "bad.yaml"
        bad.write_text("- just\n- a list\n")
        with self.assertRaises(ConfigError):
            load_yaml(bad)

    def test_resolve_feature_specs_in_order(self):
        feature_dir = self.root / "features"
        feature_dir.mkdir()
        (feature_dir / "wavlm.yaml").write_text(
            yaml.safe_dump(
                {
                    "name": "wavlm",
                    "source_dir": "d",
                    "source_format": "pickle_layers",
                    "dim": 768,
                    "rate_hz": 49.79,
                    "layers": [9],
                }
            )
        )
        (feature_dir / "dinov2.yaml").write_text(
            yaml.safe_dump(
                {
                    "name": "dinov2",
                    "source_dir": "d",
                    "source_format": "safetensors",
                    "dim": 768,
                    "rate_hz": 25.02,
                }
            )
        )

        config = ExperimentConfig.from_files(self.data_path, self.model_path, self.train_path)
        specs = config.resolve_feature_specs(feature_dir)

        self.assertEqual([s.name for s in specs], ["wavlm", "dinov2"])
        self.assertEqual(specs[0].layers, (9,))

    def test_resolve_feature_specs_missing_file_raises(self):
        config = ExperimentConfig.from_files(self.data_path, self.model_path, self.train_path)
        with self.assertRaises(ConfigError):
            config.resolve_feature_specs(self.root / "features")


if __name__ == "__main__":
    unittest.main()
