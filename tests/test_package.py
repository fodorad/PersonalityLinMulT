"""Smoke tests for the package and the configs it ships."""

import unittest
from dataclasses import asdict
from pathlib import Path


class TestPackage(unittest.TestCase):
    def test_import_and_version(self):
        import personalitylinmult

        self.assertTrue(hasattr(personalitylinmult, "__version__"))
        self.assertIsInstance(personalitylinmult.__version__, str)

    def test_module_constants(self):
        import personalitylinmult

        self.assertIsInstance(personalitylinmult.PROJECT_ROOT, Path)
        self.assertIsInstance(personalitylinmult.LIBRARY_ROOT, Path)
        self.assertIsInstance(personalitylinmult.CONFIG_DIR, Path)
        self.assertEqual(personalitylinmult.LIBRARY_ROOT.name, "personalitylinmult")

    def test_constants_point_at_real_directories(self):
        """A constant naming a directory that does not exist is a trap."""
        import personalitylinmult

        self.assertTrue(personalitylinmult.PROJECT_ROOT.is_dir())
        self.assertTrue(personalitylinmult.LIBRARY_ROOT.is_dir())
        self.assertTrue(personalitylinmult.CONFIG_DIR.is_dir())


class TestShippedConfigs(unittest.TestCase):
    """The configs under config/FI must load and validate as shipped."""

    def setUp(self):
        from personalitylinmult import CONFIG_DIR

        self.config_dir = CONFIG_DIR / "FI"

    def _experiment(self, data="avt", model="linmult", train="bigfive"):
        from personalitylinmult.train.config import ExperimentConfig

        return ExperimentConfig.from_files(
            self.config_dir / "data" / f"{data}.yaml",
            self.config_dir / "model" / f"{model}.yaml",
            self.config_dir / "train" / f"{train}.yaml",
        )

    def test_every_feature_config_is_valid(self):
        from personalitylinmult.data.schema import FeatureSpec
        from personalitylinmult.train.config import load_yaml

        paths = sorted((self.config_dir / "features").glob("*.yaml"))
        self.assertTrue(paths, "no feature configs found")
        for path in paths:
            with self.subTest(path.stem):
                spec = FeatureSpec.from_dict(load_yaml(path))
                self.assertEqual(spec.name, path.stem)

    def test_every_experiment_config_combination_loads(self):
        for data in ("avt", "smoke"):
            for model in ("linmult", "mult"):
                for train in ("bigfive", "interview", "smoke"):
                    with self.subTest(data=data, model=model, train=train):
                        config = self._experiment(data, model, train)
                        self.assertTrue(config.data.features)
                        self.assertTrue(config.train.targets)

    def test_configured_features_all_have_a_feature_config(self):
        config = self._experiment()
        specs = config.resolve_feature_specs(self.config_dir / "features")
        self.assertEqual([s.name for s in specs], config.data.features)

    def test_bigfive_trains_on_emotional_stability(self):
        config = self._experiment(train="bigfive")
        self.assertIn("emotional_stability", config.train.targets)
        self.assertNotIn("neuroticism", config.train.targets)
        self.assertEqual(config.train.output_dim, 5)

    def test_interview_is_its_own_single_target_task(self):
        config = self._experiment(train="interview")
        self.assertEqual(config.train.targets, ["interview"])
        self.assertEqual(config.train.output_dim, 1)

    def test_every_listed_feature_has_a_config(self):
        """A feature named in the data config but lacking a YAML aborts the build."""
        from personalitylinmult import CONFIG_DIR

        feature_dir = CONFIG_DIR / "FI" / "features"
        for name in self._experiment().data.features:
            self.assertTrue(
                (feature_dir / f"{name}.yaml").is_file(),
                f"{name} is listed in the data config but has no feature YAML.",
            )

    def test_default_excludes_unextracted_features(self):
        """MARLIN has never been extracted; the builder would hard-fail on it."""
        self.assertNotIn("marlin", self._experiment().data.features)

    def test_default_window_is_fifteen_seconds(self):
        self.assertEqual(self._experiment().data.window_seconds, 15.0)

    def test_mult_differs_from_linmult_only_by_attention(self):
        linmult = self._experiment(model="linmult").model
        mult = self._experiment(model="mult").model

        self.assertEqual(linmult.attention_type, "linear")
        self.assertEqual(mult.attention_type, "softmax")

        a, b = asdict(linmult), asdict(mult)
        self.assertEqual({k for k in a if a[k] != b[k]}, {"attention_type", "name"})

    def test_no_config_leaks_an_absolute_local_path(self):
        import yaml

        for path in sorted(self.config_dir.rglob("*.yaml")):
            with self.subTest(path.name):
                text = path.read_text()
                self.assertNotIn("/Users/", text)
                self.assertNotIn("/Volumes/", text)
                self.assertNotIn("/home/", text)
                self.assertIsInstance(yaml.safe_load(text), dict)


class TestNoLocalPathLeaksInSource(unittest.TestCase):
    """A hardcoded absolute local path (someone's laptop's home/mount, an
    internal hostname) is a real leak even inside a docstring/comment, not
    just a config value -- confirmed by a real instance found and fixed in
    this project (a docstring pointing at a sibling repo's local checkout).
    Scans every shipped .py file, not just YAML configs, since the config
    scanner above never would have caught that one."""

    _FORBIDDEN = ("/Users/", "/Volumes/", "/home/")

    def test_no_source_file_leaks_an_absolute_local_path(self):
        import personalitylinmult

        for path in sorted(personalitylinmult.LIBRARY_ROOT.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            with self.subTest(str(path.relative_to(personalitylinmult.PROJECT_ROOT))):
                text = path.read_text()
                for needle in self._FORBIDDEN:
                    self.assertNotIn(needle, text)


if __name__ == "__main__":
    unittest.main()
