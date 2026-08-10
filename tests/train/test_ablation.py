"""Tests for the FI feature-ablation matrix and its run manifest."""

from __future__ import annotations

import unittest

from personalitylinmult.train.ablation import (
    ACOUSTIC,
    ALL_FEATURES,
    RUNS,
    TEXTUAL,
    VISUAL,
    Run,
    run_by_name,
    validate_manifest,
)


class TestRun(unittest.TestCase):
    def test_lint_run_must_be_single_feature(self):
        with self.assertRaises(ValueError):
            Run(name="bad", features=("wavlm", "dinov2"), family="lint")

    def test_run_must_have_features(self):
        with self.assertRaises(ValueError):
            Run(name="empty", features=(), family="linmult")

    def test_overrides_carry_features_family_and_run_name(self):
        run = Run(name="uni_wavlm", features=("wavlm",), family="lint")
        overrides = run.overrides()
        self.assertEqual(overrides["data.features"], ["wavlm"])
        self.assertEqual(overrides["model.family"], "lint")
        self.assertEqual(overrides["model.name"], "LinT")
        self.assertEqual(overrides["train.mlflow.run_name"], "uni_wavlm")

    def test_linmult_run_names_itself_linmult(self):
        run = Run(name="within_visual", features=VISUAL)
        self.assertEqual(run.overrides()["model.name"], "LinMulT")


class TestManifest(unittest.TestCase):
    def test_manifest_is_valid(self):
        validate_manifest()

    def test_expected_run_count(self):
        # 7 unimodal (2 acoustic + 3 visual + 2 textual) + 3 within-modality + 1 full.
        self.assertEqual(len(RUNS), 7 + 3 + 1)

    def test_every_feature_has_a_unimodal_baseline(self):
        unimodal = {r.features[0] for r in RUNS if r.family == "lint"}
        self.assertEqual(unimodal, set(ALL_FEATURES))

    def test_within_modality_runs_cover_each_modality(self):
        by_name = {r.name: r.features for r in RUNS}
        self.assertEqual(by_name["within_acoustic"], ACOUSTIC)
        self.assertEqual(by_name["within_visual"], VISUAL)
        self.assertEqual(by_name["within_textual"], TEXTUAL)

    def test_full_run_uses_every_feature(self):
        self.assertEqual(run_by_name("full_avt").features, ALL_FEATURES)

    def test_textual_covers_both_models_over_gt(self):
        # XLM-RoBERTa and mmBERT, ground-truth transcript only.
        from personalitylinmult.train.ablation import TEXTUAL

        self.assertEqual(
            set(TEXTUAL),
            {"xlm_roberta_gt", "mmbert_gt"},
        )

    def test_unknown_run_name_raises(self):
        with self.assertRaises(KeyError):
            run_by_name("does_not_exist")

    def test_validate_rejects_duplicate_names(self):
        dupe = [Run("x", ("wavlm",), "lint"), Run("x", ("dinov2",), "lint")]
        with self.assertRaises(ValueError):
            validate_manifest(dupe)

    def test_validate_rejects_unknown_feature(self):
        bad = [Run("y", ("not_a_feature",), "lint")]
        with self.assertRaises(ValueError):
            validate_manifest(bad)


class TestBuildConfig(unittest.TestCase):
    """The ablation driver composes each run onto the shipped FI configs."""

    def _build(self, run: Run):
        from personalitylinmult.train.run_ablation import (
            DEFAULT_DATA,
            DEFAULT_TRAIN,
            build_config,
        )

        # model_path=None: build_config picks lint.yaml or linmult.yaml per
        # the run's own family, matching what the real sweep does.
        return build_config(run, DEFAULT_DATA, None, DEFAULT_TRAIN)

    def test_picks_lint_yaml_for_a_lint_run_and_linmult_yaml_for_linmult(self):
        # lint.yaml and linmult.yaml are separate, self-contained files (one
        # per architecture); build_config must pick the right one per run
        # instead of relying on a shared file + a model.family override.
        lint_config = self._build(run_by_name("uni_dinov2"))
        self.assertEqual(lint_config.model.name, "LinT")
        linmult_config = self._build(run_by_name("full_avt"))
        self.assertEqual(linmult_config.model.name, "LinMulT")

    def test_unimodal_run_yields_a_lint_config_with_one_feature(self):
        config = self._build(run_by_name("uni_dinov2"))
        self.assertEqual(config.model.family, "lint")
        self.assertEqual(config.data.features, ["dinov2"])
        self.assertEqual(config.train.mlflow.run_name, "uni_dinov2")

    def test_full_run_yields_a_linmult_config_with_every_feature(self):
        config = self._build(run_by_name("full_avt"))
        self.assertEqual(config.model.family, "linmult")
        self.assertEqual(config.data.features, list(ALL_FEATURES))

    def test_every_manifest_run_composes_and_builds_a_model(self):
        # The whole matrix must resolve against the real shipped configs AND build
        # its model, so a broken run is caught here rather than partway through an
        # overnight sweep. Dummy per-feature widths stand in for the h5 attrs.
        from personalitylinmult.train.module import build_model

        widths = {
            "wavlm": 768,
            "emotion2vec": 768,
            "dinov2": 768,
            "emotieffnet": 1280,
            "farl": 512,
            "xlm_roberta_gt": 768,
            "mmbert_gt": 768,
        }
        for run in RUNS:
            with self.subTest(run.name):
                config = self._build(run)
                self.assertEqual(config.data.features, list(run.features))
                self.assertEqual(config.model.family, run.family)
                dims = [widths[f] for f in config.data.features]
                model = build_model(config.model, dims, config.train.output_dim)
                self.assertEqual(type(model).__name__, run.overrides()["model.name"])


class TestSelectRuns(unittest.TestCase):
    def test_empty_selection_returns_the_whole_matrix(self):
        from personalitylinmult.train.run_ablation import select_runs

        self.assertEqual(select_runs([]), RUNS)

    def test_named_selection_keeps_only_those_runs_in_order(self):
        from personalitylinmult.train.run_ablation import select_runs

        selected = select_runs(["full_avt", "uni_dinov2"])
        self.assertEqual([r.name for r in selected], ["full_avt", "uni_dinov2"])

    def test_unknown_name_raises(self):
        from personalitylinmult.train.run_ablation import select_runs

        with self.assertRaises(KeyError):
            select_runs(["nope"])


class TestRunSweep(unittest.TestCase):
    """The sweep isolates each run so one failure does not abort the rest."""

    def _runs(self):
        return [
            Run("a", ("wavlm",), "lint"),
            Run("b", ("dinov2",), "lint"),
            Run("c", ("farl",), "lint"),
        ]

    def test_a_failing_run_does_not_abort_the_sweep(self):
        from personalitylinmult.train.run_ablation import run_sweep

        def run_one(run):
            if run.name == "b":
                raise RuntimeError("simulated CUDA OOM")
            return 0.5

        results = run_sweep(self._runs(), run_one, summarise=False)

        # b failed, but a and c still ran and recorded real scores.
        self.assertEqual(results["a"], 0.5)
        self.assertEqual(results["c"], 0.5)
        self.assertTrue(results["b"].startswith("FAILED: RuntimeError"))

    def test_all_runs_are_attempted_and_recorded(self):
        from personalitylinmult.train.run_ablation import run_sweep

        seen = []

        def run_one(run):
            seen.append(run.name)
            return 1.0

        results = run_sweep(self._runs(), run_one, summarise=False)
        self.assertEqual(seen, ["a", "b", "c"])
        self.assertEqual(set(results), {"a", "b", "c"})


if __name__ == "__main__":
    unittest.main()
