"""Regression test for the emotional_stability / neuroticism label fix.

The raw ChaLearn FI V2 annotation pickles store their ``"neuroticism"`` key as
``1 - original_neuroticism``: the dataset authors already inverted it so all six
fields (OCEAN + interview) point the same positive direction, but kept the old
key name instead of renaming it. A previous version of ``preprocess.py`` applied
a second, spurious ``1 - x`` inversion when computing ``emotional_stability``,
silently swapping the two output columns relative to their true meaning. This
test pins the correct behaviour with a real (synthetic) annotation pickle --
not a mock -- so the bug cannot silently reintroduce itself.
"""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from personalitylinmult.preprocess.fi.preprocess import main

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]


def write_raw_fi(
    raw_dir: Path, subset_name: str, pkl_stem: str, records: dict[str, dict[str, float]]
) -> None:
    """Write a real annotation + transcription pickle pair for one subset.

    Args:
        raw_dir: Raw FI root (will contain a ``gt/`` subdirectory).
        subset_name: One of ``training``, ``validation``, ``test`` (the pickle
            filename suffix ChaLearn uses).
        pkl_stem: Unused positional kept for symmetry with the real layout.
        records: ``{video_filename: {trait: value}}`` for every OCEAN trait plus
            ``interview``.
    """
    gt_dir = raw_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)

    annotation = {trait: {} for trait in [*TRAITS, "interview"]}
    for vid, traits in records.items():
        for trait, value in traits.items():
            annotation[trait][vid] = value

    with open(gt_dir / f"annotation_{subset_name}.pkl", "wb") as f:
        pickle.dump(annotation, f)
    with open(gt_dir / f"transcription_{subset_name}.pkl", "wb") as f:
        pickle.dump(dict.fromkeys(records, "hello world"), f)


class TestEmotionalStabilityFix(unittest.TestCase):
    """The pickle's "neuroticism" value must flow into emotional_stability
    directly, not through a second inversion."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.raw_dir = Path(self._tmp.name) / "raw"
        self.output_dir = Path(self._tmp.name) / "processed"

        # A clip where the raw pickle's "neuroticism" (== stability) is 0.6354,
        # matching the real clip (26CMYJn3u6Y.000) that surfaced this bug.
        records = {
            "clipA.mp4": {
                "openness": 0.811111,
                "conscientiousness": 0.737864,
                "extraversion": 0.532710,
                "agreeableness": 0.549451,
                "neuroticism": 0.635417,  # raw pickle value; this IS stability
                "interview": 0.672897,
            },
            "clipB.mp4": {
                "openness": 0.2,
                "conscientiousness": 0.3,
                "extraversion": 0.4,
                "agreeableness": 0.5,
                "neuroticism": 0.1,  # low raw value: this clip is LOW stability
                "interview": 0.15,
            },
        }
        write_raw_fi(self.raw_dir, "training", "training", records)
        write_raw_fi(self.raw_dir, "validation", "validation", {})
        write_raw_fi(self.raw_dir, "test", "test", {})

        main(raw_dir=self.raw_dir, output_dir=self.output_dir, labels_only=True)
        self.df = pd.read_csv(self.output_dir / "label.csv").set_index("video_id")

    def test_emotional_stability_matches_the_raw_pickle_value_directly(self):
        # The pickle's "neuroticism" field IS emotional stability -- no inversion.
        self.assertAlmostEqual(self.df.loc["clipA", "emotional_stability"], 0.635417, places=5)
        self.assertAlmostEqual(self.df.loc["clipB", "emotional_stability"], 0.1, places=5)

    def test_neuroticism_column_is_the_derived_complement(self):
        # neuroticism is now genuinely derived, not the raw (mislabelled) field.
        self.assertAlmostEqual(self.df.loc["clipA", "neuroticism"], 1.0 - 0.635417, places=5)
        self.assertAlmostEqual(self.df.loc["clipB", "neuroticism"], 0.9, places=5)

    def test_stability_and_neuroticism_sum_to_one(self):
        total = self.df["emotional_stability"] + self.df["neuroticism"]
        pd.testing.assert_series_equal(total, pd.Series(1.0, index=self.df.index, name=total.name))

    def test_high_raw_neuroticism_value_means_high_stability_not_low(self):
        # clipA's raw pickle value (0.635) is HIGH -- it must land as HIGH
        # emotional_stability, not be inverted into high neuroticism. This is
        # the exact failure mode of the original bug.
        self.assertGreater(self.df.loc["clipA", "emotional_stability"], 0.5)
        self.assertLess(self.df.loc["clipA", "neuroticism"], 0.5)


if __name__ == "__main__":
    unittest.main()
