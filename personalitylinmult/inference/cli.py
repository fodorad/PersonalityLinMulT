"""Command-line inference: features in, Big Five scores out.

Example:
    Predict with the audio-only model from an ``.npz`` of features::

        uv run python -m personalitylinmult.inference \
            --checkpoint results/FI/bigfive/checkpoints/best.ckpt \
            --features clip_features.npz

The ``.npz`` holds one ``(T, F)`` array per feature the model expects, keyed by
feature name (``wavlm``, ``emotion2vec``, ...).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from personalitylinmult.inference.predictor import FiPredictor

logger = logging.getLogger(__name__)
"""Module-level logger."""


def load_features(path: Path) -> dict[str, np.ndarray]:
    """Load a ``{feature_name: (T, F)}`` mapping from an ``.npz`` file.

    Args:
        path (Path): Path to a ``.npz`` with one array per feature.

    Returns:
        dict[str, np.ndarray]: The feature mapping.
    """
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a trained .ckpt.")
    parser.add_argument(
        "--features", type=Path, required=True, help=".npz of per-feature (T, F) arrays."
    )
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    predictor = FiPredictor.from_checkpoint(args.checkpoint)
    features = load_features(args.features)
    scores = predictor.predict(features)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
