"""Extract WavLM features for the FI dataset.

Iterates over ``video_id``s listed in ``data/processed/FI/label.csv``, reads
the matching ``{video_id}.wav`` clip from ``data/processed/FI/audio``, and
extracts WavLM hidden-state features via
:class:`~exordium.audio.wavlm.WavlmWrapper`.

Output:
    ``data/processed/FI/wavlm_baseplus/{video_id}.pkl`` (directory name
    follows the ``--wavlm-model`` variant, e.g. ``wavlm_base``,
    ``wavlm_baseplus``, ``wavlm_large``) — a list of per-layer
    ``(T, hidden_size)`` numpy arrays, holding only the layers named by
    ``--layers`` and stored as ``--dtype``.

**Why only some layers.** Keeping all 12 layers in fp32 costs 28 MB per clip
(263 GB across FI) to serve a downstream h5 build that reads one layer and
discards eleven. On a USB-attached HDD that read amplification dominates the
build: measured 35 MB/s, i.e. ~2.2 h just to move bytes that are then thrown
away. Extracting the wanted layer alone is both smaller on disk and faster to
consume, and recomputing on the GPU (~0.76 s/clip) is cheaper than reading the
fat pickles back. ``--layers`` keeps this a choice, not a hard-coding: pass
several to enable linmult's learned weighted-sum over layers.

The script is resumable: a clip whose output file already exists is skipped.
Use ``--overwrite`` to force recomputation.
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from exordium.audio.wavlm import WavlmWrapper
from tqdm import tqdm

logger = logging.getLogger(__name__)
"""Module-level logger."""

DB_PROCESSED = Path("data/processed/FI")
"""Processed FI root (contains ``label.csv`` and ``audio/``)."""

DEFAULT_LABEL_CSV = DB_PROCESSED / "label.csv"
"""Source of ``video_id``s to process."""

DEFAULT_AUDIO_DIR = DB_PROCESSED / "audio"
"""Input directory of per-clip ``{video_id}.wav`` files."""

DEFAULT_WAVLM_MODEL = "base+"
"""Default WavLM variant."""

WAVLM_DIR_NAMES = {"base": "wavlm_base", "base+": "wavlm_baseplus", "large": "wavlm_large"}
"""Maps each WavLM variant to its default output directory name."""

DEFAULT_LAYERS = (9,)
"""Transformer layers kept on disk.

Layer 9 is the usual sweet spot for paralinguistic and affect tasks, and is what
``config/FI/features/wavlm.yaml`` selects. Storing only what is used keeps the
tree ~11 GB instead of 263 GB.
"""

DEFAULT_DTYPE = "float16"
"""On-disk dtype.

The h5 builder casts to fp16 anyway, so fp32 here would be discarded downstream
while doubling the tree. Measured across the real features the largest magnitude
is ~31, far inside fp16's 65504 ceiling.
"""


def default_wavlm_dir(wavlm_model: str) -> Path:
    """Return the default WavLM output directory for a given model variant.

    Args:
        wavlm_model: WavLM variant (``"base"``, ``"base+"``, or ``"large"``).

    Returns:
        Default output directory, e.g. ``data/processed/FI/wavlm_baseplus``.
    """
    return DB_PROCESSED / WAVLM_DIR_NAMES[wavlm_model]


def select_layers(
    features: list[np.ndarray], layers: tuple[int, ...], dtype: str
) -> list[np.ndarray]:
    """Keep the requested layers and cast them for storage.

    Args:
        features: All per-layer ``(T, hidden_size)`` arrays from the extractor.
        layers: Indices to keep, into ``features``.
        dtype: Target dtype name, e.g. ``"float16"``.

    Returns:
        The selected layers, in the order given by ``layers``.

    Raises:
        IndexError: If a requested layer does not exist. Failing here beats
            writing a short list that a later run would silently misread as a
            different layer.
    """
    for layer in layers:
        if not 0 <= layer < len(features):
            raise IndexError(
                f"Layer {layer} out of range: the model returned "
                f"{len(features)} layers (0-{len(features) - 1})."
            )
    return [features[layer].astype(dtype) for layer in layers]


def extract_clip(
    extractor: WavlmWrapper,
    audio_path: Path,
    output_path: Path,
    layers: tuple[int, ...],
    dtype: str,
) -> None:
    """Extract one clip and write only the requested layers.

    ``WavlmWrapper.audio_to_feature`` is deliberately called *without*
    ``output_path``: its ``load_or_create`` decorator pickles whatever the
    function returns, so passing an output path would write all 12 layers before
    this code ever sees them. Taking the return value and saving it here is what
    keeps the fat tensor from reaching the disk.

    Args:
        extractor: Loaded WavLM wrapper.
        audio_path: Input ``{video_id}.wav``.
        output_path: Destination ``{video_id}.pkl``.
        layers: Transformer layers to keep.
        dtype: On-disk dtype.
    """
    features = extractor.audio_to_feature(audio_path)
    selected = select_layers(features, layers, dtype)

    # Record WHICH layers these are, rather than leaving it to position. A bare
    # list of one array cannot distinguish "layer 9" from "layer 0", so a reader
    # asking for layer 9 would either fail or, worse, silently take whatever sits
    # at that index. The layer ids travel with the data instead.
    payload = {"layers": list(layers), "features": selected}

    # Write to a temp file and rename, so an interrupted run cannot leave a
    # truncated pickle that the resume logic would mistake for a finished clip.
    tmp_path = output_path.with_suffix(".pkl.tmp")
    with tmp_path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(output_path)


def load_video_ids(label_csv: Path) -> list[str]:
    """Return the ordered list of ``video_id``s from the label CSV.

    Args:
        label_csv: Path to ``label.csv`` with a ``video_id`` column.

    Returns:
        List of video IDs in file order.
    """
    df = pd.read_csv(label_csv)
    return df["video_id"].astype(str).tolist()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label-csv", type=Path, default=DEFAULT_LABEL_CSV)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument(
        "--wavlm-dir",
        type=Path,
        default=None,
        help="Output directory for WavLM features. Defaults to "
        "data/processed/FI/wavlm_<model> based on --wavlm-model.",
    )
    parser.add_argument(
        "--wavlm-model",
        default=DEFAULT_WAVLM_MODEL,
        choices=["base", "base+", "large"],
        help="WavLM variant to use.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=list(DEFAULT_LAYERS),
        help="Transformer layers to keep on disk. Several layers enable "
        "linmult's learned weighted-sum over layers, at proportional cost.",
    )
    parser.add_argument(
        "--dtype",
        default=DEFAULT_DTYPE,
        choices=["float16", "float32"],
        help="On-disk dtype. The h5 builder casts to float16 regardless.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="Device index for feature extraction (negative = CPU).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute features even if the output file already exists.",
    )
    args = parser.parse_args()
    if args.wavlm_dir is None:
        args.wavlm_dir = default_wavlm_dir(args.wavlm_model)

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    video_ids = load_video_ids(args.label_csv)
    logger.info(f"Found {len(video_ids)} video_ids in {args.label_csv}")
    if not video_ids:
        logger.warning(f"No video_ids in {args.label_csv}; nothing to do.")
        return

    device_id = args.gpu_id if args.gpu_id >= 0 else -1
    layers = tuple(args.layers)
    logger.info(f"Loading WavLM ({args.wavlm_model}) on gpu={device_id}...")
    wavlm_extractor = WavlmWrapper(device_id=device_id, model_name=args.wavlm_model)

    args.wavlm_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Keeping layers {list(layers)} as {args.dtype} -> {args.wavlm_dir}. "
        "Clips already present are skipped; use --overwrite to recompute."
    )

    n_skipped = 0
    n_failed = 0
    for video_id in tqdm(video_ids, desc="WavLM features"):
        audio_path = args.audio_dir / f"{video_id}.wav"
        wavlm_path = args.wavlm_dir / f"{video_id}.pkl"

        if not audio_path.exists():
            logger.warning(f"{video_id}: missing audio file {audio_path}; skipping.")
            n_failed += 1
            continue

        if not args.overwrite and wavlm_path.exists():
            n_skipped += 1
            continue

        try:
            extract_clip(wavlm_extractor, audio_path, wavlm_path, layers, args.dtype)
        except Exception as e:
            logger.exception(f"{video_id}: {e}")
            n_failed += 1

    logger.info(
        f"Done. wavlm_dir={args.wavlm_dir}, layers={list(layers)}, "
        f"dtype={args.dtype}, skipped={n_skipped}, failed={n_failed}"
    )


if __name__ == "__main__":
    main()
