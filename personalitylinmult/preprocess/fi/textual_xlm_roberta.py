"""Extract XLM-RoBERTa token-level features for the FI dataset.

The **ground-truth transcript** is used — the ``text`` column of
``data/processed/FI/label.csv`` (the dataset's original annotation).

XLM-RoBERTa **token-level** hidden states are extracted via
:class:`~exordium.text.xlm_roberta.XlmRobertaWrapper` with ``pooling="none"``,
shape ``(T, 768)`` — one vector per token. This is the representation a
cross-modal sequence model (LinMulT) is built to consume: it lets cross-modal
attention operate token-by-token, which a single pooled sentence vector cannot.
Length ``T`` varies per sample, so a single HDF5 file is used with one dataset
per ``video_id`` — every sample at its exact length, no padding, still one file
to distribute and a fast, direct lookup at train time.

Output:
    ``data/processed/FI/xlm_roberta_gt.h5``  — from label.csv

The script is resumable: a ``video_id`` already present in the output
file is skipped. Use ``--overwrite`` to force recomputation.
"""

import argparse
import logging
from pathlib import Path

import h5py
import pandas as pd
from exordium.text.xlm_roberta import XlmRobertaWrapper
from tqdm import tqdm

logger = logging.getLogger(__name__)
"""Module-level logger."""

DB_PROCESSED = Path("data/processed/FI")
"""Processed FI root."""

DEFAULT_GT_CSV = DB_PROCESSED / "label.csv"
"""Ground-truth transcript source (``video_id``, ``text`` columns)."""

DEFAULT_GT_H5 = DB_PROCESSED / "xlm_roberta_gt.h5"
"""Output HDF5 file for ground-truth-transcript features."""


def load_texts(csv_path: Path) -> dict[str, str]:
    """Return a ``{video_id: text}`` mapping from a transcript CSV.

    Args:
        csv_path: Path to a CSV with ``video_id`` and ``text`` columns.

    Returns:
        Mapping of video ID to its transcript text (empty string if missing).
    """
    df = pd.read_csv(csv_path)
    return {
        str(video_id): "" if pd.isna(text) else str(text)
        for video_id, text in zip(df["video_id"], df["text"], strict=True)
    }


def extract_source(
    texts: dict[str, str],
    h5_path: Path,
    xlm_roberta: XlmRobertaWrapper,
    overwrite: bool,
    desc: str,
) -> None:
    """Extract XLM-RoBERTa token features for one text source into an HDF5 file.

    Args:
        texts: Mapping of ``video_id`` to transcript text.
        h5_path: Destination HDF5 file (created/appended to).
        xlm_roberta: Loaded XLM-RoBERTa wrapper.
        overwrite: If True, recompute features already present in the file.
        desc: tqdm progress bar description.
    """
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    n_skipped = 0
    n_failed = 0
    with h5py.File(h5_path, "a") as f:
        for video_id, text in tqdm(texts.items(), desc=desc, total=len(texts)):
            if not overwrite and video_id in f:
                n_skipped += 1
                continue

            try:
                # pooling="none" -> token-level (1, T, 768); squeeze the batch axis.
                feature = xlm_roberta.predict(text, pooling="none").squeeze(0)  # (T, 768)
            except Exception as e:
                logger.exception(f"{video_id}: {e}")
                n_failed += 1
                continue

            if video_id in f:
                del f[video_id]
            f.create_dataset(video_id, data=feature)

    logger.info(f"Done. h5_path={h5_path}, skipped={n_skipped}, failed={n_failed}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gt-csv", type=Path, default=DEFAULT_GT_CSV)
    parser.add_argument("--gt-h5", type=Path, default=DEFAULT_GT_H5)
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="Device index for feature extraction (negative = CPU).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute features even if already present in the output file.",
    )
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    device_id = args.gpu_id if args.gpu_id >= 0 else -1
    logger.info(f"Loading XLM-RoBERTa on gpu={device_id}...")
    xlm_roberta = XlmRobertaWrapper(device_id=device_id)

    if args.gt_csv.exists():
        gt_texts = load_texts(args.gt_csv)
        logger.info(f"Found {len(gt_texts)} video_ids in {args.gt_csv}")
        extract_source(gt_texts, args.gt_h5, xlm_roberta, args.overwrite, "XLM-RoBERTa GT text")
    else:
        logger.warning(f"{args.gt_csv} not found; skipping ground-truth transcript source.")


if __name__ == "__main__":
    main()
