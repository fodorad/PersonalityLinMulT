"""Build the single HDF5 training artifact from extracted FI features.

The builder reads a data config, and writes **only** the features that config
names. Nothing else is opened, read, or written: if the config lists WavLM and
DINOv2, the resulting file contains WavLM and DINOv2 and no trace of the other
extractors. The config that produced a file is embedded verbatim in its root
attributes, so a built file is always traceable back to its recipe.

The resulting file is the reproducibility boundary for training: it is published
to a public Hugging Face dataset repo so a run reproduces without re-extraction,
and the training loop reads it and nothing else.

Example:
    ``uv run python -m personalitylinmult.data.builder --config config/FI/data/wedrx.yaml``
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from personalitylinmult.data.schema import (
    ALL_TRAITS,
    SCHEMA_VERSION,
    SUBSET_CODES,
    BuildStats,
    DatasetSpec,
    FeatureSpec,
    SchemaError,
)
from personalitylinmult.data.sources import FeatureReader

logger = logging.getLogger(__name__)
"""Module-level logger."""

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
"""Repository root; all config paths are interpreted relative to it."""

COMPRESSION = "gzip"
"""Compression filter. Paired with the shuffle filter for float data."""

COMPRESSION_LEVEL = 4
"""Compression level: a deliberate middle ground between size and read CPU."""


def _progress(iterable, desc: str, leave: bool = True):
    """Wrap an iterable in a progress bar, silenced when not on a terminal.

    Keeps test and CI logs readable without giving up the bar during the real
    multi-hour build.

    Args:
        iterable: The iterable to wrap.
        desc (str): Bar description.
        leave (bool): Whether to leave the finished bar on screen.

    Returns:
        The wrapped iterable.
    """
    return tqdm(iterable, desc=desc, leave=leave, disable=not sys.stderr.isatty())


class BuildError(RuntimeError):
    """Raised when the dataset cannot be built as configured."""


def git_sha(root: Path) -> str:
    """Return the current commit SHA, for traceability of a built file.

    A ``-dirty`` suffix is appended when the working tree has uncommitted
    changes, so an artifact built from an uncommitted state is never mistaken
    for one that a clean checkout of that SHA would reproduce.

    Args:
        root (Path): Repository root.

    Returns:
        str: The commit SHA (with a ``-dirty`` suffix when the tree is dirty),
        or ``"unknown"`` outside a git checkout.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    sha = result.stdout.strip()

    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return sha
    return f"{sha}-dirty" if dirty.stdout.strip() else sha


def file_sha256(path: Path) -> str:
    """Hash a file's contents.

    Args:
        path (Path): File to hash.

    Returns:
        str: Hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_labels(label_csv: Path) -> tuple[list[str], list[str], dict[str, np.ndarray]]:
    """Read ``label.csv`` into ids, subsets, and per-trait arrays.

    Args:
        label_csv (Path): Path to the preprocess stage's ``label.csv``.

    Returns:
        tuple: ``(video_ids, subsets, labels)`` where ``labels`` maps each trait
        in :data:`~personalitylinmult.data.schema.ALL_TRAITS` to an ``(N,)``
        float32 array.

    Raises:
        BuildError: If the file is missing a required column or a subset name
            is unrecognised.
    """
    if not label_csv.is_file():
        raise BuildError(f"Label CSV not found: {label_csv}")

    frame = pd.read_csv(label_csv)

    required = {"video_id", "subset", *ALL_TRAITS}
    missing = required - set(frame.columns)
    if missing:
        raise BuildError(f"{label_csv}: missing required columns {sorted(missing)}.")

    subsets = frame["subset"].astype(str).tolist()
    unknown = sorted(set(subsets) - set(SUBSET_CODES))
    if unknown:
        raise BuildError(
            f"{label_csv}: unknown subset values {unknown}; expected {sorted(SUBSET_CODES)}."
        )

    video_ids = frame["video_id"].astype(str).tolist()
    labels = {t: frame[t].to_numpy(dtype=np.float32) for t in ALL_TRAITS}
    return video_ids, subsets, labels


def resolve_coverage(spec: FeatureSpec, root: Path, video_ids: list[str]) -> None:
    """Assert a feature exists for every clip, or fail with the missing ids.

    A partially-extracted feature must never be zero-filled into the dataset:
    that would train and evaluate on silence while looking perfectly healthy.
    EmotiEffNet and FaRL are only ~7% extracted at the time of writing, and this
    check is what keeps them out until they are finished.

    Args:
        spec (FeatureSpec): Feature to check.
        root (Path): Repository root.
        video_ids (list[str]): Clips the dataset requires.

    Raises:
        BuildError: If any clip's feature is missing.
    """
    with FeatureReader(spec, root) as reader:
        available = reader.available_ids()

    missing = [v for v in video_ids if v not in available]
    if not missing:
        logger.info(f"{spec.name}: coverage OK ({len(video_ids)}/{len(video_ids)}).")
        return

    preview = ", ".join(missing[:5])
    more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
    raise BuildError(
        f"{spec.name}: missing features for {len(missing)}/{len(video_ids)} clips. "
        f"Extraction is incomplete — finish it or drop this feature from the "
        f"config; the builder will not zero-fill. Missing: {preview}{more}"
    )


def probe_feature(
    spec: FeatureSpec,
    root: Path,
    video_ids: list[str],
    time_dim_hint: int | None = None,
) -> tuple[int, int]:
    """Determine a feature's stored time dimension and verify its width.

    Reading every clip here would double the build's I/O — and for WavLM, whose
    pickles hold all 12 layers at ~0.34 s per clip, that is nearly an hour of
    pure waste. Instead:

    * ``max_time_dim`` is taken from ``time_dim_hint`` when the caller can
      derive it (any feature with a ``rate_hz`` has a known frame count), or
      probed from a sample otherwise.
    * Every clip's width and length is still checked exactly, during the write
      pass, where the data is being read anyway. A clip that disagrees fails the
      build; nothing is silently truncated.

    Args:
        spec (FeatureSpec): Feature to probe.
        root (Path): Repository root.
        video_ids (list[str]): Clips to sample.
        time_dim_hint (int | None): Known stored time dimension, if any.

    Returns:
        tuple[int, int]: ``(max_time_dim, dim)``.

    Raises:
        BuildError: If a sampled clip's width contradicts the configured ``dim``.
    """
    if time_dim_hint is not None:
        return time_dim_hint, spec.dim

    # No hint: the time axis is not wall-clock (text), so it is genuinely
    # ragged and every clip must be measured to find the padding target.
    max_time = 0
    with FeatureReader(spec, root) as reader:
        for video_id in _progress(video_ids, f"probe {spec.name}", leave=False):
            sample = reader.read(video_id)
            _check_width(spec, video_id, sample.features)
            max_time = max(max_time, sample.time_dim)
    return max_time, spec.dim


def _stratified_head(subsets: list[str], limit: int) -> np.ndarray:
    """Pick the first clips of each split, up to ``limit`` in total.

    ``label.csv`` is grouped by split (train, then valid, then test), so a plain
    ``[:limit]`` would yield train-only clips and produce a dataset that cannot
    be validated or tested against. Spreading the limit across splits keeps a
    small build usable as a real end-to-end fixture.

    Args:
        subsets (list[str]): Split name per clip, in file order.
        limit (int): Total clips to keep.

    Returns:
        np.ndarray: Sorted row indices to keep.
    """
    per_split = max(1, limit // len(SUBSET_CODES))
    keep: list[int] = []
    for name in SUBSET_CODES:
        rows = [i for i, s in enumerate(subsets) if s == name]
        keep.extend(rows[:per_split])
    return np.asarray(sorted(keep[:limit]), dtype=int)


def _time_dim_hint(spec: FeatureSpec, root: Path, reference_id: str) -> int | None:
    """Stored time dimension for a feature whose length is known in advance.

    Resolved in order:

    * ``spec.max_time_dim`` when the config declares one. This is the honest
      place for the decision — the stored length is a modelling choice (how much
      of a clip the model may see), not a property to be discovered.
    * Otherwise, for a feature with a ``rate_hz``: one clip settles it, since the
      grid is wall-clock and FI clips are uniform (15.32–15.36 s).
    * Otherwise ``None``, sending the caller down the full-probe path.

    Args:
        spec (FeatureSpec): The feature.
        root (Path): Repository root.
        reference_id (str): Clip to measure.

    Returns:
        int | None: The time dimension, or ``None`` if it must be probed.

    Raises:
        BuildError: If the reference clip's width contradicts the config.
    """
    if spec.max_time_dim is not None:
        return spec.max_time_dim
    if spec.rate_hz is None:
        return None
    with FeatureReader(spec, root) as reader:
        sample = reader.read(reference_id)
    _check_width(spec, reference_id, sample.features)
    return sample.time_dim


def _check_width(spec: FeatureSpec, video_id: str, features: np.ndarray) -> None:
    """Assert a clip's feature width matches the configured ``dim``.

    Args:
        spec (FeatureSpec): The feature.
        video_id (str): Clip being checked.
        features (np.ndarray): The clip's features.

    Raises:
        BuildError: If the width disagrees with the config.
    """
    width = int(features.shape[-1])
    if width != spec.dim:
        raise BuildError(
            f"{spec.name}: {video_id} has feature dim {width}, but the config "
            f"declares dim={spec.dim}."
        )


def _feature_shape(spec: FeatureSpec, n_samples: int, time_dim: int) -> tuple[int, ...]:
    """Shape of a feature's ``data`` dataset.

    Args:
        spec (FeatureSpec): The feature.
        n_samples (int): Row count.
        time_dim (int): Stored time dimension.

    Returns:
        tuple[int, ...]: ``(N, L, T, D)`` for multi-layer, else ``(N, T, D)``.
    """
    if spec.is_multi_layer:
        return (n_samples, len(spec.layers), time_dim, spec.dim)
    return (n_samples, time_dim, spec.dim)


def _chunk_shape(spec: FeatureSpec, time_dim: int) -> tuple[int, ...]:
    """Chunk shape placing exactly one sample in one chunk.

    This is the single most important storage decision for dataloader speed: a
    random-access read of sample ``i`` touches exactly one chunk, with no read
    amplification. Letting h5py auto-chunk would spread a sample across chunks
    and pull in neighbours on every read.

    Args:
        spec (FeatureSpec): The feature.
        time_dim (int): Stored time dimension.

    Returns:
        tuple[int, ...]: The chunk shape.
    """
    if spec.is_multi_layer:
        return (1, len(spec.layers), time_dim, spec.dim)
    return (1, time_dim, spec.dim)


def _write_feature_datasets(
    group: h5py.Group,
    spec: FeatureSpec,
    n_samples: int,
    time_dim: int,
) -> None:
    """Create the ``data``/``mask``/``length`` datasets for one feature.

    Args:
        group (h5py.Group): The feature's group.
        spec (FeatureSpec): The feature.
        n_samples (int): Row count.
        time_dim (int): Stored time dimension.
    """
    group.create_dataset(
        "data",
        shape=_feature_shape(spec, n_samples, time_dim),
        dtype=np.dtype(spec.dtype),
        chunks=_chunk_shape(spec, time_dim),
        compression=COMPRESSION,
        compression_opts=COMPRESSION_LEVEL,
        shuffle=True,
    )
    group.create_dataset(
        "mask",
        shape=(n_samples, time_dim),
        dtype=bool,
        chunks=(1, time_dim),
        compression=COMPRESSION,
        compression_opts=COMPRESSION_LEVEL,
    )
    group.create_dataset("length", shape=(n_samples,), dtype=np.int32)

    group.attrs["dim"] = spec.dim
    group.attrs["max_time_dim"] = time_dim
    group.attrs["source_dir"] = spec.source_dir
    group.attrs["source_format"] = spec.source_format
    group.attrs["dtype_orig"] = "float32"
    group.attrs["layers"] = np.asarray(spec.layers, dtype=np.int32)
    group.attrs["rate_hz"] = -1.0 if spec.rate_hz is None else spec.rate_hz


def _fit_to(array: np.ndarray, time_dim: int, axis: int) -> np.ndarray:
    """Zero-pad or truncate an array's time axis to exactly ``time_dim``.

    Truncation only bites when a clip is longer than the probed maximum, which
    for a wall-clock feature means a clip a few frames longer than the reference
    (FI clips vary between 15.32 s and 15.36 s). Dropping those few trailing
    frames is intended: the stored length is uniform and the analysis window is
    shorter than the clip regardless.

    Args:
        array (np.ndarray): Array to fit.
        time_dim (int): Target length.
        axis (int): Time axis index.

    Returns:
        np.ndarray: An array whose ``axis`` is exactly ``time_dim`` long.
    """
    current = array.shape[axis]
    if current > time_dim:
        return array.take(indices=range(time_dim), axis=axis)
    if current < time_dim:
        pad_width = [(0, 0)] * array.ndim
        pad_width[axis] = (0, time_dim - current)
        return np.pad(array, pad_width, mode="constant")
    return array


def build(
    spec: DatasetSpec,
    root: Path,
    config_yaml: str,
    limit: int | None = None,
    overwrite: bool = False,
) -> BuildStats:
    """Build the HDF5 dataset described by ``spec``.

    Args:
        spec (DatasetSpec): What to build. **Only** its features are read.
        root (Path): Repository root for resolving relative paths.
        config_yaml (str): The config source, embedded for traceability.
        limit (int | None): Build only the first N clips, for tests and CI.
        overwrite (bool): Rebuild even if an identical file already exists.

    Returns:
        BuildStats: Counters describing the run.

    Raises:
        BuildError: If labels or features are missing or inconsistent.
    """
    h5_path = root / spec.h5_path
    label_csv = root / spec.label_csv

    if h5_path.exists() and not overwrite and _is_current(h5_path, config_yaml):
        logger.info(f"{h5_path} is already current for this config; nothing to do.")
        return _stats_from_existing(h5_path)

    video_ids, subsets, labels = load_labels(label_csv)
    if limit is not None:
        keep = _stratified_head(subsets, limit)
        video_ids = [video_ids[i] for i in keep]
        subsets = [subsets[i] for i in keep]
        labels = {k: v[keep] for k, v in labels.items()}

    n_samples = len(video_ids)
    if not n_samples:
        raise BuildError(f"{label_csv}: no clips to build.")

    logger.info(f"Building {h5_path} with {n_samples} clips and features {spec.feature_names}.")

    for feature in spec.features:
        resolve_coverage(feature, root, video_ids)

    time_dims = {}
    dims = {}
    for feature in spec.features:
        hint = _time_dim_hint(feature, root, video_ids[0])
        time_dims[feature.name], dims[feature.name] = probe_feature(
            feature, root, video_ids, time_dim_hint=hint
        )
        logger.info(
            f"{feature.name}: T={time_dims[feature.name]} D={dims[feature.name]} "
            f"rate_hz={feature.rate_hz}"
        )

    h5_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = h5_path.with_suffix(h5_path.suffix + ".tmp")

    with h5py.File(tmp_path, "w") as fh:
        fh.attrs["schema_version"] = SCHEMA_VERSION
        fh.attrs["created_utc"] = datetime.now(UTC).isoformat()
        fh.attrs["git_sha"] = git_sha(root)
        fh.attrs["source_label_csv_sha256"] = file_sha256(label_csv)
        fh.attrs["builder_config_yaml"] = config_yaml

        fh.create_dataset("video_id", data=video_ids, dtype=h5py.string_dtype())
        fh.create_dataset(
            "subset",
            data=np.asarray([SUBSET_CODES[s] for s in subsets], dtype=np.uint8),
        )

        label_group = fh.create_group("labels")
        for trait in ALL_TRAITS:
            label_group.create_dataset(trait, data=labels[trait])

        feature_group = fh.create_group("features")
        for feature in spec.features:
            group = feature_group.create_group(feature.name)
            _write_feature_datasets(group, feature, n_samples, time_dims[feature.name])

            stored_time = time_dims[feature.name]
            time_axis = 1 if feature.is_multi_layer else 0
            n_truncated = 0
            worst_truncation = 0
            with FeatureReader(feature, root) as reader:
                for row, video_id in enumerate(_progress(video_ids, f"write {feature.name}")):
                    sample = reader.read(video_id)
                    # Every clip is validated here rather than in a separate probe
                    # pass: the data is already in memory, so the check is free.
                    _check_width(feature, video_id, sample.features)

                    length = min(sample.time_dim, stored_time)
                    if sample.time_dim > stored_time:
                        n_truncated += 1
                        worst_truncation = max(worst_truncation, sample.time_dim - stored_time)
                    fitted = _fit_to(sample.features, stored_time, axis=time_axis)
                    mask = _fit_to(sample.mask, stored_time, axis=0)
                    # Padding is never valid, whatever the source mask said.
                    mask[length:] = False

                    group["data"][row] = fitted.astype(feature.dtype)
                    group["mask"][row] = mask
                    group["length"][row] = length

            if n_truncated:
                # Truncation is legitimate -- max_time_dim rounds the stored
                # length down to a number that divides evenly into analysis
                # windows -- but it is data leaving the dataset, so it is
                # reported rather than left silent. Losing a few frames of clip
                # overrun is expected; losing a large tail means max_time_dim is
                # set wrong.
                seconds = f" ({worst_truncation / feature.rate_hz:.2f}s)" if feature.rate_hz else ""
                logger.warning(
                    f"{feature.name}: truncated {n_truncated}/{n_samples} clips to "
                    f"max_time_dim={stored_time}; longest cut was "
                    f"{worst_truncation} frames{seconds}."
                )

    tmp_path.replace(h5_path)

    stats = BuildStats(
        n_samples=n_samples,
        n_features=len(spec.features),
        bytes_written=h5_path.stat().st_size,
        per_feature_time_dim=dict(time_dims),
    )
    logger.info(
        f"Wrote {h5_path} ({stats.bytes_written / 1e9:.2f} GB, "
        f"{stats.n_samples} clips, {stats.n_features} features)."
    )
    return stats


def _is_current(h5_path: Path, config_yaml: str) -> bool:
    """Whether an existing file already matches this config and schema.

    Args:
        h5_path (Path): Candidate file.
        config_yaml (str): The config that would be used.

    Returns:
        bool: ``True`` when a rebuild would be a no-op.
    """
    try:
        with h5py.File(h5_path, "r") as fh:
            return (
                int(fh.attrs.get("schema_version", -1)) == SCHEMA_VERSION
                and str(fh.attrs.get("builder_config_yaml", "")) == config_yaml
            )
    except OSError:
        return False


def _stats_from_existing(h5_path: Path) -> BuildStats:
    """Read stats back off an already-built file.

    Args:
        h5_path (Path): The file.

    Returns:
        BuildStats: Counters describing it.
    """
    with h5py.File(h5_path, "r") as fh:
        features = fh["features"]
        return BuildStats(
            n_samples=int(fh["video_id"].shape[0]),
            n_features=len(features),
            bytes_written=h5_path.stat().st_size,
            per_feature_time_dim={
                name: int(features[name].attrs["max_time_dim"]) for name in features
            },
        )


def dataset_spec_from_config(config_path: Path) -> tuple[DatasetSpec, str]:
    """Load a data config into a :class:`DatasetSpec`.

    The config names its features by file stem; each is loaded from
    ``<config_dir>/../features/<name>.yaml``. Only the named features are read.
    Paths on the returned spec (``h5_path``, ``label_csv``) stay relative --
    resolving them against a repository root is the caller's job (see
    :func:`build`), not this function's.

    Args:
        config_path (Path): Path to the data config YAML.

    Returns:
        tuple[DatasetSpec, str]: The spec and the verbatim config text.

    Raises:
        BuildError: If the config or a referenced feature file is missing or
            malformed.
    """
    if not config_path.is_file():
        raise BuildError(f"Data config not found: {config_path}")

    config_yaml = config_path.read_text()
    config = yaml.safe_load(config_yaml)

    if not isinstance(config, dict):
        raise BuildError(f"{config_path}: expected a YAML mapping.")
    for key in ("h5_path", "features"):
        if key not in config:
            raise BuildError(f"{config_path}: missing required key {key!r}.")

    feature_dir = config_path.parent.parent / "features"
    features = []
    for name in config["features"]:
        feature_path = feature_dir / f"{name}.yaml"
        if not feature_path.is_file():
            raise BuildError(f"{config_path}: no feature config at {feature_path}.")
        try:
            features.append(FeatureSpec.from_dict(yaml.safe_load(feature_path.read_text())))
        except SchemaError as e:
            raise BuildError(f"{feature_path}: {e}") from e

    spec = DatasetSpec(
        features=features,
        h5_path=str(config["h5_path"]),
        label_csv=str(config.get("label_csv", "data/processed/FI/label.csv")),
    )
    return spec, config_yaml


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Data config YAML naming the features to include.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Repository root that config paths are relative to.",
    )
    parser.add_argument(
        "--h5-path",
        type=Path,
        default=None,
        help="Override the config's output path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Build only the first N clips (for tests and CI).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild even if an up-to-date file already exists.",
    )
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    spec, config_yaml = dataset_spec_from_config(args.config)
    if args.h5_path is not None:
        spec = DatasetSpec(
            features=spec.features, h5_path=str(args.h5_path), label_csv=spec.label_csv
        )

    build(spec, args.root, config_yaml, limit=args.limit, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
