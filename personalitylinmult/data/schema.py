"""Typed contract shared by the HDF5 builder and the training dataloader.

This module is deliberately dependency-light (no torch, no h5py) so that the
same :class:`FeatureSpec` objects can describe a feature on the way *into* the
HDF5 file and on the way *out* of it. One schema, no drift.

**Layout of the built file** (see :mod:`personalitylinmult.data.builder`)::

    /                        attrs: schema_version, created_utc, git_sha,
                                    source_label_csv_sha256, builder_config_yaml
    /video_id                (N,)     variable-length utf-8, row order == index order
    /subset                  (N,)     uint8, see SUBSET_CODES
    /labels/<trait>          (N,)     float32
    /features/<name>/data    (N,T,D)  float16, chunks=(1,T,D)
    /features/<name>/mask    (N,T)    bool,    chunks=(1,T)
    /features/<name>/length  (N,)     int32, the clip's length before padding
        attrs on /features/<name>: dim, max_time_dim, source_dir, source_format,
                                   dtype_orig, layers, rate_hz

**Padding is post-padding.** A clip's real frames occupy ``[0, length)`` and the
zeros fill ``[length, max_time_dim)`` -- never the other way round. So
``data[i, :length[i]]`` is the clip, ``mask[i]`` is ``True`` exactly on
``[0, length[i])``, and a longer clip is truncated from the tail. Anything
reading this file may rely on that.

Masks are written for *every* sequence feature, including ones that are dense
for every sample today. The network contract is feature + mask per feature, and
a uniform contract survives windowing (where density is a property of the source
clip, not of the cropped batch) at a storage cost of roughly 0.05%.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = 1
"""Bumped whenever the on-disk HDF5 layout changes incompatibly."""

BIG_FIVE_TRAITS: tuple[str, ...] = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "emotional_stability",
)
"""Targets of the big-five regression task.

``emotional_stability`` (= ``1 - neuroticism``) is used rather than
``neuroticism`` so that all five traits point the same way: higher is more of
the named quality. :class:`~personalitylinmult.train.config.TrainConfig`
rejects ``neuroticism`` as a big-five target.
"""

INTERVIEW_TRAITS: tuple[str, ...] = ("interview",)
"""Target of the interview-variable regression task.

A distinct task from big-five regression: the interview variable is a
hireability judgement, not a personality trait, and is trained separately.
"""

ALL_TRAITS: tuple[str, ...] = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
    "emotional_stability",
    "interview",
)
"""Every label column stored in the HDF5.

All seven are stored regardless of task, because storing a scalar per clip is
free and it means a task change never requires a rebuild. ``neuroticism`` is
retained as the raw annotation for traceability even though training uses
``emotional_stability``.
"""

SUBSETS: tuple[str, ...] = ("train", "valid", "test")
"""Dataset splits, in code order."""

SUBSET_CODES: dict[str, int] = {name: code for code, name in enumerate(SUBSETS)}
"""Maps a split name to the ``uint8`` code stored in ``/subset``."""

SOURCE_FORMATS: frozenset[str] = frozenset({"pickle_layers", "npy", "safetensors", "hdf5_per_id"})
"""Recognised on-disk feature formats.

* ``pickle_layers`` — a pickled ``list`` of per-layer ``(T, D)`` arrays (WavLM).
* ``npy`` — a single ``(T, D)`` array (emotion2vec+).
* ``safetensors`` — ``features``/``frame_ids`` and optionally ``mask`` (DINOv2,
  EmotiEffNet, FaRL).
* ``hdf5_per_id`` — one HDF5 holding a ``(T, D)`` sequence dataset per
  ``video_id`` (the token-level text features).
"""

CROP_MODES: frozenset[str] = frozenset({"head", "center", "random"})
"""Analysis-window crop strategies.

``head`` takes ``[0, window)``; ``center`` takes the middle window; ``random``
takes a uniformly random offset and is only applied to the training split
(``center`` is substituted elsewhere) so evaluation stays deterministic.
"""


class SchemaError(ValueError):
    """Raised when a feature or dataset configuration is self-inconsistent."""


@dataclass(frozen=True)
class FeatureSpec:
    """Describes one feature: where it comes from and how it is stored.

    The same object drives extraction (builder) and consumption (dataset), so a
    feature can never be written under one interpretation and read under
    another.

    Args:
        name (str): Feature name. Becomes the HDF5 group name and the key the
            model sees in ``names=``, so it must be unique within a dataset.
        source_dir (str): Directory of per-clip files, or the containing
            directory for ``hdf5_per_id``. Relative to the repository root.
        source_format (str): One of :data:`SOURCE_FORMATS`.
        dim (int): Feature dimensionality ``D``.
        rate_hz (float | None): Nominal sampling rate, used to convert
            ``window_seconds`` into a frame count. ``None`` means the time axis
            is not wall-clock (e.g. text tokens) and the feature is never
            cropped by the analysis window.
        max_time_dim (int | None): Stored time dimension. Clips longer than this
            are truncated and shorter ones are zero-padded (with the mask marking
            the padding invalid), so the stored length is uniform. ``None`` means
            "measure it from the data", which is only appropriate when the length
            is not known in advance.
        layers (tuple[int, ...]): For ``pickle_layers`` only: which transformer
            layers to keep. One layer stores ``(N, T, D)``; several store
            ``(N, L, T, D)`` for linmult's learned ``weighted_sum``.
        source_file (str | None): For ``hdf5_per_id``: the HDF5 filename inside
            ``source_dir``.
        dtype (str): On-disk dtype. ``float16`` halves size and I/O; measured
            across the real features the largest magnitude is ~31 against fp16's
            65504 ceiling, so the range is safe.

    Raises:
        SchemaError: If the spec is internally inconsistent.
    """

    name: str
    source_dir: str
    source_format: str
    dim: int
    rate_hz: float | None = None
    max_time_dim: int | None = None
    layers: tuple[int, ...] = ()
    source_file: str | None = None
    dtype: str = "float16"

    def __post_init__(self) -> None:
        """Validate the spec's internal consistency.

        Raises:
            SchemaError: If any field combination is invalid.
        """
        if not self.name:
            raise SchemaError("FeatureSpec.name must be a non-empty string.")
        if self.source_format not in SOURCE_FORMATS:
            raise SchemaError(
                f"{self.name}: unknown source_format {self.source_format!r}; "
                f"expected one of {sorted(SOURCE_FORMATS)}."
            )
        if self.dim <= 0:
            raise SchemaError(f"{self.name}: dim must be positive, got {self.dim}.")
        if self.rate_hz is not None and self.rate_hz <= 0:
            raise SchemaError(f"{self.name}: rate_hz must be positive, got {self.rate_hz}.")
        if self.layers and self.source_format != "pickle_layers":
            raise SchemaError(
                f"{self.name}: 'layers' is only meaningful for source_format="
                f"'pickle_layers', not {self.source_format!r}."
            )
        if self.source_format == "pickle_layers" and not self.layers:
            raise SchemaError(
                f"{self.name}: source_format='pickle_layers' requires an explicit "
                "'layers' list. Storing every layer is a deliberate choice, not a default."
            )
        if self.source_format == "hdf5_per_id" and not self.source_file:
            raise SchemaError(f"{self.name}: source_format='hdf5_per_id' requires 'source_file'.")
        if self.max_time_dim is not None and self.max_time_dim < 1:
            raise SchemaError(
                f"{self.name}: max_time_dim must be positive, got {self.max_time_dim}."
            )

    @property
    def is_multi_layer(self) -> bool:
        """Whether this feature stores several transformer layers per sample.

        Returns:
            bool: ``True`` when the stored array carries a layer axis, which
            linmult consumes as a 4-D ``(B, L, T, F)`` weighted-sum input.
        """
        return len(self.layers) > 1

    def window_frames(self, window_seconds: float | None) -> int | None:
        """Frames spanned by an analysis window for this feature.

        Args:
            window_seconds (float | None): Window length in seconds, or ``None``
                for no windowing.

        Returns:
            int | None: Frame count, or ``None`` when this feature is not
            windowed (no ``rate_hz``, or no window requested).
        """
        if window_seconds is None or self.rate_hz is None:
            return None
        return max(1, round(self.rate_hz * window_seconds))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FeatureSpec:
        """Build a spec from a plain dict, e.g. parsed YAML.

        Args:
            d (dict): Feature configuration values.

        Returns:
            FeatureSpec: The validated spec.

        Raises:
            SchemaError: If required keys are missing or the spec is invalid.
        """
        known = {
            "name",
            "source_dir",
            "source_format",
            "dim",
            "rate_hz",
            "max_time_dim",
            "layers",
            "source_file",
            "dtype",
        }
        unknown = set(d) - known
        if unknown:
            raise SchemaError(
                f"{d.get('name', '<unnamed>')}: unknown feature config keys "
                f"{sorted(unknown)}. Known keys: {sorted(known)}."
            )
        missing = {"name", "source_dir", "source_format", "dim"} - set(d)
        if missing:
            raise SchemaError(f"Feature config is missing required keys {sorted(missing)}.")
        return cls(
            name=str(d["name"]),
            source_dir=str(d["source_dir"]),
            source_format=str(d["source_format"]),
            dim=int(d["dim"]),
            rate_hz=None if d.get("rate_hz") is None else float(d["rate_hz"]),
            max_time_dim=(None if d.get("max_time_dim") is None else int(d["max_time_dim"])),
            layers=tuple(int(x) for x in d.get("layers", ()) or ()),
            source_file=d.get("source_file"),
            dtype=str(d.get("dtype", "float16")),
        )


@dataclass(frozen=True)
class DatasetSpec:
    """The full description of one built HDF5 dataset.

    Args:
        features (list[FeatureSpec]): Features to include. **Only these are
            read from disk and written to the file** — the selection is the
            contract, not a filter applied later.
        h5_path (str): Destination path of the built file.
        label_csv (str): Source ``label.csv`` produced by the preprocess stage.

    Raises:
        SchemaError: If no features are given or names collide.
    """

    features: list[FeatureSpec]
    h5_path: str
    label_csv: str

    def __post_init__(self) -> None:
        """Validate the dataset spec.

        Raises:
            SchemaError: If the feature list is empty or contains duplicates.
        """
        if not self.features:
            raise SchemaError("A dataset needs at least one feature.")
        names = [f.name for f in self.features]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise SchemaError(f"Duplicate feature names: {duplicates}.")

    @property
    def feature_names(self) -> list[str]:
        """Feature names in configured order.

        Returns:
            list[str]: Names, in the order the model receives its inputs.
        """
        return [f.name for f in self.features]

    def feature(self, name: str) -> FeatureSpec:
        """Look up a feature by name.

        Args:
            name (str): Feature name.

        Returns:
            FeatureSpec: The matching spec.

        Raises:
            KeyError: If no feature has that name.
        """
        for spec in self.features:
            if spec.name == name:
                return spec
        raise KeyError(f"No feature named {name!r}. Available: {self.feature_names}.")


@dataclass
class BuildStats:
    """Counters describing one builder run, for logging and tests.

    Args:
        n_samples (int): Rows written.
        n_features (int): Features written.
        bytes_written (int): Size of the resulting file.
        per_feature_time_dim (dict[str, int]): Stored ``T`` per feature.
    """

    n_samples: int = 0
    n_features: int = 0
    bytes_written: int = 0
    per_feature_time_dim: dict[str, int] = field(default_factory=dict)
