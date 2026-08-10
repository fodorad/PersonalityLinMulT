"""Random-access reads from the built FI HDF5 file.

This dataset touches the HDF5 file and nothing else — no feature directories,
no label CSV, no pickles. That is what makes a training run reproducible from a
single published artifact, and what keeps ``__getitem__`` down to one chunk
read per feature.

**Analysis window.** The file stores whole clips; the window is applied here, on
read, using each feature's ``rate_hz``. Changing ``window_seconds`` from 15 to
10 to 5 is a config edit, not a rebuild, and the crop is pushed into the HDF5
read itself so windowed-away data never crosses the bus.

**Worker safety.** h5py handles do not survive ``fork``. The handle is therefore
opened lazily inside whichever process first calls ``__getitem__``, and stripped
from the pickled state so ``spawn``-based workers construct their own.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from personalitylinmult.data.schema import (
    CROP_MODES,
    SCHEMA_VERSION,
    SUBSET_CODES,
    SchemaError,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)
"""Module-level logger."""

NO_RATE_SENTINEL = -1.0
"""``rate_hz`` attribute value meaning "no wall-clock time axis"."""


class DatasetError(RuntimeError):
    """Raised when the HDF5 file cannot serve the requested configuration."""


class FeatureView:
    """Resolved read plan for one feature within one dataset.

    Args:
        name (str): Feature name.
        dim (int): Feature width.
        stored_time_dim (int): Time dimension as stored in the file.
        window_time_dim (int): Time dimension after windowing.
        rate_hz (float | None): Native rate, or ``None`` if not wall-clock.
        n_layers (int): Layer count, ``0`` when there is no layer axis.
    """

    def __init__(
        self,
        name: str,
        dim: int,
        stored_time_dim: int,
        window_time_dim: int,
        rate_hz: float | None,
        n_layers: int,
    ):
        self.name = name
        self.dim = dim
        self.stored_time_dim = stored_time_dim
        self.window_time_dim = window_time_dim
        self.rate_hz = rate_hz
        self.n_layers = n_layers

    @property
    def is_multi_layer(self) -> bool:
        """Whether reads carry a layer axis.

        Returns:
            bool: ``True`` when the model should see ``(B, L, T, F)``.
        """
        return self.n_layers > 0

    @property
    def is_windowed(self) -> bool:
        """Whether the analysis window crops this feature.

        Returns:
            bool: ``True`` when the stored clip is longer than the window.
        """
        return self.window_time_dim < self.stored_time_dim


class FiHdf5Dataset(Dataset):
    """Reads FI samples for one split out of the built HDF5 file.

    Args:
        h5_path (str | Path): The built dataset.
        subset (str): One of ``"train"``, ``"valid"``, ``"test"``.
        feature_names (Sequence[str]): Features to return, in model input order.
        target_names (Sequence[str]): Label columns to return, in head order.
        window_seconds (float | None): Analysis window. ``None`` reads whole
            clips.
        crop_mode (str): ``"head"``, ``"center"``, or ``"random"``. ``"random"``
            is honoured only on the training split; evaluation always uses
            ``"center"`` so it stays deterministic.
        seed (int): Base seed for ``"random"`` cropping.

    Raises:
        DatasetError: If the file is missing, has the wrong schema version, or
            lacks a requested feature or target.
        SchemaError: If ``subset`` or ``crop_mode`` is not recognised.
    """

    def __init__(
        self,
        h5_path: str | Path,
        subset: str,
        feature_names: Sequence[str],
        target_names: Sequence[str],
        window_seconds: float | None = None,
        crop_mode: str = "head",
        seed: int = 42,
    ):
        if subset not in SUBSET_CODES:
            raise SchemaError(f"Unknown subset {subset!r}; expected {sorted(SUBSET_CODES)}.")
        if crop_mode not in CROP_MODES:
            raise SchemaError(f"Unknown crop_mode {crop_mode!r}; expected {sorted(CROP_MODES)}.")

        self.h5_path = Path(h5_path)
        self.subset = subset
        self.feature_names = list(feature_names)
        self.target_names = list(target_names)
        self.window_seconds = window_seconds
        self.crop_mode = crop_mode
        self.seed = seed
        self._h5: h5py.File | None = None

        if not self.h5_path.is_file():
            raise DatasetError(f"Dataset not found: {self.h5_path}. Build it first: make hdf5-fi")

        # Read the index and metadata once, then close: the handle must not be
        # inherited across a fork.
        with h5py.File(self.h5_path, "r") as fh:
            self._check_schema(fh)
            self.views = [self._resolve_view(fh, name) for name in self.feature_names]
            subsets = fh["subset"][:]
            self.row_indices = np.flatnonzero(subsets == SUBSET_CODES[subset]).astype(np.int64)
            self.video_ids = [fh["video_id"][int(i)].decode() for i in self.row_indices]
            self.targets = self._read_targets(fh)

        logger.info(
            f"{self.subset}: {len(self.row_indices)} clips, "
            f"features={self.feature_names}, targets={self.target_names}, "
            f"window={self.window_seconds}s crop={self.crop_mode}"
        )

    def _check_schema(self, fh: h5py.File) -> None:
        """Verify the file was written by a compatible builder.

        Args:
            fh (h5py.File): Open dataset.

        Raises:
            DatasetError: On a schema-version mismatch.
        """
        version = int(fh.attrs.get("schema_version", -1))
        if version != SCHEMA_VERSION:
            raise DatasetError(
                f"{self.h5_path}: schema_version {version} but this code expects "
                f"{SCHEMA_VERSION}. Rebuild the dataset: make hdf5-fi"
            )

    def _resolve_view(self, fh: h5py.File, name: str) -> FeatureView:
        """Build the read plan for one feature.

        Args:
            fh (h5py.File): Open dataset.
            name (str): Feature name.

        Returns:
            FeatureView: The resolved plan.

        Raises:
            DatasetError: If the feature is absent or the window exceeds the
                stored clip.
        """
        if name not in fh["features"]:
            available = sorted(fh["features"].keys())
            raise DatasetError(
                f"{self.h5_path} has no feature {name!r}. It contains {available}. "
                "Either fix the config or rebuild with this feature included."
            )

        group = fh[f"features/{name}"]
        attrs = group.attrs
        rate = float(attrs["rate_hz"])
        rate_hz = None if rate == NO_RATE_SENTINEL else rate
        stored_time = int(attrs["max_time_dim"])
        n_layers = int(len(attrs["layers"])) if len(attrs["layers"]) > 1 else 0

        window_time = stored_time
        if rate_hz is not None and self.window_seconds is not None:
            window_time = max(1, round(rate_hz * self.window_seconds))
            if window_time > stored_time:
                raise DatasetError(
                    f"{name}: a {self.window_seconds}s window needs {window_time} "
                    f"frames at {rate_hz} Hz, but the file stores only {stored_time} "
                    f"({stored_time / rate_hz:.2f}s). Shorten window_seconds."
                )

        return FeatureView(
            name=name,
            dim=int(attrs["dim"]),
            stored_time_dim=stored_time,
            window_time_dim=window_time,
            rate_hz=rate_hz,
            n_layers=n_layers,
        )

    def _read_targets(self, fh: h5py.File) -> np.ndarray:
        """Read the label matrix for this split.

        Args:
            fh (h5py.File): Open dataset.

        Returns:
            np.ndarray: ``(n_rows, n_targets)`` float32.

        Raises:
            DatasetError: If a requested target is absent.
        """
        missing = [t for t in self.target_names if t not in fh["labels"]]
        if missing:
            raise DatasetError(
                f"{self.h5_path}: no label(s) {missing}. Available: {sorted(fh['labels'].keys())}."
            )
        columns = [fh[f"labels/{t}"][:][self.row_indices] for t in self.target_names]
        return np.stack(columns, axis=1).astype(np.float32)

    @property
    def handle(self) -> h5py.File:
        """The process-local HDF5 handle, opened on first use.

        Opening lazily is what makes the dataset fork-safe: each worker creates
        its own handle rather than inheriting a parent's, which HDF5 does not
        support and which corrupts reads.

        Returns:
            h5py.File: The open file.
        """
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", libver="latest", swmr=True)
        return self._h5

    def __getstate__(self) -> dict[str, Any]:
        """Drop the file handle so the dataset can be pickled to a worker.

        Returns:
            dict: Picklable state.
        """
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __len__(self) -> int:
        """Number of clips in this split.

        Returns:
            int: Clip count.
        """
        return len(self.row_indices)

    def _crop_fraction(self, index: int) -> float:
        """How far into a clip's slack the analysis window starts.

        Returned as a fraction of the available slack rather than a frame count,
        because the features of one clip have different frame rates and so
        different slacks. Drawing per feature — even from the same seed — would
        pick a different integer out of each feature's own range and start the
        audio and video windows at different wall-clock times. Drawing once per
        clip in a rate-free unit is what keeps the modalities aligned.

        Args:
            index (int): Row within this split.

        Returns:
            float: Offset into the slack, in ``[0, 1]``.
        """
        mode = self.crop_mode
        if mode == "random" and self.subset != "train":
            # Evaluation must be deterministic; only train sees random offsets.
            mode = "center"

        if mode == "head":
            return 0.0
        if mode == "center":
            return 0.5

        # Seeded by the clip's row, not by `index`, so a clip's offset does not
        # depend on how the split happens to be ordered.
        rng = np.random.default_rng(self.seed + int(self.row_indices[index]))
        return float(rng.random())

    def _crop_start(self, view: FeatureView, fraction: float) -> int:
        """First frame of the analysis window for one feature.

        Args:
            view (FeatureView): The feature's read plan.
            fraction (float): Offset into the slack, from :meth:`_crop_fraction`.

        Returns:
            int: Start frame.
        """
        slack = view.stored_time_dim - view.window_time_dim
        if slack <= 0:
            return 0
        return min(slack, int(round(fraction * slack)))

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Read one clip.

        Args:
            index (int): Row within this split.

        Returns:
            dict: ``{name}`` and ``{name}_mask`` per feature, plus ``targets``
            and ``video_id``. Every feature is paired with a mask, ``True`` =
            valid.
        """
        row = int(self.row_indices[index])
        fh = self.handle
        item: dict[str, Any] = {"video_id": self.video_ids[index]}
        # Drawn once, outside the loop: every feature of this clip shares it.
        fraction = self._crop_fraction(index)

        for view in self.views:
            group = fh[f"features/{view.name}"]
            start = self._crop_start(view, fraction)
            stop = start + view.window_time_dim

            # Slice inside the HDF5 read: cropped-away frames are never read.
            if view.is_multi_layer:
                data = group["data"][row, :, start:stop, :]
            else:
                data = group["data"][row, start:stop, :]
            mask = group["mask"][row, start:stop]

            features = torch.from_numpy(np.ascontiguousarray(data)).float()
            item[view.name] = features
            item[f"{view.name}_mask"] = torch.from_numpy(np.ascontiguousarray(mask))

        item["targets"] = torch.from_numpy(self.targets[index])
        return item
