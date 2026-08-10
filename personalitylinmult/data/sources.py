"""Readers for each on-disk feature format produced by ``preprocess/fi``.

Every reader returns the same thing — a :class:`FeatureSample` of
``features`` plus a ``mask`` — so the builder can treat WavLM pickles, DINOv2
safetensors and RoBERTa HDF5 datasets identically.

**Masks.** Only the track-based extractors (EmotiEffNet, FaRL, MARLIN) record
a real mask, because the protagonist is not detected on every frame. The other
formats are dense by construction and get an all-``True`` mask synthesised here.
Producing a mask unconditionally is what lets the rest of the pipeline hold a
single contract: every sequence feature has a mask, ``True`` = valid.

**AppleDouble files.** This repository lives on a non-native filesystem where
macOS writes ``._{name}`` resource-fork sidecars next to real files. A naive
``glob("*.st")`` returns twice as many paths as there are clips, and reading one
raises ``SafetensorError: header too large``. :func:`iter_feature_files` filters
them; use it rather than globbing directly.
"""

from __future__ import annotations

import pickle  # noqa: S403
from dataclasses import dataclass
from typing import TYPE_CHECKING

import h5py
import numpy as np
from safetensors.numpy import load_file

if TYPE_CHECKING:
    from pathlib import Path

    from personalitylinmult.data.schema import FeatureSpec


@dataclass(frozen=True)
class FeatureSample:
    """One clip's features for one modality.

    Args:
        features (np.ndarray): ``(T, D)`` for a sequence, or ``(L, T, D)`` for a
            multi-layer sequence.
        mask (np.ndarray): ``(T,)`` bool, ``True`` where the timestep is valid.
    """

    features: np.ndarray
    mask: np.ndarray

    @property
    def time_dim(self) -> int:
        """Length of the time axis.

        Returns:
            int: ``T``.
        """
        return int(self.mask.shape[0]) if self.mask.size else 0


class SourceError(RuntimeError):
    """Raised when a feature file is missing, unreadable, or malformed."""


def is_hidden_sidecar(path: Path) -> bool:
    """Whether a path is a macOS AppleDouble resource fork.

    Args:
        path (Path): Candidate path.

    Returns:
        bool: ``True`` for ``._*`` sidecars, which are not real feature files.
    """
    return path.name.startswith("._")


def iter_feature_files(directory: Path, suffix: str) -> list[Path]:
    """List real feature files in a directory, skipping AppleDouble sidecars.

    Args:
        directory (Path): Directory to scan.
        suffix (str): File suffix including the dot, e.g. ``".st"``.

    Returns:
        list[Path]: Sorted real files.

    Raises:
        SourceError: If the directory does not exist.
    """
    if not directory.is_dir():
        raise SourceError(f"Feature directory does not exist: {directory}")
    return sorted(p for p in directory.glob(f"*{suffix}") if not is_hidden_sidecar(p))


def _dense_mask(time_dim: int) -> np.ndarray:
    """Build an all-valid mask for a dense feature.

    Args:
        time_dim (int): Number of timesteps.

    Returns:
        np.ndarray: ``(T,)`` bool array of ``True``.
    """
    return np.ones((time_dim,), dtype=bool)


def read_pickle_layers(path: Path, layers: tuple[int, ...]) -> FeatureSample:
    """Read selected transformer layers from a pickled list of per-layer arrays.

    Used for WavLM. ``layers`` always names *model* layers (9 means WavLM's
    layer 9), never positions in the file, so the same config reads both
    supported on-disk formats:

    * ``{"layers": [...], "features": [...]}`` — what
      :mod:`personalitylinmult.preprocess.fi.acoustic_wavlm` writes now: only
      the extracted layers, each tagged with its model layer id.
    * a bare ``list`` of every layer — the older, fatter format, where position
      happens to equal the layer id.

    The distinction matters because the current extractor stores one layer per
    clip: in a bare list that array would sit at index 0 and be indistinguishable
    from layer 0. Carrying the ids makes the file self-describing instead.

    The pickles are produced locally by :mod:`personalitylinmult.preprocess.fi`
    from the operator's own audio, so they are trusted input.

    Args:
        path (Path): Path to the ``.pkl`` file.
        layers (tuple[int, ...]): Model layer ids to keep, in order.

    Returns:
        FeatureSample: ``(T, D)`` when one layer is selected, ``(L, T, D)``
        when several. The mask is dense.

    Raises:
        SourceError: If the file is unreadable or a requested layer is absent.
    """
    try:
        with path.open("rb") as fh:
            raw = pickle.load(fh)  # noqa: S301 -- locally produced by preprocess/fi
    except Exception as e:
        raise SourceError(f"{path}: could not unpickle: {e}") from e

    if isinstance(raw, dict):
        available = list(raw.get("layers", ()))
        arrays = list(raw.get("features", ()))
        if not arrays or len(arrays) != len(available):
            raise SourceError(
                f"{path}: expected matching 'layers' and 'features' entries, got "
                f"{len(available)} ids and {len(arrays)} arrays."
            )
    elif isinstance(raw, list | tuple) and raw:
        # Legacy: every layer present, so position == model layer id.
        arrays = list(raw)
        available = list(range(len(arrays)))
    else:
        raise SourceError(f"{path}: expected a non-empty list of per-layer arrays.")

    index_of = {layer: i for i, layer in enumerate(available)}
    missing = [i for i in layers if i not in index_of]
    if missing:
        raise SourceError(
            f"{path}: layer(s) {missing} not in this file, which holds layers "
            f"{available}. Re-extract with --layers to include them: "
            f"make acoustic-wavlm-fi"
        )

    selected = [np.asarray(arrays[index_of[i]]) for i in layers]
    stacked = selected[0] if len(selected) == 1 else np.stack(selected, axis=0)
    time_dim = selected[0].shape[0]
    return FeatureSample(features=stacked, mask=_dense_mask(time_dim))


def read_npy(path: Path) -> FeatureSample:
    """Read a plain ``(T, D)`` array.

    Args:
        path (Path): Path to the ``.npy`` file.

    Returns:
        FeatureSample: The array with a dense mask.

    Raises:
        SourceError: If the file is unreadable or not 2-D.
    """
    try:
        arr = np.load(path)
    except Exception as e:
        raise SourceError(f"{path}: could not load: {e}") from e
    if arr.ndim != 2:
        raise SourceError(f"{path}: expected a 2-D (T, D) array, got shape {arr.shape}.")
    return FeatureSample(features=arr, mask=_dense_mask(arr.shape[0]))


def read_safetensors(path: Path) -> FeatureSample:
    """Read a safetensors feature file, honouring its mask when present.

    DINOv2 runs on full frames and emits no ``mask`` key; the track-based
    extractors emit one. Both are normalised to a mask here.

    Args:
        path (Path): Path to the ``.st`` file.

    Returns:
        FeatureSample: ``(T, D)`` features and a ``(T,)`` mask.

    Raises:
        SourceError: If the file is unreadable, lacks ``features``, or its mask
            length disagrees with its feature length.
    """
    try:
        tensors = load_file(str(path))
    except Exception as e:
        raise SourceError(f"{path}: could not load safetensors: {e}") from e

    if "features" not in tensors:
        raise SourceError(f"{path}: safetensors file has no 'features' key.")

    features = tensors["features"]
    if features.ndim != 2:
        raise SourceError(f"{path}: expected 2-D features, got shape {features.shape}.")

    if "mask" in tensors:
        mask = tensors["mask"].astype(bool)
        if mask.shape[0] != features.shape[0]:
            raise SourceError(
                f"{path}: mask length {mask.shape[0]} does not match feature "
                f"length {features.shape[0]}."
            )
    else:
        mask = _dense_mask(features.shape[0])

    return FeatureSample(features=features, mask=mask)


def read_hdf5_entry(h5_file: h5py.File, video_id: str) -> FeatureSample:
    """Read one clip's ``(T, D)`` sequence entry from a per-id HDF5 feature file.

    Args:
        h5_file (h5py.File): Open feature file, keyed by ``video_id``.
        video_id (str): Clip identifier.

    Returns:
        FeatureSample: The entry, with a dense all-valid mask.

    Raises:
        SourceError: If the id is absent or the entry is not a 2-D ``(T, D)``
            sequence.
    """
    if video_id not in h5_file:
        raise SourceError(f"{h5_file.filename}: no entry for video_id {video_id!r}.")

    arr = np.asarray(h5_file[video_id])
    if arr.ndim != 2:
        raise SourceError(
            f"{h5_file.filename}:{video_id}: expected a 2-D (T, D) sequence, got shape {arr.shape}."
        )
    return FeatureSample(features=arr, mask=_dense_mask(arr.shape[0]))


class FeatureReader:
    """Reads one feature for arbitrary clips, holding open handles as needed.

    ``hdf5_per_id`` features live in a single file keyed by ``video_id``, so the
    handle is opened once and reused; the per-clip formats are opened on demand.
    Use as a context manager so the handle is always released.

    Args:
        spec (FeatureSpec): The feature to read.
        root (Path): Repository root that ``spec.source_dir`` is relative to.
    """

    def __init__(self, spec: FeatureSpec, root: Path):
        self.spec = spec
        self.root = root
        self.directory = root / spec.source_dir
        self._h5: h5py.File | None = None

    def __enter__(self) -> FeatureReader:
        """Open any shared handle.

        Returns:
            FeatureReader: This reader.
        """
        if self.spec.source_format == "hdf5_per_id":
            path = self.directory / str(self.spec.source_file)
            if not path.is_file():
                raise SourceError(f"{self.spec.name}: feature file not found: {path}")
            self._h5 = h5py.File(path, "r")
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Release any shared handle."""
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def available_ids(self) -> set[str]:
        """Clip ids this feature actually has on disk.

        The builder compares this against ``label.csv`` and fails loudly on a
        shortfall, rather than silently zero-filling clips whose features were
        never extracted.

        Returns:
            set[str]: Available ``video_id`` values.

        Raises:
            SourceError: If the source directory or file is missing.
        """
        if self.spec.source_format == "hdf5_per_id":
            path = self.directory / str(self.spec.source_file)
            if not path.is_file():
                raise SourceError(f"{self.spec.name}: feature file not found: {path}")
            if self._h5 is not None:
                return set(self._h5.keys())
            with h5py.File(path, "r") as fh:
                return set(fh.keys())

        suffix = {"pickle_layers": ".pkl", "npy": ".npy", "safetensors": ".st"}[
            self.spec.source_format
        ]
        return {p.stem for p in iter_feature_files(self.directory, suffix)}

    def read(self, video_id: str) -> FeatureSample:
        """Read one clip's feature.

        Args:
            video_id (str): Clip identifier.

        Returns:
            FeatureSample: Features plus mask.

        Raises:
            SourceError: If the clip's feature cannot be read.
        """
        fmt = self.spec.source_format

        if fmt == "hdf5_per_id":
            if self._h5 is None:
                raise SourceError(
                    f"{self.spec.name}: reader used outside its context manager; "
                    "the HDF5 handle is not open."
                )
            return read_hdf5_entry(self._h5, video_id)

        if fmt == "pickle_layers":
            return read_pickle_layers(self.directory / f"{video_id}.pkl", self.spec.layers)
        if fmt == "npy":
            return read_npy(self.directory / f"{video_id}.npy")
        if fmt == "safetensors":
            return read_safetensors(self.directory / f"{video_id}.st")

        raise SourceError(f"{self.spec.name}: unhandled source_format {fmt!r}.")
