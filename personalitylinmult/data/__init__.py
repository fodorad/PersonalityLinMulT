"""Training-data layer: the FI HDF5 artifact, its builder, and its loaders.

This package owns everything between the feature extractors in
:mod:`personalitylinmult.preprocess.fi` and the training loop in
:mod:`personalitylinmult.train`:

* :mod:`~personalitylinmult.data.schema` — the typed feature/dataset contract.
* :mod:`~personalitylinmult.data.sources` — readers for each on-disk format.
* :mod:`~personalitylinmult.data.builder` — the config-driven HDF5 writer.
* :mod:`~personalitylinmult.data.dataset` — random-access reads from that HDF5.
* :mod:`~personalitylinmult.data.collate` — batching with per-feature masks.
* :mod:`~personalitylinmult.data.datamodule` — the Lightning ``DataModule``.

The HDF5 file is the single training artifact and the reproducibility boundary:
training reads it and nothing else. It is published to a public Hugging Face
dataset repo (``make push-fi-h5``) so a run reproduces without re-extraction, and
rebuilt from source by ``make pipeline-fi``.
"""
