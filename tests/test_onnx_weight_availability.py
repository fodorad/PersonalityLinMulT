"""Every ONNX weight this project downloads must be reachable -- checked
without downloading.

`predict_from_media` and `predict_from_audio`'s real download/inference
code (`personalitylinmult/onnx.py`, `personalitylinmult/inference
/onnx_export.py`) only runs end-to-end under `RUN_HEAVY_ONNX_TESTS=1`,
which fetches gigabytes of real models and is intentionally not part of
the default `make test`/CI run. That leaves a real gap: nothing would
notice if a published `.onnx` file vanished from the Hub, since no test
in the fast/default suite ever names it.

These tests close that gap the cheap way, matching the pattern already
used by the `exordium` project's `tests/utils/test_weight_availability.py`:
a `list_repo_files`/`model_info` Hub API call per file (no download), for
every filename `personalitylinmult.onnx` actually asks the Hub for.
"""

from __future__ import annotations

import logging
import time
import unittest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

_HF_PROBE_RETRY_PAUSES_S = (0, 1)
"""Delays between retry attempts for a Hub probe -- see :func:`_hf_probe`."""


def _hf_probe(probe: Callable[[], bool]) -> bool:
    """Run a Hub availability probe, retrying only on transient failures.

    A probe returning ``False`` (or raising a not-found error) is a
    definitive answer, returned immediately. Any other exception is
    treated as transient (network blip, momentary 5xx) and retried per
    :data:`_HF_PROBE_RETRY_PAUSES_S`. If every attempt raises, the probe
    is reported as unreachable (``False``) rather than failing the test
    process itself -- a real removal is what should fail the test, not an
    exception escaping this helper.

    Args:
        probe: A zero-argument callable that returns ``True``/``False``.

    Returns:
        bool: The probe's result, or ``False`` if every attempt errored.
    """
    from huggingface_hub.utils import RepositoryNotFoundError

    attempts = len(_HF_PROBE_RETRY_PAUSES_S) + 1
    for attempt in range(attempts):
        try:
            return probe()
        except RepositoryNotFoundError:
            return False
        except Exception as error:  # noqa: BLE001 -- transient Hub/network failure, retry
            if attempt == attempts - 1:
                logger.warning(f"Hub probe failed after {attempts} attempts: {error}")
                return False
            time.sleep(_HF_PROBE_RETRY_PAUSES_S[attempt])
    return False


def _hf_repo_exists(repo_id: str, timeout: int = 15) -> bool:
    """Return True if a HF Hub model repo exists and is reachable.

    ``token=False`` forces an anonymous request so a stale/invalid local
    token cannot poison the check -- the repo under test is public.
    """
    from huggingface_hub import model_info

    def probe() -> bool:
        model_info(repo_id, timeout=timeout, token=False)
        return True

    return _hf_probe(probe)


def _hf_file_exists(repo_id: str, filename: str) -> bool:
    """Return True if *filename* exists in a HF Hub model repo (no download)."""
    from huggingface_hub import list_repo_files

    def probe() -> bool:
        return filename in list(list_repo_files(repo_id, token=False))

    return _hf_probe(probe)


class TestModelRepoAvailable(unittest.TestCase):
    def test_hf_model_repo_reachable(self):
        from personalitylinmult.model import HF_MODEL_REPO

        self.assertTrue(_hf_repo_exists(HF_MODEL_REPO), f"Unreachable: {HF_MODEL_REPO}")


class TestChampionWeightsAvailable(unittest.TestCase):
    """The two promoted champions' .onnx graph + .onnx.json sidecar."""

    def test_every_champion_onnx_pair_is_present(self):
        from personalitylinmult.model import HF_MODEL_REPO
        from personalitylinmult.onnx import _onnx_filenames

        champion_ids = ["wavlm_best-ccc", "avt_best-ccc"]
        missing = []
        for model_id in champion_ids:
            graph_filename, metadata_filename = _onnx_filenames(model_id)
            for filename in (graph_filename, metadata_filename):
                if not _hf_file_exists(HF_MODEL_REPO, filename):
                    missing.append(filename)
        self.assertEqual(missing, [], f"Missing from {HF_MODEL_REPO}: {missing}")


class TestExtractorWeightsAvailable(unittest.TestCase):
    """Every shared per-feature ONNX extractor graph predict_from_media needs."""

    def test_every_extractor_onnx_graph_is_present(self):
        from personalitylinmult.model import HF_MODEL_REPO
        from personalitylinmult.onnx import EXTRACTOR_ONNX_FILENAMES

        missing = [
            filename
            for filename in EXTRACTOR_ONNX_FILENAMES.values()
            if not _hf_file_exists(HF_MODEL_REPO, filename)
        ]
        self.assertEqual(missing, [], f"Missing from {HF_MODEL_REPO}: {missing}")


if __name__ == "__main__":
    unittest.main()
