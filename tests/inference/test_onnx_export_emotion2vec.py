"""Tests for emotion2vec+ ONNX export.

Downloads and traces the real emotion2vec+ (fully custom, vendored
architecture) checkpoint, so this only runs when explicitly requested
(RUN_HEAVY_ONNX_TESTS=1) -- same rationale as test_onnx_export_wavlm.py.

The highest-risk extractor in this project's ONNX export effort: a bespoke
transformer with ALiBi relative positional bias, not a thin wrapper over a
library model. Two concrete risks are checked explicitly, not assumed away
by a successful export:

1. The traced graph must match the real wrapper's output at the length it
   was traced at (the standard numeric-parity check every export gets).
2. The traced graph must ALSO match at a genuinely DIFFERENT length -- ALiBi's
   position bias is recomputed from the input's time dimension via
   torch.arange on every call, which is exactly the kind of shape-dependent
   internal computation a legacy tracer can silently bake in as a constant
   instead of keeping dynamic. Re-testing only at the tracing length would
   not catch that failure mode; a second, different length is what does.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

RUN_HEAVY = os.environ.get("RUN_HEAVY_ONNX_TESTS") == "1"


class TestExportEmotion2vecToOnnxContract(unittest.TestCase):
    def test_export_function_exists_and_is_callable(self):
        from personalitylinmult.inference.onnx_export import export_emotion2vec_to_onnx

        self.assertTrue(callable(export_emotion2vec_to_onnx))


@unittest.skipUnless(RUN_HEAVY, "set RUN_HEAVY_ONNX_TESTS=1 to run (downloads emotion2vec+)")
class TestExportEmotion2vecToOnnxReal(unittest.TestCase):
    def test_export_matches_wrapper_at_the_traced_length(self):
        import numpy as np
        import onnxruntime as ort
        import torch

        from personalitylinmult.inference.onnx_export import export_emotion2vec_to_onnx

        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = export_emotion2vec_to_onnx(Path(tmp) / "emotion2vec.onnx")
            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

            from exordium.audio.emotion2vec import Emotion2vecWrapper

            wrapper = Emotion2vecWrapper(device_id=-1)

            torch.manual_seed(0)
            waveform = torch.randn(1, 16000)  # 1s @ 16kHz -- the tracing length

            (onnx_out,) = session.run(["features"], {"source": waveform.numpy()})
            wrapper_out = wrapper(waveform.squeeze(0)).unsqueeze(0).numpy()

            np.testing.assert_allclose(onnx_out, wrapper_out, atol=1e-3, rtol=1e-3)

    def test_export_matches_wrapper_at_a_different_length(self):
        """The critical check: ALiBi's position bias is recomputed from the
        input's time dimension via torch.arange -- a shape-dependent
        computation that could get silently baked in as a constant sized to
        the tracing length. Re-testing at the SAME length the graph was
        traced at (1s) would not catch that; this uses a genuinely
        different length (2.5s) instead."""
        import numpy as np
        import onnxruntime as ort
        import torch

        from personalitylinmult.inference.onnx_export import export_emotion2vec_to_onnx

        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = export_emotion2vec_to_onnx(Path(tmp) / "emotion2vec.onnx")
            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

            from exordium.audio.emotion2vec import Emotion2vecWrapper

            wrapper = Emotion2vecWrapper(device_id=-1)

            torch.manual_seed(1)
            waveform = torch.randn(1, 40000)  # 2.5s @ 16kHz -- NOT the tracing length

            (onnx_out,) = session.run(["features"], {"source": waveform.numpy()})
            wrapper_out = wrapper(waveform.squeeze(0)).unsqueeze(0).numpy()

            self.assertEqual(onnx_out.shape, wrapper_out.shape)
            np.testing.assert_allclose(onnx_out, wrapper_out, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
