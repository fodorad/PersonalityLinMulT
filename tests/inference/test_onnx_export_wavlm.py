"""Tests for WavLM ONNX export.

Downloads and traces the real ~380 MB WavLM model, so this only runs when
explicitly requested (RUN_HEAVY_ONNX_TESTS=1) -- same rationale as
test_audio_features.py skipping the real exordium extractors: fast, offline
by default, with a real end-to-end check available on demand.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

RUN_HEAVY = os.environ.get("RUN_HEAVY_ONNX_TESTS") == "1"


class TestExportWavlmToOnnxContract(unittest.TestCase):
    def test_export_function_exists_and_is_callable(self):
        from personalitylinmult.inference.onnx_export import export_wavlm_to_onnx

        self.assertTrue(callable(export_wavlm_to_onnx))

    def test_wavlm_layer_constant_matches_the_live_extractor(self):
        from personalitylinmult.inference.audio_features import WAVLM_LAYER
        from personalitylinmult.inference.onnx_export import WAVLM_LAYER as ONNX_WAVLM_LAYER

        self.assertEqual(ONNX_WAVLM_LAYER, WAVLM_LAYER)


@unittest.skipUnless(RUN_HEAVY, "set RUN_HEAVY_ONNX_TESTS=1 to run (downloads ~380 MB WavLM)")
class TestExportWavlmToOnnxReal(unittest.TestCase):
    def test_export_matches_wavlm_wrapper_layer_output(self):
        import numpy as np
        import onnxruntime as ort
        import torch

        from personalitylinmult.inference.onnx_export import export_wavlm_to_onnx

        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = export_wavlm_to_onnx(Path(tmp) / "wavlm.onnx")
            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

            torch.manual_seed(0)
            waveform = torch.randn(1, 16000).numpy()
            (onnx_out,) = session.run(["hidden_state"], {"input_values": waveform})

            from exordium.audio.wavlm import WavlmWrapper

            wrapper = WavlmWrapper(device_id=-1, model_name="base+")
            wrapper_out = wrapper(waveform)[9].numpy()

            np.testing.assert_allclose(onnx_out[0], wrapper_out[0], atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
