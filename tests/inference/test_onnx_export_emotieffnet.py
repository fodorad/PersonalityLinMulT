"""Tests for EmotiEffNet ONNX export.

Downloads and traces the real EmotiEffNet (timm EfficientNet-B0) checkpoint,
so this only runs when explicitly requested (RUN_HEAVY_ONNX_TESTS=1) -- same
rationale as test_onnx_export_wavlm.py.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

RUN_HEAVY = os.environ.get("RUN_HEAVY_ONNX_TESTS") == "1"


class TestExportEmotiEffNetToOnnxContract(unittest.TestCase):
    def test_export_function_exists_and_is_callable(self):
        from personalitylinmult.inference.onnx_export import export_emotieffnet_to_onnx

        self.assertTrue(callable(export_emotieffnet_to_onnx))


@unittest.skipUnless(RUN_HEAVY, "set RUN_HEAVY_ONNX_TESTS=1 to run (downloads EmotiEffNet B0)")
class TestExportEmotiEffNetToOnnxReal(unittest.TestCase):
    def test_export_matches_emotieffnet_wrapper_output(self):
        import numpy as np
        import onnxruntime as ort
        import torch

        from personalitylinmult.inference.onnx_export import export_emotieffnet_to_onnx

        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = export_emotieffnet_to_onnx(Path(tmp) / "emotieffnet.onnx")
            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

            from exordium.video.deep.emotieffnet import EmotiEffNetWrapper

            wrapper = EmotiEffNetWrapper(model_name="enet_b0_8_best_vgaf", device_id=-1)

            torch.manual_seed(0)
            frame = torch.randint(0, 256, (224, 224, 3), dtype=torch.uint8).numpy()

            preprocessed = wrapper.preprocess(frame)  # (1, 3, 224, 224), wrapper's own contract
            (onnx_out,) = session.run(["features"], {"pixel_values": preprocessed.numpy()})

            wrapper_out = wrapper.inference(preprocessed).detach().numpy()

            np.testing.assert_allclose(onnx_out, wrapper_out, atol=1e-3, rtol=1e-3)
            self.assertEqual(onnx_out.shape, (1, 1280))


if __name__ == "__main__":
    unittest.main()
