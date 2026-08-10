"""Tests for DINOv2 ONNX export.

Downloads and traces the real DINOv2 base model, so this only runs when
explicitly requested (RUN_HEAVY_ONNX_TESTS=1) -- same rationale as
test_onnx_export_wavlm.py: fast, offline by default, with a real end-to-end
check available on demand.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

RUN_HEAVY = os.environ.get("RUN_HEAVY_ONNX_TESTS") == "1"


class TestExportDinov2ToOnnxContract(unittest.TestCase):
    def test_export_function_exists_and_is_callable(self):
        from personalitylinmult.inference.onnx_export import export_dinov2_to_onnx

        self.assertTrue(callable(export_dinov2_to_onnx))


@unittest.skipUnless(RUN_HEAVY, "set RUN_HEAVY_ONNX_TESTS=1 to run (downloads DINOv2 base)")
class TestExportDinov2ToOnnxReal(unittest.TestCase):
    def test_export_matches_dinov2_wrapper_output(self):
        import numpy as np
        import onnxruntime as ort
        import torch

        from personalitylinmult.inference.onnx_export import export_dinov2_to_onnx

        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = export_dinov2_to_onnx(Path(tmp) / "dinov2.onnx")
            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

            from exordium.video.deep.dinov2 import DINOv2Wrapper

            wrapper = DINOv2Wrapper(model_name="base", device_id=-1)

            torch.manual_seed(0)
            frame = torch.randint(0, 256, (224, 224, 3), dtype=torch.uint8).numpy()

            preprocessed = wrapper.preprocess(frame)  # (1, 3, 224, 224), wrapper's own contract
            (onnx_out,) = session.run(["cls_embedding"], {"pixel_values": preprocessed.numpy()})

            wrapper_out = wrapper.inference(preprocessed).detach().numpy()

            np.testing.assert_allclose(onnx_out, wrapper_out, atol=1e-3, rtol=1e-3)

    def test_output_is_l2_normalized(self):
        """A raw-forward-only export that omitted the CLS-token selection or
        the L2-normalize step would still run without error, just produce
        wrong features -- verify the norm explicitly, not just shape."""
        import numpy as np
        import onnxruntime as ort
        import torch

        from personalitylinmult.inference.onnx_export import export_dinov2_to_onnx

        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = export_dinov2_to_onnx(Path(tmp) / "dinov2.onnx")
            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

            torch.manual_seed(1)
            dummy = torch.randn(1, 3, 224, 224).numpy()
            (out,) = session.run(["cls_embedding"], {"pixel_values": dummy})

            np.testing.assert_allclose(np.linalg.norm(out, axis=-1), 1.0, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
