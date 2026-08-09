"""Tests for FaRL ONNX export.

Downloads and traces the real FaRL (CLIP ViT-B/16, OpenAI-to-HF weight
conversion) checkpoint, so this only runs when explicitly requested
(RUN_HEAVY_ONNX_TESTS=1) -- same rationale as test_onnx_export_wavlm.py.

The weight-conversion correctness risk (FarlWrapper._convert_openai_to_hf)
is exactly what the numeric-parity test below catches: a structurally
successful but numerically wrong export is the dangerous failure mode here,
not an export-time crash.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

RUN_HEAVY = os.environ.get("RUN_HEAVY_ONNX_TESTS") == "1"


class TestExportFarlToOnnxContract(unittest.TestCase):
    def test_export_function_exists_and_is_callable(self):
        from personalitylinmult.inference.onnx_export import export_farl_to_onnx

        self.assertTrue(callable(export_farl_to_onnx))


@unittest.skipUnless(RUN_HEAVY, "set RUN_HEAVY_ONNX_TESTS=1 to run (downloads FaRL ep64)")
class TestExportFarlToOnnxReal(unittest.TestCase):
    def test_export_matches_farl_wrapper_output(self):
        import numpy as np
        import onnxruntime as ort
        import torch

        from personalitylinmult.inference.onnx_export import export_farl_to_onnx

        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = export_farl_to_onnx(Path(tmp) / "farl.onnx")
            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

            from exordium.video.deep.farl import FarlWrapper

            wrapper = FarlWrapper(model_name="ep64", device_id=-1)

            torch.manual_seed(0)
            frame = torch.randint(0, 256, (224, 224, 3), dtype=torch.uint8).numpy()

            preprocessed = wrapper.preprocess(frame)  # (1, 3, 224, 224), wrapper's own contract
            (onnx_out,) = session.run(["embedding"], {"pixel_values": preprocessed.numpy()})

            wrapper_out = wrapper.inference(preprocessed).detach().numpy()

            np.testing.assert_allclose(onnx_out, wrapper_out, atol=1e-3, rtol=1e-3)
            self.assertEqual(onnx_out.shape, (1, 512))

    def test_output_is_l2_normalized(self):
        """A raw-forward-only export that omitted the L2-normalize step (or
        forgot the projection) would still run without error -- verify the
        norm explicitly, not just shape, same as the DINOv2 export."""
        import numpy as np
        import onnxruntime as ort
        import torch

        from personalitylinmult.inference.onnx_export import export_farl_to_onnx

        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = export_farl_to_onnx(Path(tmp) / "farl.onnx")
            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

            torch.manual_seed(1)
            dummy = torch.randn(1, 3, 224, 224).numpy()
            (out,) = session.run(["embedding"], {"pixel_values": dummy})

            np.testing.assert_allclose(np.linalg.norm(out, axis=-1), 1.0, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
