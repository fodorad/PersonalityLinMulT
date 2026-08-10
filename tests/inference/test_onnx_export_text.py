"""Tests for the mmBERT and XLM-RoBERTa (mpnet) ONNX exports.

Downloads and traces the real HF models, so this only runs when explicitly
requested (RUN_HEAVY_ONNX_TESTS=1) -- same rationale as
test_onnx_export_wavlm.py: fast, offline by default, with a real end-to-end
check available on demand.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

RUN_HEAVY = os.environ.get("RUN_HEAVY_ONNX_TESTS") == "1"


class TestExportTextEncodersToOnnxContract(unittest.TestCase):
    def test_export_functions_exist_and_are_callable(self):
        from personalitylinmult.inference.onnx_export import (
            export_mmbert_to_onnx,
            export_xlm_roberta_to_onnx,
        )

        self.assertTrue(callable(export_mmbert_to_onnx))
        self.assertTrue(callable(export_xlm_roberta_to_onnx))

    def test_xlm_roberta_model_id_is_the_mpnet_backbone(self):
        """This project's fi.h5 was built with XlmRobertaWrapper()'s default
        backbone ("mpnet"), confirmed by reading
        personalitylinmult/preprocess/fi/textual_xlm_roberta.py directly --
        not the wrapper's other supported backbone ("e5")."""
        from personalitylinmult.inference.onnx_export import XLM_ROBERTA_MODEL_ID

        self.assertEqual(
            XLM_ROBERTA_MODEL_ID, "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )


@unittest.skipUnless(
    RUN_HEAVY, "set RUN_HEAVY_ONNX_TESTS=1 to run (downloads mmBERT + XLM-RoBERTa)"
)
class TestExportMmbertToOnnxReal(unittest.TestCase):
    def test_export_matches_mmbert_wrapper_output(self):
        import numpy as np
        import onnxruntime as ort

        from personalitylinmult.inference.onnx_export import export_mmbert_to_onnx

        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = export_mmbert_to_onnx(Path(tmp) / "mmbert.onnx")
            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

            from exordium.text.mmbert import MmbertWrapper

            wrapper = MmbertWrapper(device_id=-1)
            text = "This is a short test sentence."
            tokens = wrapper.tokenizer(text, return_tensors="pt")

            onnx_inputs = {
                "input_ids": tokens["input_ids"].numpy(),
                "attention_mask": tokens["attention_mask"].numpy(),
            }
            (onnx_out,) = session.run(["last_hidden_state"], onnx_inputs)

            wrapper_out = wrapper.predict(text, pooling="none")

            np.testing.assert_allclose(onnx_out, wrapper_out, atol=1e-3, rtol=1e-3)


@unittest.skipUnless(
    RUN_HEAVY, "set RUN_HEAVY_ONNX_TESTS=1 to run (downloads mmBERT + XLM-RoBERTa)"
)
class TestExportXlmRobertaToOnnxReal(unittest.TestCase):
    def test_export_matches_xlm_roberta_wrapper_output(self):
        import numpy as np
        import onnxruntime as ort

        from personalitylinmult.inference.onnx_export import export_xlm_roberta_to_onnx

        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = export_xlm_roberta_to_onnx(Path(tmp) / "xlm_roberta.onnx")
            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

            from exordium.text.xlm_roberta import XlmRobertaWrapper

            wrapper = XlmRobertaWrapper(device_id=-1)  # default backbone: mpnet
            text = "This is a short test sentence."
            tokens = wrapper.tokenizer(text, return_tensors="pt")

            onnx_inputs = {
                "input_ids": tokens["input_ids"].numpy(),
                "attention_mask": tokens["attention_mask"].numpy(),
            }
            (onnx_out,) = session.run(["last_hidden_state"], onnx_inputs)

            wrapper_out = wrapper.predict(text, pooling="none")

            np.testing.assert_allclose(onnx_out, wrapper_out, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
