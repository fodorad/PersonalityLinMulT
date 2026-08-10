"""Tests for the ONNX-only REST server (``deploy/serve.py``).

Lightweight: ``deploy.serve._load`` is monkeypatched to return a fake
in-process model, so every request goes through FastAPI's real routing/
validation stack (``fastapi.testclient.TestClient``) without a real HF Hub
download or an ONNX Runtime session -- this is what would have caught the
missing ``python-multipart`` runtime dependency this session found only via
a manual ``docker run``, since a route that fails to register at import time
fails here too.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deploy import serve  # noqa: E402

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "emotional_stability"]


class FakeModel:
    """A stand-in for PersonalityModelONNX: fixed feature name and traits."""

    feature_names = ["wavlm"]
    traits = TRAITS

    def predict(self, features):
        del features
        return dict.fromkeys(TRAITS, 0.5)

    def predict_from_audio(self, path):
        del path
        return dict.fromkeys(TRAITS, 0.5)

    def predict_from_media(self, path, transcript=None):
        del path, transcript
        return dict.fromkeys(TRAITS, 0.5)


class ServeTestCase(unittest.TestCase):
    def setUp(self):
        self._orig_model = serve._model
        serve._model = FakeModel()
        self.client = TestClient(serve.app)

    def tearDown(self):
        serve._model = self._orig_model


class TestPing(ServeTestCase):
    def test_returns_ok(self):
        response = self.client.get("/ping")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})


class TestIndex(ServeTestCase):
    def test_landing_page_mentions_the_model_id_and_endpoints(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(serve.MODEL_ID, response.text)
        self.assertIn("/predict_audio", response.text)
        self.assertIn("/predict_media", response.text)
        self.assertIn("/invocations", response.text)


class TestInvocations(ServeTestCase):
    def test_returns_one_prediction_per_sample(self):
        payload = {
            "inputs": [
                {"wavlm": np.random.randn(10, 8).astype("float32").tolist()},
                {"wavlm": np.random.randn(5, 8).astype("float32").tolist()},
            ]
        }
        response = self.client.post("/invocations", json=payload)
        self.assertEqual(response.status_code, 200)
        predictions = response.json()["predictions"]
        self.assertEqual(len(predictions), 2)
        for prediction in predictions:
            self.assertEqual(set(prediction), set(TRAITS))

    def test_rejects_a_malformed_request_body(self):
        response = self.client.post("/invocations", json={"wrong_key": []})
        self.assertEqual(response.status_code, 422)


class TestUploadSizeLimit(ServeTestCase):
    def setUp(self):
        super().setUp()
        self._orig_max = serve.MAX_UPLOAD_BYTES
        serve.MAX_UPLOAD_BYTES = 10  # small, so the test doesn't send real MBs

    def tearDown(self):
        serve.MAX_UPLOAD_BYTES = self._orig_max
        super().tearDown()

    def test_predict_audio_rejects_an_oversized_upload(self):
        response = self.client.post(
            "/predict_audio",
            files={"file": ("clip.wav", b"x" * 100, "audio/wav")},
        )
        self.assertEqual(response.status_code, 413)

    def test_predict_media_rejects_an_oversized_upload(self):
        response = self.client.post(
            "/predict_media",
            files={"file": ("clip.mp4", b"x" * 100, "video/mp4")},
        )
        self.assertEqual(response.status_code, 413)

    def test_an_upload_within_the_limit_still_succeeds(self):
        serve.MAX_UPLOAD_BYTES = 1000
        response = self.client.post(
            "/predict_audio",
            files={"file": ("clip.wav", b"x" * 10, "audio/wav")},
        )
        self.assertEqual(response.status_code, 200)


class TestPredictAudio(ServeTestCase):
    def test_returns_predictions_for_an_uploaded_file(self):
        response = self.client.post(
            "/predict_audio",
            files={"file": ("clip.wav", b"fake-audio-bytes", "audio/wav")},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json()), set(TRAITS))

    def test_a_decode_failure_surfaces_as_400_not_500_with_a_generic_message(self):
        # The raw exception (which can embed this server's own tempfile
        # path, e.g. "ffmpeg: /tmp/tmpXXXXXX.wav: invalid data") must never
        # reach the client -- only a fixed, generic message. See
        # deploy/serve.py's module docstring.
        class BoomingModel(FakeModel):
            def predict_from_audio(self, path):
                raise RuntimeError(f"ffmpeg: {path}: invalid data found")

        serve._model = BoomingModel()
        response = self.client.post(
            "/predict_audio",
            files={"file": ("clip.wav", b"not-actually-audio", "audio/wav")},
        )
        self.assertEqual(response.status_code, 400)
        detail = response.json()["detail"]
        self.assertNotIn("ffmpeg", detail)
        self.assertNotIn("tmp", detail)
        self.assertEqual(detail, "failed to decode or process the uploaded file")


class TestPredictMedia(ServeTestCase):
    def test_returns_predictions_for_an_uploaded_file(self):
        response = self.client.post(
            "/predict_media",
            files={"file": ("clip.mp4", b"fake-video-bytes", "video/mp4")},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json()), set(TRAITS))

    def test_accepts_an_optional_transcript_field(self):
        response = self.client.post(
            "/predict_media",
            files={"file": ("clip.mp4", b"fake-video-bytes", "video/mp4")},
            data={"transcript": "hello world"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json()), set(TRAITS))

    def test_a_decode_failure_surfaces_as_400_not_500_with_a_generic_message(self):
        class BoomingModel(FakeModel):
            def predict_from_media(self, path, transcript=None):
                raise RuntimeError(f"ffmpeg: {path}: invalid data found")

        serve._model = BoomingModel()
        response = self.client.post(
            "/predict_media",
            files={"file": ("clip.mp4", b"not-actually-video", "video/mp4")},
        )
        self.assertEqual(response.status_code, 400)
        detail = response.json()["detail"]
        self.assertNotIn("ffmpeg", detail)
        self.assertNotIn("tmp", detail)
        self.assertEqual(detail, "failed to decode or process the uploaded file")


if __name__ == "__main__":
    unittest.main()
