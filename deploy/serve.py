"""A minimal REST server for the champion model, built for Cloud Run.

Runs the **ONNX** inference path (``PersonalityModelONNX``), not the PyTorch
one: no ``torch``/``lightning`` at runtime, a much smaller image, and
measurably faster full-pipeline inference -- ~6-8x faster end to end than the
PyTorch path once warmed up, since WavLM dominates wall-clock time and its
ONNX export is where the real speedup is.

Deliberately not ``mlflow models build-docker``: that image spawns a second,
isolated virtualenv at container startup purely to launch the model server,
which defeats any fix applied to the outer Python's site-packages, and its
nginx reverse-proxy layer produced no diagnosable log output under Cloud
Run's sandbox (the server itself started and logged fine; nothing ever
listened on the externally-reachable port). This app is a handful of lines,
loads the champion the same way ``PersonalityModelONNX.from_pretrained``
always does (HF Hub-backed, no MLflow), and binds ``0.0.0.0`` explicitly --
no proxy layer to go silently missing.

Three endpoints:

    POST /invocations
    {"inputs": [{"wavlm": [[...]]}]}
    -> {"predictions": [{"openness": 0.9, ...}]}

    POST /predict_audio  (multipart file upload)
    -> {"openness": 0.9, ...}

    POST /predict_media  (multipart file upload, optional "transcript" field)
    -> {"openness": 0.9, ...}

``/invocations`` matches the ``mlflow models serve`` convention this project
already documents, for pre-extracted features. ``/predict_audio`` runs the
audio-only pipeline (decode -> WavLM -> LinT) on a raw uploaded audio/video
file, for the ``wavlm_best-ccc`` champion. ``/predict_media`` runs the full
AVT pipeline (face detection/tracking + optional Whisper ASR + six ONNX
feature extractors + the trained LinMulT head) on a raw uploaded video file,
for the ``avt_best-ccc`` champion -- see
:meth:`personalitylinmult.onnx.PersonalityModelONNX.predict_from_media`'s
docstring for the three-way transcript contract (explicit string / omitted
-> Whisper ASR / empty string -> skip text entirely). Both file endpoints
require ``ffmpeg`` as an external binary in the image (see
``deploy/Dockerfile``); ``/predict_media`` additionally needs the YOLO face
detector + AdaFace + Whisper weights, which download lazily on first use.

Each deployed instance serves exactly one champion, selected by the
``MODEL_ID`` environment variable (see the Makefile's per-champion
``deploy-serve-cloudrun`` pattern) -- ``/predict_audio`` and
``/predict_media`` both exist on every instance, but only the one matching
the loaded champion's ``feature_names`` will actually succeed; the other
returns a clear 400, not a 500.

**Only ``wavlm_best-ccc`` is actually deployed to Cloud Run in this
project.** The image and this server support ``avt_best-ccc`` identically
(build it, run it locally with ``MODEL_ID=avt_best-ccc``, it works -- this
was verified with a real ``/predict_media`` call), but its full pipeline
(six ONNX extractor sessions + PyTorch face detection/tracking loaded at
once) genuinely needs more memory than fits this project's free-tier
budget: it OOM'd a 4 GiB Cloud Run instance (~4.1 GiB used), then an 8 GiB
instance too (~8.3 GiB used and still climbing). Cloud Run's 16-32 GiB
tiers (``--cpu=8``) would fit, but cost real money per request rather than
staying in the always-free allotment -- see the README's Deployments
section and ``SERVE_MEMORY`` in the Makefile if you want to deploy it
yourself anyway. ``avt_best-ccc`` remains fully reachable elsewhere at no
extra cost: the Hugging Face Space (ZeroGPU, free) and the plain
``pip install`` API both serve it identically to ``wavlm_best-ccc``.

A failure in either endpoint returns a fixed, generic 400 message, not the
raw exception text: ``ffmpeg``/``onnxruntime``/``exordium`` errors commonly
embed the failing file's path, and that path is this server's own
``tempfile.NamedTemporaryFile`` location -- returning it verbatim to an
unauthenticated caller (this app is deployed with
``--allow-unauthenticated``, see the README's Cloud Run section) would leak
the container's internal filesystem layout for no benefit to the caller.
The real exception is still logged server-side (``logger.exception``) for
debugging.

Run locally::

    uv run --extra serve --extra onnx uvicorn deploy.serve:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from personalitylinmult.onnx import PersonalityModelONNX

logger = logging.getLogger(__name__)
"""Module-level logger."""

MODEL_ID = os.environ.get("MODEL_ID", "wavlm_best-ccc")
"""The champion's HF Hub model id (see README.md's pretrained-models table)."""

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 200 * 1024 * 1024))
"""Upload size cap for ``/predict_audio``/``/predict_media`` (default 200 MB,
generous for a single short clip). Enforced by reading in chunks rather than
trusting the client-supplied ``Content-Length`` header, so a request that
lies about its size still can't exhaust memory on this publicly reachable
(``--allow-unauthenticated``) endpoint."""

_model: PersonalityModelONNX | None = None


async def _read_upload_within_limit(file: UploadFile) -> bytes:
    """Read an uploaded file's bytes, rejecting anything over the cap.

    Args:
        file (UploadFile): The incoming multipart file.

    Returns:
        bytes: The file's full content.

    Raises:
        HTTPException: 413, if the upload exceeds :data:`MAX_UPLOAD_BYTES`.
    """
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(1024 * 1024):
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"upload exceeds the {MAX_UPLOAD_BYTES} byte limit",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _load() -> PersonalityModelONNX:
    """Load (and cache) the champion ONNX model from HF Hub."""
    global _model
    if _model is None:
        logger.info(f"Loading champion model {MODEL_ID!r} (onnx)...")
        _model = PersonalityModelONNX.from_pretrained(MODEL_ID)
        logger.info(f"Loaded. feature_names={_model.feature_names} traits={_model.traits}")
    return _model


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Load the model at process startup, not on the first request.

    Cloud Run's health check probes immediately after the port opens;
    loading lazily on first request would make that first real request pay
    the model-load cost instead, and risks the health check itself timing
    out.
    """
    del app
    _load()
    yield


app = FastAPI(title="PersonalityLinMulT (ONNX)", lifespan=_lifespan)


class InvocationRequest(BaseModel):
    """One or more samples to score.

    Attributes:
        inputs (list[dict[str, list]]): One ``{feature_name: (T, F) nested
            list}`` mapping per sample.
    """

    inputs: list[dict[str, list]]


_LANDING_PAGE = """\
<!doctype html>
<html>
<head><title>PersonalityLinMulT (ONNX)</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 720px; margin: 3rem auto; padding: 0 1rem; }}
code, pre {{ background: #f4f4f4; padding: 0.15em 0.4em; border-radius: 4px; }}
pre {{ padding: 1em; overflow-x: auto; }}
</style>
</head>
<body>
<h1>PersonalityLinMulT (ONNX Runtime)</h1>
<p>Big Five apparent-personality prediction, served from the champion model
<code>{model_id}</code> (features: {feature_names}) via ONNX Runtime -- no
torch/lightning in this image.</p>

<h2>Endpoints</h2>
<ul>
<li><code>GET /ping</code> &mdash; health check</li>
<li><code>POST /invocations</code> &mdash; predict from pre-extracted features</li>
<li><code>POST /predict_audio</code> &mdash; predict from a raw audio/video file,
    audio-only champion (multipart upload)</li>
<li><code>POST /predict_media</code> &mdash; predict from a raw video file, full
    AVT champion (multipart upload, optional <code>transcript</code> field)</li>
<li><a href="/docs">/docs</a> &mdash; interactive API docs (Swagger UI)</li>
</ul>

<h2>Example: pre-extracted features</h2>
<pre>curl -X POST {base_url}/invocations \\
  -H "Content-Type: application/json" \\
  -d '{{"inputs": [{{"wavlm": [[0.1, 0.2, ...], ...]}}]}}'</pre>

<h2>Example: raw audio file (audio-only champion)</h2>
<pre>curl -X POST {base_url}/predict_audio -F "file=@clip.wav"</pre>

<h2>Example: raw video file (AVT champion)</h2>
<pre>curl -X POST {base_url}/predict_media -F "file=@clip.mp4" -F "transcript=hello world"</pre>
<p>Omit <code>transcript</code> to transcribe with Whisper automatically, or
pass an empty string to skip text entirely (placeholder + all-False mask).</p>

<p><code>wavlm</code> must be a <code>(T, 768)</code> nested list &mdash; WavLM
layer-9 features, extracted the same way training data was (see
<code>personalitylinmult.inference.audio_features.extract_wavlm</code> in the
<a href="https://github.com/fodorad/PersonalityLinMulT">PersonalityLinMulT</a>
package). Response: <code>{{"predictions": [{{trait: score, ...}}]}}</code>.</p>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """A minimal landing page: what this is, and how to call it."""
    model = _load()
    return _LANDING_PAGE.format(
        model_id=MODEL_ID,
        feature_names=", ".join(model.feature_names),
        base_url="https://<this-service-url>",
    )


@app.get("/ping")
def ping() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/invocations")
def invocations(request: InvocationRequest) -> dict[str, list[dict[str, float]]]:
    """Predict Big Five scores for one or more pre-extracted-feature samples.

    Args:
        request (InvocationRequest): The samples to score.

    Returns:
        dict: ``{"predictions": [{trait: score}, ...]}``, one entry per
        input sample, in request order.
    """
    model = _load()
    predictions = [
        model.predict({name: np.asarray(array, dtype="float32") for name, array in sample.items()})
        for sample in request.inputs
    ]
    return {"predictions": predictions}


@app.post("/predict_audio")
async def predict_audio(file: UploadFile) -> dict[str, float]:
    """Predict Big Five scores directly from an uploaded audio/video file.

    Runs the audio-only pipeline (decode -> WavLM -> LinT) on ONNX Runtime
    alone. Requires ``ffmpeg`` to be present in the image for decoding.

    Args:
        file (UploadFile): The audio or video file to score.

    Returns:
        dict[str, float]: ``{trait: score}``.

    Raises:
        HTTPException: If the file cannot be decoded, the loaded champion
            does not accept audio-only input, or the upload exceeds
            :data:`MAX_UPLOAD_BYTES`.
    """
    model = _load()
    content = await _read_upload_within_limit(file)
    suffix = Path(file.filename or "clip").suffix or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(content)
        tmp.flush()
        try:
            return model.predict_from_audio(tmp.name)
        except Exception as exc:  # noqa: BLE001 -- logged, not leaked (see module docstring)
            logger.exception(f"predict_audio failed for {file.filename!r}")
            raise HTTPException(
                status_code=400, detail="failed to decode or process the uploaded file"
            ) from exc


@app.post("/predict_media")
async def predict_media(
    file: UploadFile, transcript: str | None = Form(default=None)
) -> dict[str, float]:
    """Predict Big Five scores directly from an uploaded video file (AVT).

    Runs the full pipeline (face detection/tracking + optional Whisper ASR +
    six ONNX feature extractors + the trained LinMulT head) via
    :meth:`~personalitylinmult.onnx.PersonalityModelONNX.predict_from_media`.
    Requires ``ffmpeg`` to be present in the image for decoding.

    Args:
        file (UploadFile): The video file to score.
        transcript (str | None): Omit the form field entirely to transcribe
            with Whisper automatically; pass an empty string to skip text
            entirely (placeholder + all-False mask); pass any other string
            to use it directly as the transcript.

    Returns:
        dict[str, float]: ``{trait: score}``.

    Raises:
        HTTPException: If the file cannot be decoded, the loaded champion is
            not the full AVT model, or the upload exceeds
            :data:`MAX_UPLOAD_BYTES`.
    """
    model = _load()
    content = await _read_upload_within_limit(file)
    suffix = Path(file.filename or "clip").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(content)
        tmp.flush()
        try:
            return model.predict_from_media(tmp.name, transcript=transcript)
        except Exception as exc:  # noqa: BLE001 -- logged, not leaked (see module docstring)
            logger.exception(f"predict_media failed for {file.filename!r}")
            raise HTTPException(
                status_code=400, detail="failed to decode or process the uploaded file"
            ) from exc
