"""Gradio demo for the PersonalityLinMulT Big Five models.

A Hugging Face Space front end for this project's two champions:

- ``wavlm_best-ccc`` — audio-only (LinT). Upload an audio *or video* clip; a
  video's audio track is extracted automatically (mono 44.1 kHz, matching
  training). Supports both the ``pytorch`` and ``onnx`` backends below.
- ``avt_best-ccc`` — full multimodal (LinMulT): audio + video frames + face
  crops + text. Upload a video clip and, optionally, its transcript. Leave
  the transcript blank to run Whisper ASR automatically, or check "skip
  text" to predict from audio/video alone (a zero-placeholder + all-False
  mask for the text features, not a crash — see
  ``PersonalityModelONNX.predict_from_media``'s docstring). **ONNX only**:
  this champion's full media -> prediction orchestration
  (``predict_from_media``) is only implemented for the ``onnx`` backend,
  since it is a batch pipeline (face detection/tracking + Whisper ASR +
  six ONNX feature extractors + the trained LinMulT head), not a manual
  per-feature extraction the ``pytorch`` backend could replicate simply.

A **backend dropdown** picks between the two inference paths this project
supports for the audio-only champion:

- ``pytorch`` — ``PersonalityModel.from_pretrained`` (``torch``/``lightning``).
- ``onnx`` — ``PersonalityModelONNX.from_pretrained`` + ``predict_from_audio``
  (``onnxruntime`` only, no ``torch`` needed for this path).

Both download from the same public HF Hub model repo, so the Space and
``pip install personalitylinmult[onnx]`` share one loading mechanism. The
progress log reports each stage's wall-clock time, so the two backends can be
compared on the fly on real hardware.

Run locally::

    uv run --extra demo --extra onnx python app/app.py
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

try:
    # Only present when actually running on a Hugging Face Space; must be
    # imported before torch/exordium touch CUDA. Falls back to a no-op
    # decorator for local runs (`uv run --extra demo --extra onnx python
    # app/app.py`), where the real `spaces` package isn't installed.
    import spaces
except ImportError:

    class spaces:  # noqa: N801 -- mirrors the real module's `spaces.GPU` API
        """Local-only stand-in for the `spaces` package's `GPU` decorator."""

        @staticmethod
        def GPU(*args, **kwargs):  # noqa: ARG004 -- mirrors spaces.GPU's real signature
            """Return the wrapped function unchanged (no ZeroGPU locally)."""

            def decorator(fn):
                return fn

            return decorator if not (args and callable(args[0])) else args[0]


import gradio as gr

logger = logging.getLogger(__name__)


@spaces.GPU(duration=1)
def _noop() -> None:
    """Never called.

    This app is CPU-only (ONNX Runtime / small PyTorch heads), but ZeroGPU
    (the only free Gradio hardware tier) refuses to start a Space with zero
    ``@spaces.GPU``-decorated functions. Nothing in the real prediction path
    is decorated, so no GPU quota is ever spent by a visitor -- this exists
    purely to satisfy ZeroGPU's startup check.
    """


TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "emotional_stability"]

MODEL_IDS = ("wavlm_best-ccc", "avt_best-ccc")
"""HF Hub model ids for this project's two champions (see README.md's
"Pretrained models" table): ``wavlm_best-ccc`` (LinT, audio-only,
mean_one_minus_mae=0.8918, mean_ccc=0.5604, mean_std_ratio=1.0013) and
``avt_best-ccc`` (LinMulT, full multimodal, mean_one_minus_mae=0.9074,
mean_ccc=0.6901, mean_std_ratio=0.9924)."""

BACKENDS = ("pytorch", "onnx")

_models: dict[tuple[str, str], object] = {}


def _load(model_id: str, backend: str):
    """Load (and cache) one champion model for one backend.

    Args:
        model_id: One of :data:`MODEL_IDS`.
        backend: ``"pytorch"`` or ``"onnx"``.

    Returns:
        The loaded model (``PersonalityModel`` or ``PersonalityModelONNX``).
    """
    key = (model_id, backend)
    if key not in _models:
        if backend == "pytorch":
            from personalitylinmult import PersonalityModel

            _models[key] = PersonalityModel.from_pretrained(model_id)
        else:
            from personalitylinmult.onnx import PersonalityModelONNX

            _models[key] = PersonalityModelONNX.from_pretrained(model_id)
    return _models[key]


def _scores_to_plot(scores: dict[str, float]):
    """Render a Big Five bar chart with the [0, 1] value on each bar."""
    import matplotlib.pyplot as plt

    values = [scores[t] for t in TRAITS]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(range(len(TRAITS)), values, color="#4C72B0")
    ax.set_xticks(range(len(TRAITS)))
    ax.set_xticklabels([t.replace("_", "\n") for t in TRAITS])
    ax.set_ylim(0, 1)
    ax.set_ylabel("score [0, 1]")
    ax.set_title("Big Five")
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    return fig


def _predict_pytorch(model_id: str, clip_path: str, log) -> dict[str, float]:
    """Run the PyTorch path: manual extract-per-feature, then predict.

    Only the audio-only champion is supported here -- see this module's
    docstring for why ``avt_best-ccc`` is ONNX-only.

    Args:
        model_id: One of :data:`MODEL_IDS`.
        clip_path: Path to the uploaded audio or video file.
        log: Callable appending one line to the progress log.

    Returns:
        dict[str, float]: ``{trait: score}``.
    """
    from personalitylinmult.inference.audio_features import (
        AUDIO_SUFFIXES,
        AudioExtractionError,
        ensure_audio,
        extract_emotion2vec,
        extract_wavlm,
    )

    model = _load(model_id, "pytorch")
    extractors = {"wavlm": extract_wavlm, "emotion2vec": extract_emotion2vec}
    unknown = set(model.feature_names) - set(extractors)
    if unknown:
        raise gr.Error(
            f"The pytorch backend only supports the audio-only champion "
            f"(wavlm/emotion2vec); {model_id!r} needs {sorted(unknown)}. "
            "Pick the onnx backend instead."
        )

    is_video = Path(clip_path).suffix.lower() not in AUDIO_SUFFIXES
    log("Extracting audio from video (mono 44.1 kHz)..." if is_video else "Reading audio clip...")
    t0 = time.perf_counter()
    try:
        wav_path, is_temp = ensure_audio(clip_path)
    except AudioExtractionError as exc:
        raise gr.Error(str(exc)) from exc
    log(f"  decode: {time.perf_counter() - t0:.3f}s")

    features: dict = {}
    try:
        for name in model.feature_names:
            t0 = time.perf_counter()
            array = extractors[name](wav_path, device_id=-1)
            features[name] = array
            elapsed = time.perf_counter() - t0
            log(f"  {name}: {array.shape[0]} frames x {array.shape[1]} ({elapsed:.3f}s)")
    finally:
        if is_temp:
            # lgtm[py/path-injection] -- wav_path is ensure_audio's own
            # tempfile.mkstemp() output, not attacker-controlled; only
            # deleted when is_temp confirms this function created it.
            Path(wav_path).unlink(missing_ok=True)

    t0 = time.perf_counter()
    scores = model.predict(features)
    log(f"  predict: {time.perf_counter() - t0:.3f}s")
    return scores


def _predict_onnx(
    model_id: str, clip_path: str, transcript: str | None, skip_text: bool, log
) -> dict[str, float]:
    """Run the ONNX path: the full pipeline in one call.

    Dispatches on ``model_id``: the audio-only champion runs
    ``predict_from_audio`` (decode -> WavLM -> LinT); the AVT champion runs
    ``predict_from_media`` (face detection/tracking + optional Whisper ASR +
    six ONNX feature extractors + the trained LinMulT head).

    Args:
        model_id: One of :data:`MODEL_IDS`.
        clip_path: Path to the uploaded audio or video file.
        transcript: User-supplied transcript text (AVT only), or empty/``None``.
        skip_text: If ``True`` (AVT only), predict without text features
            (zero-placeholder + all-False mask) instead of running ASR.
        log: Callable appending one line to the progress log.

    Returns:
        dict[str, float]: ``{trait: score}``.
    """
    model = _load(model_id, "onnx")

    if list(model.feature_names) == ["wavlm"]:
        log("Running full ONNX pipeline (decode -> WavLM -> LinT)...")
        t0 = time.perf_counter()
        scores = model.predict_from_audio(clip_path)
        log(f"  total: {time.perf_counter() - t0:.3f}s")
        return scores

    if skip_text:
        media_transcript = ""
        log("Running full ONNX AVT pipeline (skipping text: placeholder + all-False mask)...")
    elif transcript:
        media_transcript = transcript
        log("Running full ONNX AVT pipeline (using the supplied transcript)...")
    else:
        media_transcript = None
        log("Running full ONNX AVT pipeline (no transcript given -- transcribing with Whisper)...")

    t0 = time.perf_counter()
    scores = model.predict_from_media(clip_path, transcript=media_transcript)
    log(f"  total: {time.perf_counter() - t0:.3f}s")
    return scores


def predict_clip(model_id: str, clip_path: str, backend: str, transcript: str, skip_text: bool):
    """Predict Big Five from an audio/video clip, streaming a per-stage log.

    A generator so the UI updates the log textbox as each stage finishes,
    each line annotated with its own wall-clock time -- the point being an
    on-the-fly PyTorch-vs-ONNX comparison on real hardware.

    Args:
        model_id: One of :data:`MODEL_IDS`.
        clip_path: Path to the uploaded audio or video file.
        backend: ``"pytorch"`` or ``"onnx"``.
        transcript: User-supplied transcript text (AVT + onnx only).
        skip_text: If ``True`` (AVT + onnx only), skip ASR and predict with
            the text features masked out instead.

    Yields:
        tuple: ``(figure, log)`` -- ``figure`` is ``None`` until the final step;
        ``log`` accumulates the stage messages.
    """
    if not clip_path:
        raise gr.Error("Please upload an audio or video clip first.")
    # Gradio's own File component already writes the upload to a
    # server-managed temp path before this function ever sees it -- clip_path
    # is never a raw client-supplied path string. Still validated explicitly
    # (existing, regular file) rather than trusted implicitly, so this
    # boundary holds even if a future caller wires predict_clip up
    # differently.
    # lgtm[py/path-injection] -- see comment above: clip_path originates
    # from Gradio's own upload handling, not a raw request path.
    resolved_clip_path = Path(clip_path).resolve()
    # lgtm[py/path-injection] -- same clip_path; this is the validation
    # itself (existence check), not a use of unvalidated input.
    if not resolved_clip_path.is_file():
        raise gr.Error("Uploaded clip could not be found on disk.")
    if model_id not in MODEL_IDS:
        raise gr.Error(f"Unknown model {model_id!r}; choose one of {MODEL_IDS}.")
    if backend not in BACKENDS:
        raise gr.Error(f"Unknown backend {backend!r}; choose one of {BACKENDS}.")

    lines: list[str] = [f"Model: {model_id}  Backend: {backend}"]

    def log(message: str) -> str:
        lines.append(message)
        logger.info(message)
        return "\n".join(lines)

    yield None, log("Loading model (cached after first use)...")

    total_start = time.perf_counter()
    try:
        if backend == "pytorch":
            scores = _predict_pytorch(model_id, str(resolved_clip_path), log)
        else:
            scores = _predict_onnx(model_id, str(resolved_clip_path), transcript, skip_text, log)
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"Prediction failed: {exc}") from exc
    total = time.perf_counter() - total_start
    yield _scores_to_plot(scores), log(f"Done. Total wall-clock time: {total:.3f}s")


def _on_model_change(model_id: str):
    """Show/hide the AVT-only transcript controls and force the backend.

    Args:
        model_id: One of :data:`MODEL_IDS`.

    Returns:
        tuple: Three ``gr.update(...)`` values for ``transcript_in``,
        ``skip_text_in``, and ``backend_in`` -- AVT (``predict_from_media``)
        is ONNX-only, so selecting it forces the backend dropdown to
        ``"onnx"``.
    """
    is_avt = model_id != "wavlm_best-ccc"
    backend_update = gr.update(value="onnx") if is_avt else gr.update()
    return gr.update(visible=is_avt), gr.update(visible=is_avt), backend_update


def build_demo() -> gr.Blocks:
    """Assemble the Gradio interface."""
    with gr.Blocks(title="Big Five") as demo:
        gr.Markdown("# Big Five")
        gr.Markdown(
            "Pick a champion model and backend, upload a clip, and compare timings "
            "on the fly — the progress log below reports each stage's wall-clock time."
        )

        # Row 1: model + backend selectors.
        with gr.Row():
            model_in = gr.Dropdown(
                choices=list(MODEL_IDS),
                value="wavlm_best-ccc",
                label="Model",
                info="wavlm_best-ccc = audio-only (LinT). "
                "avt_best-ccc = full multimodal (LinMulT).",
            )
            backend_in = gr.Dropdown(
                choices=list(BACKENDS),
                value="pytorch",
                label="Backend",
                info="pytorch = torch/lightning (audio-only champion only). "
                "onnx = onnxruntime, full pipeline (both champions).",
            )

        # Row 2: the upload box.
        clip_in = gr.File(
            type="filepath",
            label="Upload an audio or video clip",
            file_types=["audio", "video"],
        )

        # Row 3: AVT-only transcript controls, hidden for the audio-only champion.
        with gr.Row():
            transcript_in = gr.Textbox(
                label="Transcript (optional, AVT only)",
                placeholder="Leave blank to transcribe with Whisper automatically...",
                visible=False,
            )
            skip_text_in = gr.Checkbox(
                label="Skip text (AVT only)",
                info="Predict from audio/video alone, ignoring the transcript field above.",
                visible=False,
            )

        model_in.change(_on_model_change, [model_in], [transcript_in, skip_text_in, backend_in])

        # Row 4: the predict button.
        clip_btn = gr.Button("Predict", variant="primary")

        # Row 5: progress log (left) + plot (right).
        with gr.Row():
            log_out = gr.Textbox(
                label="Progress log (with per-stage timing)",
                lines=10,
                interactive=False,
                placeholder="Pipeline stages appear here as they run, each with its own timing...",
            )
            plot_out = gr.Plot(label="Big Five")

        clip_btn.click(
            predict_clip,
            [model_in, clip_in, backend_in, transcript_in, skip_text_in],
            [plot_out, log_out],
        )

    return demo


def main() -> None:
    """Launch the demo."""
    logging.basicConfig(level=logging.INFO)
    build_demo().launch()


if __name__ == "__main__":
    main()
