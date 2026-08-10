"""The ONNX Runtime-backed counterpart to :class:`~personalitylinmult.model.PersonalityModel`.

A separate class, not a ``backend=`` flag on ``PersonalityModel``: the two
back ends have genuinely different runtimes (``torch``/``lightning`` vs.
``onnxruntime``) and this keeps a plain ``pip install personalitylinmult``
free of ``onnxruntime`` for callers who only want the PyTorch-native path
(e.g. loading the trained weights as an ``nn.Module`` inside another PyTorch
project). Install the ``onnx`` extra to use this class::

    pip install personalitylinmult[onnx]

Both single-feature (LinT) and multi-feature (LinMulT, e.g. the AVT champion)
checkpoints have an ONNX export path -- see
:mod:`personalitylinmult.inference.onnx_export` for how each is traced.
:meth:`predict` handles any number of features transparently, reading
``feature_names`` from the ``.onnx.json`` sidecar (always plural, regardless
of family -- a LinT sidecar simply has length-1 lists). Model ids for this
class carry a ``.onnx`` suffix by HF Hub filename convention (see
:func:`_onnx_filenames`), the same repo as the ``.ckpt`` files
:class:`~personalitylinmult.model.PersonalityModel` downloads.

Two full raw-media pipelines are offered, matched to the champion they serve:

- :meth:`predict_from_audio` -- audio only (``wavlm``-only champions): raw
  audio/video in, Big Five out, on ONNX Runtime **alone**, no
  ``torch``/``transformers``/``exordium`` at inference time. WavLM's own
  ONNX export (:data:`WAVLM_ONNX_FILENAME`) downloads lazily.
- :meth:`predict_from_media` -- the full multimodal (AVT) champion: every
  learned feature extractor (WavLM, emotion2vec, DINOv2, EmotiEffNet, FaRL,
  mmBERT, XLM-RoBERTa) and the trained head itself run on ONNX Runtime, with
  two narrowly-scoped, explicitly-documented exceptions that stay
  ``torch``/``exordium``-based: face detection/tracking (EmotiEffNet/FaRL
  need a face crop, not a raw frame -- exordium's existing offline-pipeline
  detector/tracker, reused for a single already-complete video file, not a
  live/streaming input) and Whisper ASR (an encoder-decoder model with no
  ONNX export in this project, used only when ``transcript=None``). Both
  are dependencies of this method specifically, not of the class or of
  ``predict``/``predict_from_audio`` -- a caller who never calls
  ``predict_from_media`` never needs ``torch``/``exordium`` installed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from personalitylinmult.model import HF_MODEL_REPO

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)
"""Module-level logger."""


def _onnx_filenames(model_id: str) -> tuple[str, str]:
    """Map a model id to its ONNX graph and metadata filenames in the HF repo.

    Args:
        model_id (str): e.g. ``"wavlm_best-ccc"``.

    Returns:
        tuple[str, str]: ``(graph_filename, metadata_filename)``, e.g.
        ``("wavlm_best-ccc.onnx", "wavlm_best-ccc.onnx.json")``.
    """
    return f"{model_id}.onnx", f"{model_id}.onnx.json"


WAVLM_ONNX_FILENAME = "wavlm-base-plus.onnx"
"""Shared WavLM ONNX graph filename in the HF model repo -- one file, reused
by every champion whose feature is ``wavlm``, not tied to a specific
champion id (unlike the trained-model graphs, which are one file per
champion)."""


def _download_wavlm_onnx(revision: str | None = None) -> Path:
    """Download (or reuse the cached) shared WavLM ONNX graph.

    Args:
        revision (str | None): HF Hub revision; ``None`` uses the default
            branch.

    Returns:
        Path: Local path to the ``.onnx`` graph.
    """
    return _download_shared_onnx(WAVLM_ONNX_FILENAME, revision=revision)


EXTRACTOR_ONNX_FILENAMES: dict[str, str] = {
    "wavlm": WAVLM_ONNX_FILENAME,
    "emotion2vec": "emotion2vec-plus-seed.onnx",
    "dinov2": "dinov2-base.onnx",
    "emotieffnet": "emotieffnet-enet-b0-8-best-vgaf.onnx",
    "farl": "farl-ep64.onnx",
    "xlm_roberta_gt": "xlm-roberta-mpnet.onnx",
    "mmbert_gt": "mmbert-base.onnx",
}
"""Shared ONNX graph filename per AVT feature, matching the Makefile's
``push-<name>-onnx`` targets exactly -- one file per extractor, reused
across every champion that consumes that feature, not tied to one champion
id (unlike the trained-model graph, which is)."""


def _download_shared_onnx(filename: str, revision: str | None = None) -> Path:
    """Download (or reuse the cached) a shared extractor ONNX graph.

    Args:
        filename (str): The graph's filename in the HF model repo, e.g.
            :data:`WAVLM_ONNX_FILENAME` or one of
            :data:`EXTRACTOR_ONNX_FILENAMES`'s values.
        revision (str | None): HF Hub revision; ``None`` uses the default
            branch.

    Returns:
        Path: Local path to the ``.onnx`` graph.
    """
    from huggingface_hub import hf_hub_download

    from personalitylinmult import WEIGHTS_DIR

    path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=filename,
        revision=revision,
        cache_dir=str(WEIGHTS_DIR),
    )
    logger.info(f"Downloaded shared ONNX graph: {path}.")
    return Path(path)


def _download(model_id: str, revision: str | None = None) -> tuple[Path, Path]:
    """Download an exported ONNX graph and its metadata sidecar from HF Hub.

    A separate function (not inlined into
    :meth:`PersonalityModelONNX.from_pretrained`) so tests can substitute
    local paths without a network call -- see ``tests/test_onnx.py``.

    Args:
        model_id (str): The champion id to download.
        revision (str | None): HF Hub revision (branch/tag/commit); ``None``
            uses the repo's default branch.

    Returns:
        tuple[Path, Path]: ``(onnx_path, metadata_path)``, both cached under
        :data:`personalitylinmult.WEIGHTS_DIR` across calls.
    """
    from huggingface_hub import hf_hub_download

    # Imported lazily so this module places no import-time requirement on the
    # package's top-level __init__ beyond what PersonalityModel already needs.
    from personalitylinmult import WEIGHTS_DIR

    graph_filename, metadata_filename = _onnx_filenames(model_id)
    onnx_path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=graph_filename,
        revision=revision,
        cache_dir=str(WEIGHTS_DIR),
    )
    metadata_path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=metadata_filename,
        revision=revision,
        cache_dir=str(WEIGHTS_DIR),
    )
    # lgtm[py/log-injection] -- repr(), not str(): model_id is a public API
    # argument (caller-supplied), and repr() escapes newlines/control
    # characters so it cannot forge extra log lines.
    logger.info(f"{model_id!r}: downloaded {onnx_path} and {metadata_path}.")
    return Path(onnx_path), Path(metadata_path)


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
"""ImageNet RGB channel means, matching ``exordium.video.deep.base
._IMAGENET_MEAN`` -- used by DINOv2 and EmotiEffNet's preprocessing."""

_IMAGENET_STD = [0.229, 0.224, 0.225]
"""ImageNet RGB channel standard deviations, matching ``exordium.video.deep
.base._IMAGENET_STD``."""

_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
"""CLIP RGB channel means, matching ``exordium.video.deep.farl._CLIP_MEAN``
-- FaRL inherits OpenAI CLIP's normalization, not ImageNet's."""

_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
"""CLIP RGB channel standard deviations, matching ``exordium.video.deep
.farl._CLIP_STD``."""


def _preprocess_frame_224(
    frame,
    mean: list[float],
    std: list[float],
    interpolation: str,
):
    """Resize a ``(3, H, W)`` uint8 frame to 224x224 and normalize it.

    Replicates the resize+normalize half of each vision wrapper's own
    ``preprocess()`` (:class:`~exordium.video.deep.dinov2.DINOv2Wrapper`,
    :class:`~exordium.video.deep.emotieffnet.EmotiEffNetWrapper`,
    :class:`~exordium.video.deep.farl.FarlWrapper`) exactly -- the three
    differ in interpolation mode (DINOv2/FaRL: bicubic; EmotiEffNet:
    bilinear) and normalization constants (DINOv2/EmotiEffNet: ImageNet;
    FaRL: CLIP), so both must be passed explicitly per call rather than
    assumed, matching the module-level warning in ``onnx_export.py`` about
    these three being an easy copy-paste footgun.

    Args:
        frame: ``(3, H, W)`` uint8 array or tensor (channel-first, matching
            both ``exordium.video.core.io.load_video``'s and
            ``Detection.crop()``'s output convention).
        mean (list[float]): Per-channel normalization mean.
        std (list[float]): Per-channel normalization standard deviation.
        interpolation (str): ``"bicubic"`` or ``"bilinear"``.

    Returns:
        np.ndarray: ``(1, 3, 224, 224)`` float32, ready for an ONNX session.
    """
    import numpy as np
    import torch
    import torchvision.transforms.functional as TF

    # torchvision (not PIL directly) to match the real wrappers'
    # DINOv2Wrapper/EmotiEffNetWrapper/FarlWrapper.preprocess() exactly,
    # including antialias=True -- already a required dependency for this
    # method's face-detection/tracking step, so no new dependency is added
    # by using it here too.
    x = torch.as_tensor(np.asarray(frame))
    resample = (
        TF.InterpolationMode.BICUBIC
        if interpolation == "bicubic"
        else TF.InterpolationMode.BILINEAR
    )
    resized = TF.resize(x, [224, 224], interpolation=resample, antialias=True)
    chw = resized.float().div(255).numpy()

    mean_arr = np.asarray(mean, dtype="float32").reshape(3, 1, 1)
    std_arr = np.asarray(std, dtype="float32").reshape(3, 1, 1)
    normalized = (chw - mean_arr) / std_arr
    return normalized[np.newaxis, :, :, :].astype("float32")  # (1, 3, 224, 224)


def _densify_face_features(protagonist, num_frames: int, emotieffnet_session, farl_session):
    """Run EmotiEffNet + FaRL on a protagonist track, densified to every frame.

    A face track (:class:`exordium.video.core.detection.Track`) commonly has
    gaps -- occlusion, brief exits, scene cuts -- so it can have fewer
    detections than the clip has frames. This computes each extractor's
    *sparse* per-detection features (one row per real detection, matching
    :meth:`~exordium.video.deep.base.VisualModelWrapper.track_to_feature`'s
    own sparse-by-default output), then scatters them onto the full frame
    grid via exordium's own :func:`~exordium.video.core.densify.densify` --
    the same function every frame-wise exordium vision wrapper
    (EmotiEffNet, FaRL, DINOv2, ...) already uses internally for
    ``track_to_feature(..., densify=True)``. Reused here rather than
    reimplemented because the real ``EmotiEffNetWrapper``/``FarlWrapper``
    objects run PyTorch inference, while this class runs ONNX Runtime
    sessions on the same crops -- the densify *step* is architecture-
    independent, so only that part is reused, not the whole wrapper.

    Args:
        protagonist: A :class:`~exordium.video.core.detection.Track`-like
            object exposing ``frame_ids() -> list[int]`` and
            ``get_detection(frame_id) -> Detection``, or ``None``/empty if no
            protagonist was found.
        num_frames (int): Total frames in the clip -- the full frame grid to
            densify onto.
        emotieffnet_session: ONNX Runtime session for EmotiEffNet.
        farl_session: ONNX Runtime session for FaRL.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: ``(emotieffnet_features,
        farl_features, mask)`` -- the first two ``(num_frames, D)`` float32
        (``D`` = 1280 / 512), ``mask`` a shared ``(num_frames,)`` bool array,
        ``True`` where a real detection was used (both extractors share one
        mask since they run on the same crops, at the same frame ids).
    """
    import numpy as np
    import torch
    from exordium.video.core.densify import densify

    if protagonist is None or len(protagonist) == 0:
        empty_mask = torch.zeros(num_frames, dtype=torch.bool)
        return (
            np.zeros((num_frames, 1280), dtype="float32"),
            np.zeros((num_frames, 512), dtype="float32"),
            empty_mask.numpy(),
        )

    frame_ids: list[int] = []
    emotieffnet_rows: list[np.ndarray] = []
    farl_rows: list[np.ndarray] = []
    for frame_id in protagonist.frame_ids():
        if frame_id >= num_frames:
            continue  # detector-reported index outside the decoded frame range
        detection = protagonist.get_detection(frame_id)
        crop = detection.crop(square=True, extra_space=1.5).numpy()  # (3, H, W) uint8

        # EmotiEffNetWrapper.preprocess uses bilinear resize + ImageNet
        # normalization; FarlWrapper.preprocess uses bicubic resize + CLIP
        # normalization -- replicated exactly, not copy-pasted from one
        # another (see onnx_export.py's export functions for the same
        # distinction, and the easy-footgun note in their docstrings about
        # these three vision extractors' preprocessing differing in ways
        # that look interchangeable but are not).
        emotieffnet_input = _preprocess_frame_224(
            crop, mean=_IMAGENET_MEAN, std=_IMAGENET_STD, interpolation="bilinear"
        )
        (emotieffnet_out,) = emotieffnet_session.run(
            ["features"], {"pixel_values": emotieffnet_input}
        )
        farl_input = _preprocess_frame_224(
            crop, mean=_CLIP_MEAN, std=_CLIP_STD, interpolation="bicubic"
        )
        (farl_out,) = farl_session.run(["embedding"], {"pixel_values": farl_input})

        frame_ids.append(frame_id)
        emotieffnet_rows.append(emotieffnet_out[0])
        farl_rows.append(farl_out[0])

    frame_ids_t = torch.tensor(frame_ids, dtype=torch.long)
    emotieffnet_sparse = torch.from_numpy(np.stack(emotieffnet_rows).astype("float32"))
    farl_sparse = torch.from_numpy(np.stack(farl_rows).astype("float32"))

    emotieffnet_dense = densify(frame_ids_t, emotieffnet_sparse, end_frame_id=num_frames)
    farl_dense = densify(frame_ids_t, farl_sparse, end_frame_id=num_frames)

    return (
        emotieffnet_dense["features"].numpy(),
        farl_dense["features"].numpy(),
        emotieffnet_dense["mask"].numpy(),
    )


def _read_metadata(metadata_path: Path) -> tuple[list[str], list[str]]:
    """Read a ``.onnx.json`` sidecar, normalizing old and new schemas.

    Sidecars written by the current :func:`~personalitylinmult.inference
    .onnx_export.export_to_onnx` always use the plural schema
    (``feature_names``, ``input_dims``), for both LinT and LinMulT graphs.
    Older sidecars (published before the LinMulT export existed) use the
    singular ``feature_name``/``input_dim`` keys -- read transparently here
    so a previously-published graph (e.g. ``wavlm_best-ccc.onnx.json``)
    keeps working without a re-export, and the rest of this class never has
    to branch on which schema was on disk.

    Args:
        metadata_path (Path): Path to a ``.onnx.json`` sidecar.

    Returns:
        tuple[list[str], list[str]]: ``(feature_names, traits)``, in the
        checkpoint's original order.
    """
    metadata = json.loads(metadata_path.read_text())
    if "feature_names" in metadata:
        feature_names = list(metadata["feature_names"])
    else:
        feature_names = [metadata["feature_name"]]
    return feature_names, list(metadata["traits"])


class PersonalityModelONNX:
    """A champion FI model, running on ONNX Runtime instead of PyTorch.

    Args:
        session: An ``onnxruntime.InferenceSession`` over the exported graph.
        feature_names (list[str]): Features this model consumes, in input
            order (length 1 for a LinT graph, more for a LinMulT graph).
        traits (list[str]): Trait names, in output-column order.
    """

    def __init__(self, session, feature_names: list[str], traits: list[str]):
        self._session = session
        self.feature_names = list(feature_names)
        self.traits = list(traits)
        self._wavlm_session = None
        self._wavlm_revision: str | None = None
        self._extractor_sessions: dict[str, object] = {}
        self._whisper = None

    @classmethod
    def from_pretrained(cls, model_id: str, revision: str | None = None) -> PersonalityModelONNX:
        """Download (or reuse the cached) ONNX graph and load it.

        Args:
            model_id (str): A champion id, e.g. ``"wavlm_best-ccc"``. See the
                module docstring for which ids have an ONNX export available.
            revision (str | None): HF Hub revision (branch/tag/commit);
                ``None`` uses the repo's default branch.

        Returns:
            PersonalityModelONNX: Ready to predict.
        """
        import onnxruntime as ort

        onnx_path, metadata_path = _download(model_id, revision=revision)
        feature_names, traits = _read_metadata(metadata_path)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        # lgtm[py/log-injection] -- repr(), not str(): model_id is a public
        # API argument (caller-supplied), and repr() escapes newlines/control
        # characters so it cannot forge extra log lines. feature_names/traits
        # come from _read_metadata's downloaded .onnx.json sidecar, not from
        # this call's caller.
        logger.info(f"Loaded ONNX {model_id!r} (features={feature_names}, traits={traits}).")
        return cls(session, feature_names=feature_names, traits=traits)

    def predict(
        self,
        features: dict[str, np.ndarray],
        masks: dict[str, np.ndarray] | None = None,
    ) -> dict[str, float]:
        """Predict Big Five scores for one sample.

        Args:
            features (dict[str, np.ndarray]): ``{feature_name: (T, F) array}``
                -- one entry per :attr:`feature_names`.
            masks (dict[str, np.ndarray] | None): Optional per-feature
                ``(T,)`` bool override mask (``True`` = valid). Features not
                present here default to an all-``True`` mask, matching every
                caller before this parameter existed. Used by
                :meth:`predict_from_media` to mark a whole feature as
                genuinely missing (all-``False``) when, e.g., no transcript
                is available and ASR/skip-text mode was requested -- verified
                NaN-safe end-to-end for this project's champion architecture,
                see ``tests/inference/test_linmult_mask_validation.py`` and
                ``tests/inference/test_onnx_export.py``'s
                ``test_onnx_all_false_text_masks_match_pytorch``. The
                feature's own *content* does not matter once masked
                all-``False`` -- only the mask does.

        Returns:
            dict[str, float]: ``{trait: score}``, clamped to ``[0, 1]``, in
            the model's trait order.

        Raises:
            KeyError: If ``features`` is missing a required feature.
        """
        import numpy as np

        masks = masks or {}

        def _mask_for(name: str, x: np.ndarray) -> np.ndarray:
            if name in masks:
                return np.asarray(masks[name], dtype=bool)[np.newaxis, :]  # (1, T)
            return np.ones(x.shape[:2], dtype=bool)  # (1, T), all-True default

        onnx_inputs: dict[str, np.ndarray] = {}
        if len(self.feature_names) == 1:
            name = self.feature_names[0]
            array = np.asarray(features[name], dtype="float32")
            x = array[np.newaxis, :, :]  # (1, T, F)
            onnx_inputs["x"] = x
            onnx_inputs["mask"] = _mask_for(name, x)
        else:
            for i, name in enumerate(self.feature_names):
                array = np.asarray(features[name], dtype="float32")
                x = array[np.newaxis, :, :]  # (1, T, F)
                onnx_inputs[f"x_{i}"] = x
                onnx_inputs[f"mask_{i}"] = _mask_for(name, x)

        (raw,) = self._session.run(["predictions"], onnx_inputs)
        clamped = np.clip(raw[0], 0.0, 1.0)
        return dict(zip(self.traits, clamped.tolist(), strict=True))

    def predict_from_audio(self, audio_path: str, revision: str | None = None) -> dict[str, float]:
        """Predict Big Five scores directly from an audio/video file.

        The full pipeline (decode -> WavLM -> LinT -> scores) on ONNX Runtime
        alone -- no ``torch``, ``transformers``, or ``exordium`` needed.
        ``ffmpeg`` is still required as an external binary for decoding
        (a subprocess call, not a Python dependency).

        The shared WavLM ONNX graph (:data:`WAVLM_ONNX_FILENAME`) downloads
        on first use and is cached on this instance for later calls.

        Args:
            audio_path (str): An audio or video file.
            revision (str | None): HF Hub revision for the WavLM graph
                download; ``None`` uses the default branch. Ignored on
                subsequent calls once the WavLM session is cached.

        Returns:
            dict[str, float]: ``{trait: score}``, in the model's trait order.

        Raises:
            ValueError: If this model's only feature is not ``"wavlm"`` -- the
                only feature this method currently knows how to extract from
                raw audio. A multi-feature (AVT) champion needs
                ``predict_from_media`` instead (in progress).
        """
        if self.feature_names != ["wavlm"]:
            raise ValueError(
                f"predict_from_audio only supports a wavlm-only champion, "
                f"got feature_names={self.feature_names!r}."
            )

        if self._wavlm_session is None:
            import onnxruntime as ort

            wavlm_path = _download_wavlm_onnx(revision=revision)
            self._wavlm_session = ort.InferenceSession(
                str(wavlm_path), providers=["CPUExecutionProvider"]
            )

        from personalitylinmult.inference.audio_features_onnx import extract_wavlm_onnx

        features = extract_wavlm_onnx(audio_path, self._wavlm_session)
        return self.predict({"wavlm": features})

    AVT_FEATURE_NAMES = (
        "wavlm",
        "emotion2vec",
        "dinov2",
        "emotieffnet",
        "farl",
        "xlm_roberta_gt",
        "mmbert_gt",
    )
    """Feature order the full multimodal (AVT) champion expects -- matches
    ``config/FI/data/avt.yaml``'s ``features:`` list, which is what
    ``avt_best-ccc``'s checkpoint was actually trained and exported with."""

    def _extractor_session(self, feature_name: str, revision: str | None = None):
        """Lazily download and cache one extractor's ONNX session.

        Mirrors :meth:`predict_from_audio`'s ``self._wavlm_session`` lazy
        pattern, generalized to any of :data:`EXTRACTOR_ONNX_FILENAMES` --
        a caller who never touches a given feature (e.g. always supplies a
        transcript, skipping the text extractors) never pays that
        extractor's download/session-build cost.
        """
        if feature_name not in self._extractor_sessions:
            import onnxruntime as ort

            filename = EXTRACTOR_ONNX_FILENAMES[feature_name]
            path = _download_shared_onnx(filename, revision=revision)
            self._extractor_sessions[feature_name] = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
        return self._extractor_sessions[feature_name]

    def predict_from_media(
        self,
        video_path: str,
        transcript: str | None = None,
        revision: str | None = None,
    ) -> dict[str, float]:
        """Predict Big Five scores from a video file, for the full AVT champion.

        Runs every extractor this champion needs (:attr:`feature_names`,
        expected to be :data:`AVT_FEATURE_NAMES`) on ONNX Runtime, except for
        two narrowly-scoped, explicitly-documented exceptions that stay
        ``torch``/``exordium``-based -- see this method's own steps below and
        the module docstring's "Dependencies" note:

        - **Face detection/tracking** (for ``emotieffnet``/``farl``, which
          need a face crop, not a full frame): exordium's
          ``YoloFace11Detector`` + ``FaceClusterTracker`` + ``AdaFaceWrapper``
          stack, the same one this project's own offline dataset-build
          (``personalitylinmult/preprocess/fi/create_vdets.py``,
          ``create_tracks.py``) already uses per-clip -- this is a *batch*
          call over one already-complete video file, not a live/streaming
          use case, so the existing offline-oriented API is exactly the
          right fit, not a mismatch to work around.
        - **ASR** (when ``transcript is None``): exordium's
          ``WhisperWrapper``, lazily loaded, since Whisper's encoder-decoder
          architecture with beam-search ``generate()`` has no ONNX export in
          this project (see ``onnx_export.py``'s module docstring).

        The video is only decoded/detected/tracked once and every extractor
        reuses that single pass -- not once per feature.

        Args:
            video_path (str): A video file (audio + visual features are both
                extracted from it).
            transcript (str | None): Controls how the two text features
                (``xlm_roberta_gt``, ``mmbert_gt``) are resolved:

                - A non-empty string -- used directly, no ASR.
                - ``None`` (default) -- run Whisper ASR on the video's audio
                  track to obtain a transcript.
                - ``""`` (empty string) -- skip text entirely: both text
                  features get a zero placeholder tensor with an all-``False``
                  mask, verified NaN-safe and content-independent for this
                  champion's architecture (see :meth:`predict`'s docstring).
            revision (str | None): HF Hub revision for every extractor graph
                download; ``None`` uses the default branch. Ignored for
                already-cached sessions.

        Returns:
            dict[str, float]: ``{trait: score}``, in the model's trait order.

        Raises:
            ValueError: If this model's features are not exactly
                :data:`AVT_FEATURE_NAMES` (in any order) -- this method is
                specific to the full multimodal champion, not the audio-only
                one (use :meth:`predict_from_audio` there instead).
        """
        if set(self.feature_names) != set(self.AVT_FEATURE_NAMES):
            raise ValueError(
                f"predict_from_media only supports the full AVT champion "
                f"(features={sorted(self.AVT_FEATURE_NAMES)}), got "
                f"feature_names={self.feature_names!r}."
            )

        import numpy as np

        from personalitylinmult.inference.audio_features_onnx import (
            extract_emotion2vec_onnx,
            extract_wavlm_onnx,
        )

        features: dict[str, np.ndarray] = {}
        masks: dict[str, np.ndarray] = {}

        # -- Audio features: reuse the existing WavLM/emotion2vec ONNX decode
        # path unchanged, one ffmpeg decode each (matching predict_from_audio;
        # not deduplicated into a single shared waveform since the two
        # extractors' sample-rate/format contracts are already independently
        # correct and this keeps each extractor function self-contained).
        wavlm_session = self._extractor_session("wavlm", revision=revision)
        features["wavlm"] = extract_wavlm_onnx(video_path, wavlm_session)

        emotion2vec_session = self._extractor_session("emotion2vec", revision=revision)
        features["emotion2vec"] = extract_emotion2vec_onnx(video_path, emotion2vec_session)

        # -- Visual features: DINOv2 (full frames) + EmotiEffNet/FaRL (face
        # crops, via exordium's face detection/tracking -- a torch/exordium
        # dependency by design, see this method's docstring).
        features["dinov2"] = self._extract_dinov2(video_path, revision=revision)
        emotieffnet_features, farl_features, face_mask = self._extract_face_features(
            video_path, revision=revision
        )
        features["emotieffnet"] = emotieffnet_features
        features["farl"] = farl_features
        masks["emotieffnet"] = face_mask
        masks["farl"] = face_mask

        # -- Text features: the 3-way transcript contract.
        if transcript == "":
            text_dim = 768  # both xlm_roberta_gt and mmbert_gt are 768-d
            placeholder = np.zeros((1, text_dim), dtype="float32")
            features["xlm_roberta_gt"] = placeholder
            features["mmbert_gt"] = placeholder
            masks["xlm_roberta_gt"] = np.zeros(1, dtype=bool)
            masks["mmbert_gt"] = np.zeros(1, dtype=bool)
        else:
            resolved_transcript = transcript
            if resolved_transcript is None:
                resolved_transcript = self._transcribe(video_path)
            xlm_features, mmbert_features = self._extract_text_features(
                resolved_transcript, revision=revision
            )
            features["xlm_roberta_gt"] = xlm_features
            features["mmbert_gt"] = mmbert_features

        return self.predict(features, masks=masks or None)

    def _extract_dinov2(self, video_path: str, revision: str | None = None):
        """Extract L2-normalized DINOv2 CLS-token features for every frame.

        Args:
            video_path (str): A video file.
            revision (str | None): HF Hub revision for the DINOv2 graph.

        Returns:
            np.ndarray: ``(T, 768)`` float32 features, one per frame.
        """
        import numpy as np
        from exordium.video.core.io import load_video

        session = self._extractor_session("dinov2", revision=revision)
        frames, _actual_fps = load_video(video_path)  # (T, 3, H, W) uint8

        outputs = []
        for frame in frames:
            pixel_values = _preprocess_frame_224(
                frame.numpy(), mean=_IMAGENET_MEAN, std=_IMAGENET_STD, interpolation="bicubic"
            )
            (embedding,) = session.run(["cls_embedding"], {"pixel_values": pixel_values})
            outputs.append(embedding[0])
        return np.stack(outputs).astype("float32")

    def _extract_face_features(self, video_path: str, revision: str | None = None):
        """Detect and track the protagonist, then run EmotiEffNet + FaRL.

        A single detection/tracking pass over ``video_path`` feeds both
        extractors -- exordium's own offline dataset-build pipeline
        (``create_vdets.py`` + ``create_tracks.py``) does the same two steps
        per clip; this reuses that pattern for a single ad hoc call rather
        than a whole-dataset batch job.

        Args:
            video_path (str): A video file.
            revision (str | None): HF Hub revision for the EmotiEffNet/FaRL
                graphs.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: ``(emotieffnet_features,
            farl_features, mask)``, the first two ``(num_frames, D)`` float32,
            densified onto the full frame grid: a frame where the protagonist
            track has a detection gets that detection's real feature; a frame
            with no detection (occlusion, brief exit, or no protagonist found
            at all) gets an all-zero placeholder row. ``mask`` is a shared
            ``(num_frames,)`` bool array -- ``True`` where a real detection
            was used, ``False`` for the zero-filled gaps -- passed by the
            caller to :meth:`predict`'s ``masks`` override so a placeholder
            frame is genuinely excluded from attention, not just zeroed.
        """
        from exordium.video.core.detection import FaceClusterTracker
        from exordium.video.core.io import load_video
        from exordium.video.deep.adaface import AdaFaceWrapper
        from exordium.video.face.detector.yolo11 import YoloFace11Detector

        detector = YoloFace11Detector(device_id=-1)
        vdets = detector.detect_video(video_path)

        frames, _actual_fps = load_video(video_path)  # (T, 3, H, W) uint8
        num_frames = max(len(frames), 1)

        encoder = AdaFaceWrapper(device_id=-1)
        tracker = FaceClusterTracker(encoder=encoder, verbose=False)
        protagonist = tracker.run(vdets, min_score=0.5)

        emotieffnet_session = self._extractor_session("emotieffnet", revision=revision)
        farl_session = self._extractor_session("farl", revision=revision)

        if protagonist is None or len(protagonist) == 0:
            logger.warning(f"{video_path}: no protagonist face track found; using zero features.")

        return _densify_face_features(protagonist, num_frames, emotieffnet_session, farl_session)

    def _extract_text_features(self, text: str, revision: str | None = None):
        """Tokenize and run mmBERT + XLM-RoBERTa on a transcript.

        Args:
            text (str): The transcript to encode.
            revision (str | None): HF Hub revision for the text extractor
                graphs.

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(xlm_roberta_features,
            mmbert_features)``, each ``(T, 768)`` float32 token-level
            embeddings.
        """
        import numpy as np
        from transformers import AutoTokenizer

        from personalitylinmult.inference.onnx_export import (
            MMBERT_MODEL_ID,
            XLM_ROBERTA_MODEL_ID,
        )

        def _run(feature_name: str, model_id: str) -> np.ndarray:
            session = self._extractor_session(feature_name, revision=revision)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            # AutoTokenizer.from_pretrained's stub allows None for exotic
            # backend-selection edge cases that do not apply to a concrete,
            # known-good model id -- same class of false ambiguity as
            # onnx_export.py's emotion2vec wrapper, see its comment.
            tokens = tokenizer(text, return_tensors="np")  # ty: ignore[call-non-callable]
            onnx_inputs = {
                "input_ids": tokens["input_ids"].astype(np.int64),
                "attention_mask": tokens["attention_mask"].astype(np.int64),
            }
            (hidden,) = session.run(["last_hidden_state"], onnx_inputs)
            result: np.ndarray = hidden[0]  # (T, 768)
            return result.astype("float32")

        xlm_features = _run("xlm_roberta_gt", XLM_ROBERTA_MODEL_ID)
        mmbert_features = _run("mmbert_gt", MMBERT_MODEL_ID)
        return xlm_features, mmbert_features

    def _transcribe(self, video_path: str) -> str:
        """Lazily load Whisper and transcribe a video's audio track.

        ``torch``/``transformers``-based (see the module docstring's
        Dependencies note) -- only loaded the first time ``transcript=None``
        is actually hit, so a caller who always supplies a transcript (or
        ``""``) never pays this cost.

        Args:
            video_path (str): A video file.

        Returns:
            str: The transcribed text.
        """
        if self._whisper is None:
            from exordium.text.whisper import WhisperWrapper

            self._whisper = WhisperWrapper(device_id=-1)
            logger.info("Loaded Whisper for predict_from_media's ASR fallback.")

        text: str = self._whisper(video_path)
        return text
