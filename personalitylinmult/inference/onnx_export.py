"""Export a trained checkpoint's model, and its feature extractors, to ONNX.

Three independent exports live here:

- :func:`export_to_onnx` -- the *trained regression model*, LinT (single
  feature) or LinMulT (multiple features, e.g. the AVT champion).
- :func:`export_wavlm_to_onnx` -- WavLM itself (via ``transformers``, the same
  model :class:`exordium.audio.wavlm.WavlmWrapper` wraps), so the full
  audio -> features -> prediction pipeline can run on ONNX Runtime alone, with
  no ``torch``/``transformers``/``exordium`` at inference time.

``export_to_onnx`` handles both model families with two different wrapper
classes, since ONNX graphs have a fixed input signature:

- LinT's ``forward(x, mask, name)`` takes two tensors and one constant
  string, which traces directly (:class:`_LinTExportWrapper`).
- LinMulT's ``forward(inputs, masks, names)`` takes a *list* of tensors, one
  per modality -- ONNX/``torch.onnx.export`` do not accept a Python
  ``list[Tensor]`` as a traced input the way LinMulT's own signature does, so
  :class:`_LinMulTExportWrapper` unpacks a fixed-arity ``*args`` (built once
  per checkpoint, sized to that checkpoint's ``len(feature_names)``) into two
  equal halves -- the first half is ``inputs``, the second is ``masks``, in
  ``feature_names`` order. The consumer (:class:`~personalitylinmult.onnx
  .PersonalityModelONNX`) already knows this order from the ``.onnx.json``
  sidecar, so the graph's input names (``x_0``, ``mask_0``, ``x_1``, ...) need
  no further metadata to interpret correctly.

Both branches are exported via the legacy TorchScript tracer
(``dynamo=False``): both LinT's and LinMulT's forward passes branch on a mask
tensor's *content* (LinT: ``if mask is not None and not mask.any()``;
LinMulT's ``_normalize_masks``: ``mask.all()``), which torch's newer
``torch.export``-based exporter (``dynamo=True``, the default since torch
2.9) cannot trace -- it requires a fully static graph. The tracer instead
bakes in whichever branch the trace-time input took. For LinT, every current
caller passes an all-``True`` mask (:meth:`FiPredictor._to_batch` never
pads), so tracing with an all-``True`` dummy mask is safe in practice, but a
caller feeding a genuinely padded (some-``False``) mask through the ONNX
graph would silently get the all-valid-mask behavior regardless of the
mask's real content. For LinMulT, the same caveat applies to *partial*
per-timestep masking -- but a **whole-modality** all-``False`` mask (the
"skip this feature entirely" case, e.g. no transcript available) is a
different, verified-safe code path: see
``tests/inference/test_linmult_mask_validation.py``, which confirms with a
real forward pass (not just by reading the source) that neither attention
type this project supports produces NaN for a fully-masked branch --
``linmult.core.transformer``'s ``TransformerLayer.forward`` has an explicit
guard for exactly this case, independent of ``attention_type``. Document
these limitations wherever the ONNX path is offered as an alternative to the
PyTorch path, which has no such restriction.

The exported artifact is two files, not one: the ``.onnx`` graph itself does
not carry Python-level metadata like trait names, so a ``.onnx.json`` sidecar
travels with it -- the same information
:class:`~personalitylinmult.inference.predictor.FiPredictor` reads from a
checkpoint's hyperparameters, made explicit here since the ONNX graph cannot
hold it. The sidecar schema is always plural (``feature_names``,
``input_dims``) regardless of family, so
:class:`~personalitylinmult.onnx.PersonalityModelONNX` never has to branch on
which family produced a given graph -- a LinT sidecar simply has
length-1 lists.

Note: ``linmult`` versions before 2.1.1 cached
``linmult.core.pe.PositionalEncoding``'s table across a
``TransformerEncoder``'s query and key/value calls, which could bake a
shared, trace-time sequence length into both calls of an ONNX-exported
LinMulT graph -- breaking inference for features with genuinely different
lengths (WavLM ~50 Hz vs. DINOv2 ~25 fps vs. one row per token for text).
Fixed upstream in ``linmult`` 2.1.1; this project requires ``linmult>=2.1.1``
accordingly.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from personalitylinmult.train.module import FiLightningModule

logger = logging.getLogger(__name__)
"""Module-level logger."""

ONNX_OPSET = 17
"""ONNX opset version to export against; 17 is broadly supported by current
onnxruntime releases and covers every op LinT's forward pass uses."""


class _LinTExportWrapper(nn.Module):
    """Closes over the fixed ``name`` argument LinT's forward needs.

    ``torch.onnx.export`` traces a module called with tensor arguments only;
    ``name`` is a constant string carried by the checkpoint, not a per-call
    input, so it is bound here rather than exposed as a graph input. Returns
    the single head tensor directly (not the ``{head_name: tensor}`` dict
    :meth:`FiLightningModule.forward` returns), since a bare tensor output is
    what ONNX/onnxruntime expects for a plain regression graph.
    """

    def __init__(self, model: nn.Module, name: str, head_name: str):
        super().__init__()
        self.model = model
        self._name = name
        self._head_name = head_name

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Run the wrapped model and select its single output head.

        Args:
            x (torch.Tensor): ``(B, T, F)`` input features.
            mask (torch.Tensor): ``(B, T)`` bool mask, ``True`` = valid.

        Returns:
            torch.Tensor: ``(B, output_dim)`` raw (unclamped) predictions.
        """
        return self.model(x, mask, name=self._name)[self._head_name]


class _LinMulTExportWrapper(nn.Module):
    """Closes over the fixed ``names`` list LinMulT's forward needs.

    ``torch.onnx.export`` traces a module called with a fixed tuple of tensor
    arguments, not a Python ``list[Tensor]`` the way ``LinMulT.forward``
    itself accepts -- ``forward`` here takes ``2 * n_features`` separate
    positional tensors instead (first half ``inputs``, second half ``masks``,
    both in ``names`` order) and reassembles them into the lists LinMulT
    expects before calling it.
    """

    def __init__(self, model: nn.Module, names: list[str], head_name: str):
        super().__init__()
        self.model = model
        self._names = names
        self._head_name = head_name
        self._n = len(names)

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        """Run the wrapped model and select its single output head.

        Args:
            *args (torch.Tensor): ``2 * n_features`` tensors -- the first
                ``n_features`` are ``(B, T, F)`` inputs in ``names`` order,
                the second ``n_features`` are their ``(B, T)`` bool masks.

        Returns:
            torch.Tensor: ``(B, output_dim)`` raw (unclamped) predictions.
        """
        inputs = list(args[: self._n])
        masks = list(args[self._n :])
        return self.model(inputs, masks, names=self._names)[self._head_name]


def export_to_onnx(module: FiLightningModule, output_path: str | Path) -> Path:
    """Export a trained checkpoint's model (LinT or LinMulT) to ONNX.

    Args:
        module (FiLightningModule): A loaded module (e.g. via
            ``FiLightningModule.load_from_checkpoint``), in eval mode or not
            -- this function calls ``.eval()`` itself.
        output_path (str | Path): Destination ``.onnx`` path. A sidecar
            ``.onnx.json`` is written alongside it with the same stem.

    Returns:
        Path: The written ``.onnx`` path.
    """
    from personalitylinmult.train.module import HEAD_NAME

    output_path = Path(output_path)
    feature_names = list(module.feature_names)
    input_dims = list(module.input_feature_dim)
    n = len(feature_names)

    # A checkpoint saved from an MPS/CUDA run keeps its weights on that
    # device via load_from_checkpoint (unlike FiPredictor.from_checkpoint,
    # which passes map_location="cpu" explicitly) -- ONNX export traces on
    # CPU tensors, so the wrapped model must be moved there regardless of
    # where its checkpoint was trained.
    if module.is_lint:
        wrapper: nn.Module = _LinTExportWrapper(
            module.model, name=feature_names[0], head_name=HEAD_NAME
        )
    else:
        wrapper = _LinMulTExportWrapper(module.model, names=feature_names, head_name=HEAD_NAME)
    wrapper.eval()
    wrapper.to("cpu")

    # A representative (batch=1, T=50) input for tracing; T is dynamic (see
    # dynamic_axes below), so its exact value here does not fix the exported
    # graph's shape.
    dummy_inputs = [torch.randn(1, 50, dim) for dim in input_dims]

    if module.is_lint:
        # LinT's forward nulls the mask only when it is all-False
        # (`if mask is not None and not mask.any()`), so an all-True dummy
        # mask is still "not null" at trace time and stays a live graph
        # input -- this has always worked for the LinT path.
        dummy_masks = [torch.ones(1, 50, dtype=torch.bool)]
        trace_args: tuple[torch.Tensor, ...] = (dummy_inputs[0], dummy_masks[0])
        input_names = ["x", "mask"]
        dynamic_axes = {
            "x": {0: "batch", 1: "time"},
            "mask": {0: "batch", 1: "time"},
            "predictions": {0: "batch"},
        }
    else:
        # LinMulT's `_normalize_masks` does the *opposite*: it nulls a mask
        # only when it is all-True ("nothing to mask, skip the overhead"),
        # collapsing it to None *individually, per feature* before it
        # reaches any op that would consume it as a tensor. Any dummy mask
        # that is all-True gets traced away as an unused graph input
        # entirely (torch.onnx.export drops inputs the traced computation
        # never actually touches) -- a real caller later passing a mask for
        # that feature would hit "invalid input name" at session.run time.
        #
        # A single False *timestep* is not enough either: linmult.core
        # .transformer's `TransformerLayer.forward` has a second, coarser
        # data-dependent branch -- `fully_masked_keys = ~key_mask.any(dim=1)`,
        # `if fully_masked_keys.any(): ... torch.where(...)` -- that only
        # fires when a whole *sample* (not just a timestep) has every key
        # masked. Tracing with a merely-partial mask never triggers that
        # `if`, so the zero-out `torch.where` guard gets traced away, and a
        # real all-False mask fed to the exported graph later silently skips
        # it -- caught by test_onnx_all_false_text_masks_match_pytorch as a
        # real numeric mismatch (not NaN -- LinearAttention itself doesn't
        # NaN on this input -- but genuinely wrong values), not a
        # hypothetical.
        #
        # Both constraints together mean every dummy mask must be all-False
        # at trace time: this keeps every mask_i input alive (none collapse
        # to None) AND exercises the fully-masked-branch guard for every
        # feature simultaneously, so the guard survives into the exported
        # graph and applies correctly regardless of which real feature(s) a
        # caller masks out later -- verified against real, varied mask
        # patterns (all-True, one feature all-False, mixed) in
        # test_onnx_export.py, not assumed safe from tracing alone.
        dummy_masks = [torch.zeros(1, 50, dtype=torch.bool) for _ in feature_names]
        trace_args = (*dummy_inputs, *dummy_masks)
        input_names = [f"x_{i}" for i in range(n)] + [f"mask_{i}" for i in range(n)]
        # Each feature gets its OWN symbolic time-axis name ("time_0",
        # "time_1", ...), not a single shared "time" for all 14 inputs.
        # ONNX's dynamic_axes convention treats identically-named symbolic
        # dimensions as "must be equal at runtime" -- a shared name across
        # every feature would (and, caught by predict_from_media's real
        # end-to-end test, DID) reject the model's actual, architecturally
        # correct input at inference time: each of the 7 AVT features has
        # its own independent time length (WavLM ~50 Hz, DINOv2 ~25 fps,
        # text one row per token, ...), exactly like training reads them
        # (personalitylinmult/data/dataset.py windows each feature from its
        # own rate_hz, never a shared T). x_i and mask_i share one symbol
        # (their T must agree with each other), but different i's do not.
        dynamic_axes = {}
        for i in range(n):
            dynamic_axes[f"x_{i}"] = {0: "batch", 1: f"time_{i}"}
            dynamic_axes[f"mask_{i}"] = {0: "batch", 1: f"time_{i}"}
        dynamic_axes["predictions"] = {0: "batch"}

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            trace_args,
            str(output_path),
            input_names=input_names,
            output_names=["predictions"],
            dynamic_axes=dynamic_axes,
            opset_version=ONNX_OPSET,
            # Both LinT and LinMulT branch on a mask tensor's *content*
            # (LinT: `if mask is not None and not mask.any()`; LinMulT's
            # `_normalize_masks`: `mask.all()`), which the new torch.export-
            # based exporter (dynamo=True, torch's own default since 2.11)
            # cannot trace -- it requires a fully static graph. The legacy
            # TorchScript tracer (dynamo=False) handles data-dependent
            # control flow like this correctly.
            dynamo=False,
        )
    logger.info(f"Exported ONNX graph ({module.model_config.family}) to {output_path}.")

    metadata = {
        "feature_names": feature_names,
        "input_dims": input_dims,
        "traits": list(module.train_config.targets),
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2))
    logger.info(f"Wrote ONNX metadata to {metadata_path}.")

    return output_path


WAVLM_MODEL_ID = "microsoft/wavlm-base-plus"
"""HF model id matching ``WavlmWrapper(model_name="base+")``, the variant
every feature in this project's ``fi.h5`` was extracted with."""

WAVLM_LAYER = 9
"""Which transformer layer's hidden state to export -- matches
``personalitylinmult.inference.audio_features.WAVLM_LAYER``. Indexes into
``WavlmWrapper``'s already-sliced ``hidden_states[1:]`` (12 transformer
layers, CNN feature-extractor output excluded), so the raw HF model's
``hidden_states[WAVLM_LAYER + 1]`` is the equivalent index -- verified
numerically equal to ``WavlmWrapper(...)(waveform)[WAVLM_LAYER]`` before
this export was written."""


class _WavLMLayerWrapper(nn.Module):
    """Runs WavLM and returns exactly one hidden-state layer.

    A raw ``transformers.WavLMModel`` returns a
    ``Wav2Vec2BaseModelOutput`` (a dataclass of several optional tensors);
    ONNX/onnxruntime need a single, fixed tensor output, so this selects the
    one layer this project actually uses.
    """

    def __init__(self, model: nn.Module, layer_index: int):
        super().__init__()
        self.model = model
        self._layer_index = layer_index

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """Run WavLM and select one hidden-state layer.

        Args:
            input_values (torch.Tensor): ``(B, samples)`` mono waveform at
                16 kHz.

        Returns:
            torch.Tensor: ``(B, T, 768)`` hidden states from the configured
            layer.
        """
        outputs = self.model(input_values=input_values, output_hidden_states=True)
        return outputs.hidden_states[self._layer_index]


def export_wavlm_to_onnx(
    output_path: str | Path,
    model_id: str = WAVLM_MODEL_ID,
    layer_index: int = WAVLM_LAYER + 1,
) -> Path:
    """Export WavLM (one hidden-state layer) to ONNX.

    Downloads the model from HF Hub (``transformers``) once, at export time
    -- the resulting ``.onnx`` graph is self-contained and needs neither
    ``transformers`` nor a network connection to run afterward.

    Args:
        output_path (str | Path): Destination ``.onnx`` path.
        model_id (str): HF model id to export. Defaults to
            :data:`WAVLM_MODEL_ID` (WavLM base+, matching every feature this
            project's ``fi.h5`` was built with).
        layer_index (int): Index into the *raw* model's ``hidden_states``
            tuple (13 entries: CNN output + 12 transformer layers). Defaults
            to :data:`WAVLM_LAYER` ``+ 1``, the raw-model equivalent of
            :data:`WAVLM_LAYER` in ``WavlmWrapper``'s already-sliced list --
            see :data:`WAVLM_LAYER`'s docstring for how that offset was
            confirmed.

    Returns:
        Path: The written ``.onnx`` path.
    """
    from transformers import WavLMModel

    output_path = Path(output_path)
    model = WavLMModel.from_pretrained(model_id)
    model.eval()

    wrapper = _WavLMLayerWrapper(model, layer_index=layer_index)
    wrapper.eval()

    # A representative ~1s waveform for tracing; both batch and sample count
    # are dynamic (see dynamic_axes below).
    dummy_input = torch.randn(1, 16000)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_input,),
            str(output_path),
            input_names=["input_values"],
            output_names=["hidden_state"],
            dynamic_axes={
                "input_values": {0: "batch", 1: "samples"},
                "hidden_state": {0: "batch", 1: "time"},
            },
            opset_version=ONNX_OPSET,
            dynamo=False,
        )
    logger.info(f"Exported WavLM ({model_id}, layer {layer_index}) ONNX graph to {output_path}.")
    return output_path


DINOV2_MODEL_ID = "facebook/dinov2-base"
"""HF model id matching ``DINOv2Wrapper(model_name="base")``, the variant
this project's ``fi.h5`` was built with (768-d)."""


class _DINOv2Wrapper(nn.Module):
    """Runs DINOv2 and returns the L2-normalised CLS token.

    Replicates :meth:`exordium.video.deep.dinov2.DINOv2Wrapper.inference`
    exactly -- a raw-forward-only export that omitted the CLS-token
    selection or the L2-normalize step would silently produce wrong
    (unnormalized, or pooled-over-all-tokens) features, not just a slightly
    different graph.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run DINOv2 and select+normalize the CLS token.

        Args:
            pixel_values (torch.Tensor): ``(B, 3, 224, 224)`` preprocessed
                (resized + ImageNet-normalized) frame tensor -- preprocessing
                itself stays outside the graph, matching WavLM's export
                (the ffmpeg decode step stays outside that graph too).

        Returns:
            torch.Tensor: ``(B, 768)`` L2-normalized CLS-token embedding.
        """
        outputs = self.model(pixel_values=pixel_values)
        cls = outputs.last_hidden_state[:, 0, :]
        return cls / cls.norm(dim=-1, keepdim=True)


def export_dinov2_to_onnx(output_path: str | Path, model_id: str = DINOV2_MODEL_ID) -> Path:
    """Export DINOv2 (CLS token, L2-normalized) to ONNX.

    Downloads the model from HF Hub (``transformers``) once, at export time
    -- the resulting ``.onnx`` graph is self-contained and needs neither
    ``transformers`` nor a network connection to run afterward. Unlike
    WavLM (one call per whole clip), a live pipeline calls this graph once
    per video frame (or batches several frames into one call via the
    dynamic batch axis) -- DINOv2 runs on full, unmodified frames, no face
    detection or tracking involved (see
    :mod:`personalitylinmult.inference.audio_features_onnx`'s WavLM-decode
    pattern for the analogous "preprocessing stays outside the graph"
    convention this export also follows).

    Args:
        output_path (str | Path): Destination ``.onnx`` path.
        model_id (str): HF model id to export. Defaults to
            :data:`DINOV2_MODEL_ID` (DINOv2 base, matching every feature this
            project's ``fi.h5`` was built with).

    Returns:
        Path: The written ``.onnx`` path.
    """
    from transformers import Dinov2Model

    output_path = Path(output_path)
    model = Dinov2Model.from_pretrained(model_id)
    model.eval()

    wrapper = _DINOv2Wrapper(model)
    wrapper.eval()

    # A representative frame batch for tracing; batch is dynamic (see
    # dynamic_axes below). 224x224 is DINOv2's fixed input resolution, not
    # dynamic -- unlike WavLM's sample count or LinT/LinMulT's time axis,
    # a ViT's patch embedding is defined for one input size.
    dummy_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_input,),
            str(output_path),
            input_names=["pixel_values"],
            output_names=["cls_embedding"],
            dynamic_axes={
                "pixel_values": {0: "batch"},
                "cls_embedding": {0: "batch"},
            },
            opset_version=ONNX_OPSET,
            dynamo=False,
        )
    logger.info(f"Exported DINOv2 ({model_id}) ONNX graph to {output_path}.")
    return output_path


MMBERT_MODEL_ID = "jhu-clsp/mmBERT-base"
"""HF model id matching ``MmbertWrapper()``, the variant this project's
``fi.h5`` (``mmbert_gt`` feature) was built with."""

XLM_ROBERTA_MODEL_ID = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
"""HF model id matching ``XlmRobertaWrapper()``'s default ``backbone="mpnet"``
-- confirmed (not assumed) to be what this project's ``fi.h5``
(``xlm_roberta_gt`` feature) was built with by reading
``personalitylinmult/preprocess/fi/textual_xlm_roberta.py``, which calls
``XlmRobertaWrapper(device_id=device_id)`` with no ``backbone=`` override, so
the wrapper's default applies. Not ``"e5"`` (``intfloat/multilingual-e5-base``),
the wrapper's other supported backbone."""


class _TokenLevelTextWrapper(nn.Module):
    """Runs a text encoder and returns its token-level hidden states.

    Replicates ``PooledTokenTextWrapper.inference(..., pooling="none")``
    exactly -- ``last_hidden_state``, no pooling. Shared by both the mmBERT
    and XLM-RoBERTa exports since both wrappers subclass
    ``PooledTokenTextWrapper`` and use this exact same forward contract;
    only the underlying HF model id differs.

    Tokenization stays outside the graph (tokenizers are not cleanly
    traceable the way a model forward is) -- the live pipeline calls
    ``transformers.AutoTokenizer`` itself and feeds ``input_ids``/
    ``attention_mask`` into this graph directly, the same way WavLM's
    export leaves ffmpeg decoding outside its graph.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Run the encoder and return token-level hidden states.

        Args:
            input_ids (torch.Tensor): ``(B, T)`` int64 token ids.
            attention_mask (torch.Tensor): ``(B, T)`` int64, ``1`` = real
                token, ``0`` = padding -- the tokenizer's own convention, the
                opposite dtype (not semantic) of LinMulT's ``(B, T)`` bool
                mask. The live pipeline must cast this to bool before
                passing it to the trained model's ONNX graph; this export's
                own graph still takes the tokenizer's native int64 form,
                since that is what ``AutoModel.forward`` itself expects.

        Returns:
            torch.Tensor: ``(B, T, 768)`` token-level hidden states.
        """
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


def _export_text_encoder_to_onnx(output_path: str | Path, model_id: str, log_label: str) -> Path:
    """Shared export body for mmBERT and XLM-RoBERTa.

    Both are token-level, ``AutoModel``-based, and share an identical
    forward contract -- see :class:`_TokenLevelTextWrapper`.
    """
    from transformers import AutoModel

    output_path = Path(output_path)
    model = AutoModel.from_pretrained(model_id)
    model.eval()

    wrapper = _TokenLevelTextWrapper(model)
    wrapper.eval()

    # A representative short token sequence for tracing; T is dynamic (see
    # dynamic_axes below).
    dummy_input_ids = torch.randint(0, 1000, (1, 12), dtype=torch.int64)
    dummy_attention_mask = torch.ones(1, 12, dtype=torch.int64)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_input_ids, dummy_attention_mask),
            str(output_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "time"},
                "attention_mask": {0: "batch", 1: "time"},
                "last_hidden_state": {0: "batch", 1: "time"},
            },
            opset_version=ONNX_OPSET,
            dynamo=False,
        )
    logger.info(f"Exported {log_label} ({model_id}) ONNX graph to {output_path}.")
    return output_path


def export_mmbert_to_onnx(output_path: str | Path, model_id: str = MMBERT_MODEL_ID) -> Path:
    """Export mmBERT (token-level hidden states) to ONNX.

    Downloads the model from HF Hub (``transformers``) once, at export time.
    Tokenization stays outside the graph -- see :class:`_TokenLevelTextWrapper`.

    Args:
        output_path (str | Path): Destination ``.onnx`` path.
        model_id (str): HF model id to export. Defaults to
            :data:`MMBERT_MODEL_ID`.

    Returns:
        Path: The written ``.onnx`` path.
    """
    return _export_text_encoder_to_onnx(output_path, model_id, log_label="mmBERT")


def export_xlm_roberta_to_onnx(
    output_path: str | Path, model_id: str = XLM_ROBERTA_MODEL_ID
) -> Path:
    """Export XLM-RoBERTa (``mpnet`` backbone, token-level) to ONNX.

    Downloads the model from HF Hub (``transformers``) once, at export time.
    Tokenization stays outside the graph -- see :class:`_TokenLevelTextWrapper`.

    Args:
        output_path (str | Path): Destination ``.onnx`` path.
        model_id (str): HF model id to export. Defaults to
            :data:`XLM_ROBERTA_MODEL_ID` (the ``mpnet`` backbone, confirmed to
            be what this project's ``xlm_roberta_gt`` feature actually uses).

    Returns:
        Path: The written ``.onnx`` path.
    """
    return _export_text_encoder_to_onnx(output_path, model_id, log_label="XLM-RoBERTa")


EMOTIEFFNET_MODEL_NAME = "enet_b0_8_best_vgaf"
"""Variant matching ``EmotiEffNetWrapper()``'s default -- ``timm``
``tf_efficientnet_b0.ns_jft_in1k``, 224x224 input, 1280-d penultimate
features. Matches every feature this project's ``fi.h5`` was built with."""


class _PassthroughWrapper(nn.Module):
    """Traces an already-fully-built model's forward as-is.

    Used for extractors whose ``exordium`` wrapper does non-trivial,
    hard-to-replicate setup before its model is ready to run (EmotiEffNet's
    legacy-pickle state_dict remapping; FaRL's OpenAI-CLIP-to-HF weight
    conversion) -- rather than re-implementing that setup here, this
    instantiates the real wrapper class at export time (an accepted,
    export-time-only dependency, the same precedent WavLM's own export set:
    a heavy `transformers` download at export time, never at inference time)
    and traces its already-correctly-loaded ``.model`` attribute directly.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run the wrapped model's forward pass unmodified.

        Args:
            pixel_values (torch.Tensor): ``(B, 3, H, H)`` preprocessed frame
                tensor, ``H`` matching the wrapped model's expected input
                resolution.

        Returns:
            torch.Tensor: ``(B, D)`` feature tensor, ``D`` matching the
            wrapped model's output width.
        """
        return self.model(pixel_values)


def export_emotieffnet_to_onnx(
    output_path: str | Path, model_name: str = EMOTIEFFNET_MODEL_NAME
) -> Path:
    """Export EmotiEffNet (penultimate-layer features) to ONNX.

    Instantiates the real :class:`exordium.video.deep.emotieffnet
    .EmotiEffNetWrapper` at export time (downloads and correctly remaps its
    legacy-pickle checkpoint -- see that wrapper's module docstring for why
    a fresh ``timm`` model plus a remapped ``state_dict`` is needed rather
    than loading the pickle directly) and traces its ``.model`` -- already
    has ``classifier = nn.Identity()`` applied, so the traced graph itself
    needs no special output selection, unlike WavLM (must select one
    ``hidden_states`` layer) or DINOv2 (must select+normalize the CLS token).

    Like DINOv2, a live pipeline calls this graph once per face crop (or
    batches several into one call via the dynamic batch axis); preprocessing
    (resize + ImageNet normalize) stays outside the graph.

    Args:
        output_path (str | Path): Destination ``.onnx`` path.
        model_name (str): EmotiEffNet variant to export. Defaults to
            :data:`EMOTIEFFNET_MODEL_NAME` (matching every feature this
            project's ``fi.h5`` was built with).

    Returns:
        Path: The written ``.onnx`` path.
    """
    from exordium.video.deep.emotieffnet import EmotiEffNetWrapper

    output_path = Path(output_path)
    exordium_wrapper = EmotiEffNetWrapper(model_name=model_name, device_id=-1)
    img_size = exordium_wrapper.img_size

    wrapper = _PassthroughWrapper(exordium_wrapper.model)
    wrapper.eval()

    # A representative frame batch for tracing; batch is dynamic (see
    # dynamic_axes below). The input resolution itself is fixed per variant
    # (224 or 260), not dynamic -- same reasoning as DINOv2's export.
    dummy_input = torch.randn(1, 3, img_size, img_size)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_input,),
            str(output_path),
            input_names=["pixel_values"],
            output_names=["features"],
            dynamic_axes={
                "pixel_values": {0: "batch"},
                "features": {0: "batch"},
            },
            opset_version=ONNX_OPSET,
            dynamo=False,
        )
    logger.info(f"Exported EmotiEffNet ({model_name}) ONNX graph to {output_path}.")
    return output_path


FARL_MODEL_NAME = "ep64"
"""Variant matching ``FarlWrapper()``'s default -- CLIP ViT-B/16, LAION-Face
20M, 64 epochs, 512-d projected embedding. Matches every feature this
project's ``fi.h5`` was built with."""


class _FaRLWrapper(nn.Module):
    """Runs FaRL and returns the L2-normalised projected embedding.

    Replicates :meth:`exordium.video.deep.farl.FarlWrapper.inference`
    exactly (select ``image_embeds``, then L2-normalize) -- a
    raw-forward-only export that omitted either step would silently produce
    wrong (unprojected, or unnormalized) features.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run FaRL and select+normalize the projected embedding.

        Args:
            pixel_values (torch.Tensor): ``(B, 3, 224, 224)`` preprocessed
                (resized + CLIP-normalized, not ImageNet) face-crop tensor.

        Returns:
            torch.Tensor: ``(B, 512)`` L2-normalized embedding.
        """
        embeds = self.model(pixel_values=pixel_values).image_embeds
        return embeds / embeds.norm(dim=-1, keepdim=True)


def export_farl_to_onnx(output_path: str | Path, model_name: str = FARL_MODEL_NAME) -> Path:
    """Export FaRL (L2-normalized projected embedding) to ONNX.

    Instantiates the real :class:`exordium.video.deep.farl.FarlWrapper` at
    export time -- FaRL ships raw OpenAI-CLIP weights with no HF Hub repo to
    pull an architecture from, so the wrapper's own
    ``_build_vision_tower()`` + ``_convert_openai_to_hf(...)`` construction
    is reused rather than re-implemented here (the same export-time-only
    heavy-dependency precedent WavLM's export set: this only runs once, at
    export time, never at inference time).

    Like DINOv2 and EmotiEffNet, a live pipeline calls this graph once per
    face crop (or batches several into one call via the dynamic batch
    axis); preprocessing (resize + **CLIP** mean/std -- not ImageNet's --
    normalize) stays outside the graph.

    Args:
        output_path (str | Path): Destination ``.onnx`` path.
        model_name (str): FaRL variant to export. Defaults to
            :data:`FARL_MODEL_NAME` (matching every feature this project's
            ``fi.h5`` was built with).

    Returns:
        Path: The written ``.onnx`` path.
    """
    from exordium.video.deep.farl import FarlWrapper

    output_path = Path(output_path)
    exordium_wrapper = FarlWrapper(model_name=model_name, device_id=-1)
    img_size = exordium_wrapper.img_size

    wrapper = _FaRLWrapper(exordium_wrapper.model)
    wrapper.eval()

    # A representative frame batch for tracing; batch is dynamic (see
    # dynamic_axes below). 224x224 is FaRL's fixed input resolution.
    dummy_input = torch.randn(1, 3, img_size, img_size)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_input,),
            str(output_path),
            input_names=["pixel_values"],
            output_names=["embedding"],
            dynamic_axes={
                "pixel_values": {0: "batch"},
                "embedding": {0: "batch"},
            },
            opset_version=ONNX_OPSET,
            dynamo=False,
        )
    logger.info(f"Exported FaRL ({model_name}) ONNX graph to {output_path}.")
    return output_path


class _Emotion2vecExportWrapper(nn.Module):
    """Runs emotion2vec's ``extract_features`` with ``padding_mask=None``.

    ``Emotion2vecWrapper.__call__`` (the live single-clip inference path this
    export mirrors) never passes a ``padding_mask`` -- only the batch APIs
    that align multiple different-length clips into one padded tensor do.
    Exposing ``padding_mask`` as a graph input would require tracing
    ``_compute_padding_mask``'s own data-dependent branches
    (``if not mask.any(): return None``) for no benefit this project's
    single-clip pipeline ever uses, so it is fixed at ``None`` here instead
    -- the same "trace only the code path a real caller takes" choice
    ``_LinTExportWrapper``/``_LinMulTExportWrapper`` make for their own
    mask-content branches, just resolved by omitting the argument entirely
    rather than by choosing a specific dummy value.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        """Run emotion2vec and return its frame-level feature tensor.

        Args:
            source (torch.Tensor): ``(B, samples)`` mono waveform at 16 kHz.

        Returns:
            torch.Tensor: ``(B, T, 768)`` frame-level features (~50 fps).
        """
        # self.model.extract_features(...) is an attribute access (unlike
        # every other wrapper in this file, which calls plain
        # self.model(...), i.e. __call__) resolved through
        # nn.Module.__getattr__'s stub, typed Tensor | Module -- ty cannot
        # statically know `model` is a submodule and not a registered
        # parameter/buffer of the same name, so the union survives even a
        # locally re-annotated variable.
        return self.model.extract_features(source, padding_mask=None)["x"]  # ty: ignore[call-non-callable]


def export_emotion2vec_to_onnx(output_path: str | Path) -> Path:
    """Export emotion2vec+ (frame-level features) to ONNX.

    Instantiates the real :class:`exordium.audio.emotion2vec
    .Emotion2vecWrapper` at export time -- emotion2vec+ is a fully custom,
    vendored architecture (``_Emotion2vecModel``, not a thin wrapper over a
    library model), so its own checkpoint download and key-remapping logic
    is reused rather than re-implemented here.

    ALiBi's relative position bias (``_get_alibi_bias``) is recomputed from
    the input's time dimension on every call, using ``torch.arange`` over a
    shape-derived count -- traced correctly under a dynamic time axis (see
    ``test_onnx_export_emotion2vec.py``'s explicit different-length check,
    added specifically because a shape-dependent internal computation like
    this is a plausible way for the tracer to silently bake in a constant
    instead of staying dynamic, the same class of risk already documented
    for LinMulT's mask-content branching in this module's docstring).

    Args:
        output_path (str | Path): Destination ``.onnx`` path.

    Returns:
        Path: The written ``.onnx`` path.
    """
    from exordium.audio.emotion2vec import Emotion2vecWrapper

    output_path = Path(output_path)
    exordium_wrapper = Emotion2vecWrapper(device_id=-1)

    wrapper = _Emotion2vecExportWrapper(exordium_wrapper.model)
    wrapper.eval()

    # A representative ~1s waveform for tracing; both batch and sample count
    # are dynamic (see dynamic_axes below) -- same convention as WavLM's
    # export.
    dummy_input = torch.randn(1, 16000)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_input,),
            str(output_path),
            input_names=["source"],
            output_names=["features"],
            dynamic_axes={
                "source": {0: "batch", 1: "samples"},
                "features": {0: "batch", 1: "time"},
            },
            opset_version=ONNX_OPSET,
            dynamo=False,
        )
    logger.info(f"Exported emotion2vec+ ONNX graph to {output_path}.")
    return output_path
