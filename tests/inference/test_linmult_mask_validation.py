"""Validates LinMulT's all-False whole-modality mask path before it is relied on.

`avt_best-ccc`'s live-serving pipeline (planned) needs a "skip text entirely"
mode: a placeholder tensor with an all-False mask for a whole modality, not
just a partially-padded one. No caller in this project has ever built one
before, so this is verified with a real forward pass -- not assumed from
reading `LinMulT._normalize_masks`'s source -- before any ONNX export work is
built around it.

Real finding, read directly from the `linmult` package source: an all-False
mask is passed through `_normalize_masks` unmodified (only an all-*True* mask
collapses to `None`). The initial hypothesis going into this test was that
`SoftmaxAttention` (used by `attention_type="softmax"`, the MulT baseline)
would NaN on an all-False mask via an additive `-inf` bias row, while
`LinearAttention` (this project's default, `attention_type="linear"`, used by
`avt_best-ccc`) would not, since it masks by zeroing Q/K directly rather than
building an additive bias.

**That hypothesis was wrong for this codebase specifically** -- running both
configurations shows neither NaNs. Reading `linmult.core.transformer`'s
cross-modal/self-attention block (`TransformerLayer.forward`) explains why:
it has an explicit, deliberate guard for exactly this case, independent of
attention type -- `fully_masked_keys = ~key_mask.any(dim=1)` catches a
whole-branch-masked sample and overwrites it with zeros via `torch.where`
*before* returning, and query-side padding is zeroed with `masked_fill` (not
multiplication, since `NaN * 0 = NaN` under IEEE 754) rather than left to
propagate through softmax. Both attention types are therefore NaN-safe for a
whole-modality-missing mask in this specific `linmult` version -- this is
verified with a real forward pass below for both, not assumed from reading
`_normalize_masks` alone.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from personalitylinmult.train.config import ModelConfig, TrainConfig
from personalitylinmult.train.module import FiLightningModule

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "emotional_stability"]
FEATURES = ["wavlm", "emotion2vec", "dinov2", "emotieffnet", "farl", "xlm_roberta_gt", "mmbert_gt"]
TEXT_FEATURES = ("xlm_roberta_gt", "mmbert_gt")


def write_linmult_checkpoint(
    path: Path, attention_type: str = "linear", dims: list[int] | None = None
) -> None:
    """Build a tiny 7-feature LinMulT checkpoint, matching avt_best-ccc's shape.

    Args:
        path (Path): Where to save the checkpoint.
        attention_type (str): ``"linear"`` (the real champion's setting) or
            ``"softmax"`` (the deliberate negative-case counterexample).
        dims (list[int] | None): Per-feature input width, in :data:`FEATURES`
            order. ``None`` defaults to ``[8] * 7`` (fast, tiny -- the right
            choice for a mask-validation-only test that never feeds real
            extractor output through this checkpoint). Pass real per-
            extractor widths (e.g. 768 for the text/audio features, 1280 for
            EmotiEffNet, 512 for FaRL) when testing against actual ONNX
            extractor output, which is a fixed width the checkpoint's input
            layer must match exactly, unlike the toy 8-d case.
    """
    import lightning as L

    model_config = ModelConfig(
        family="linmult",
        d_model=8,
        num_heads=2,
        cmt_num_layers=1,
        branch_sat_num_layers=1,
        attention_type=attention_type,
        # time_dim_reducer defaults to "gap", matching avt_best-ccc.
    )
    train_config = TrainConfig(task="bigfive", targets=TRAITS)
    module = FiLightningModule(
        model_config,
        train_config,
        feature_names=FEATURES,
        input_feature_dim=dims if dims is not None else [8] * len(FEATURES),
    )
    trainer = L.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False)
    trainer.strategy.connect(module)
    trainer.save_checkpoint(str(path))


def _build_inputs_and_masks(
    seed: int, text_mask_value: bool, text_fill: float | None = None
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Build (inputs, masks) for all 7 features, with text masks overridable.

    Args:
        seed (int): RNG seed for the random (non-text) feature content.
        text_mask_value (bool): Mask value for both text features -- ``True``
            (normal) or ``False`` (whole-modality-missing).
        text_fill (float | None): If given, fills the text features'
            *content* with this constant instead of random noise -- used to
            prove the mask, not the content, is what determines the output
            when ``text_mask_value=False``.

    Returns:
        tuple: ``(inputs, masks)``, one ``(1, 20, 8)`` tensor / ``(1, 20)``
        bool mask per feature, in :data:`FEATURES` order.
    """
    generator = torch.Generator().manual_seed(seed)
    inputs: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for name in FEATURES:
        if name in TEXT_FEATURES:
            if text_fill is None:
                tensor = torch.randn(1, 20, 8, generator=generator)
            else:
                tensor = torch.full((1, 20, 8), text_fill)
            mask = torch.full((1, 20), text_mask_value, dtype=torch.bool)
        else:
            tensor = torch.randn(1, 20, 8, generator=generator)
            mask = torch.ones(1, 20, dtype=torch.bool)
        inputs.append(tensor)
        masks.append(mask)
    return inputs, masks


class TestLinearAttentionAllFalseTextMask(unittest.TestCase):
    """The real champion's configuration (attention_type='linear')."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        ckpt = Path(self._tmp.name) / "linmult.ckpt"
        write_linmult_checkpoint(ckpt, attention_type="linear")
        self.module = FiLightningModule.load_from_checkpoint(str(ckpt)).eval()

    def test_all_true_masks_produce_no_nan(self):
        inputs, masks = _build_inputs_and_masks(seed=0, text_mask_value=True)
        with torch.no_grad():
            output = self.module(inputs, masks)
        self.assertFalse(torch.isnan(output).any().item())
        self.assertFalse(torch.isinf(output).any().item())

    def test_all_false_text_masks_produce_no_nan(self):
        inputs, masks = _build_inputs_and_masks(seed=0, text_mask_value=False)
        with torch.no_grad():
            output = self.module(inputs, masks)
        self.assertFalse(torch.isnan(output).any().item(), "linear attention should not NaN")
        self.assertFalse(torch.isinf(output).any().item())
        # Sane range -- LinMulT's raw head output isn't bounded to [0, 1] but
        # should not blow up either.
        self.assertTrue(torch.all(output.abs() < 10.0).item())

    def test_all_false_mask_output_differs_from_all_true(self):
        """A masked-out modality must actually change the prediction --
        otherwise the mask isn't doing anything and this whole mechanism
        would be silently inert."""
        inputs, true_masks = _build_inputs_and_masks(seed=0, text_mask_value=True)
        _, false_masks = _build_inputs_and_masks(seed=0, text_mask_value=False)
        with torch.no_grad():
            out_true = self.module(inputs, true_masks)
            out_false = self.module(inputs, false_masks)
        self.assertFalse(torch.allclose(out_true, out_false))

    def test_all_false_mask_output_is_independent_of_placeholder_content(self):
        """When masked all-False, the *content* of the text tensors must not
        matter -- only the mask does. Compares zero-filled vs. garbage-filled
        placeholders under an identical all-False mask."""
        inputs_zero, masks = _build_inputs_and_masks(seed=0, text_mask_value=False, text_fill=0.0)
        inputs_garbage, _ = _build_inputs_and_masks(seed=0, text_mask_value=False, text_fill=999.0)
        with torch.no_grad():
            out_zero = self.module(inputs_zero, masks)
            out_garbage = self.module(inputs_garbage, masks)
        torch.testing.assert_close(out_zero, out_garbage, atol=1e-5, rtol=1e-5)


class TestSoftmaxAttentionAllFalseTextMask(unittest.TestCase):
    """Not this project's configuration (avt_best-ccc uses attention_type=
    "linear"), but checked anyway: softmax attention was hypothesized to NaN
    on an all-False mask, and does NOT -- linmult.core.transformer's
    fully-masked-branch guard (see module docstring) applies regardless of
    attention type. Kept as a regression test: if a future linmult upgrade
    removes that guard, this is what catches softmax silently starting to
    NaN, rather than assuming the guard is permanent."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        ckpt = Path(self._tmp.name) / "linmult_softmax.ckpt"
        write_linmult_checkpoint(ckpt, attention_type="softmax")
        self.module = FiLightningModule.load_from_checkpoint(str(ckpt)).eval()

    def test_all_false_text_masks_do_not_produce_nan(self):
        inputs, masks = _build_inputs_and_masks(seed=0, text_mask_value=False)
        with torch.no_grad():
            output = self.module(inputs, masks)
        self.assertFalse(
            torch.isnan(output).any().item(),
            "softmax attention NaN'd on an all-False mask -- the "
            "fully-masked-branch guard in linmult.core.transformer appears "
            'to have regressed; re-audit before trusting transcript="" '
            "(skip-text) mode in predict_from_media.",
        )


if __name__ == "__main__":
    unittest.main()
