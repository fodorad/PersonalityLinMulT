
# PersonalityLinMulT — First Impressions V2 extracted features (`fi.h5`)

<!--
  This is the source of truth for the Hugging Face dataset card at
  https://huggingface.co/datasets/fodorad/personalitylinmult-fi
  Keep the two in sync (see `make push-hf-card`). The published card adds a YAML
  frontmatter block (license/tags/size) that HF renders; this docs copy omits it.
-->

Pre-extracted multimodal features for the **First Impressions V2 (FI)** dataset,
packaged as a single HDF5 file for training the models in
[**PersonalityLinMulT**](https://github.com/fodorad/PersonalityLinMulT):
Big Five personality perception and interview-variable (hireability) estimation.

> ⚠️ **These are derived features (embeddings), not the source videos.**
> **No raw ChaLearn First Impressions video, audio, or frames are re-shared
> here.** This file contains only numerical feature vectors extracted from the
> official dataset. To obtain the raw videos, register for and download the
> dataset from the [official ChaLearn LAP
> release](https://chalearnlap.cvc.uab.cat/dataset/24/description/) under its
> own license terms.

## What this is for

Training reads this file **and nothing else** — it is the reproducibility
boundary for the project. Pull it and train immediately, with no need to obtain
the raw videos or re-run the multi-hour GPU feature extraction:

```bash
# In a clone of https://github.com/fodorad/PersonalityLinMulT
make pull-fi-h5        # downloads fi.h5 into data/processed/FI/h5/
make train-fi-bigfive  # train Big Five regression
```

Or download directly:

```python
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    "fodorad/personalitylinmult-fi", "fi.h5", repo_type="dataset"
)
```

## Contents

`fi.h5` — 10 000 clips (6000 train / 2000 valid / 2000 test), ~33 GB, `float16`.
Row-indexed and feature-major: `__getitem__(i)` reads exactly one chunk per
feature. Every sequence feature carries a `mask` (`True` = the protagonist was
detected / a real token) and a `length`.

| feature | modality | source model | shape `(N, T, D)` |
|---|---|---|---|
| `wavlm` | acoustic | WavLM Base+ (layer 9) | `(10000, 750, 768)` |
| `emotion2vec` | acoustic | emotion2vec+ | `(10000, 750, 768)` |
| `dinov2` | visual | DINOv2 Base (full frame) | `(10000, 375, 768)` |
| `emotieffnet` | visual | EmotiEffNet (EfficientNet-B0) | `(10000, 375, 1280)` |
| `farl` | visual | FaRL (ViT-B/16) | `(10000, 375, 512)` |
| `xlm_roberta_gt` | textual | XLM-RoBERTa tokens, ground-truth | `(10000, 128, 768)` |
| `mmbert_gt` | textual | mmBERT tokens, ground-truth | `(10000, 128, 768)` |

Visual features (`emotieffnet`, `farl`) track the **protagonist's** face; the
mask marks frames where the protagonist was detected. `dinov2` is dense (full
frame). Text is **token-level** `(T, 768)` (not pooled) for both XLM-RoBERTa and
mmBERT — the representation a cross-modal sequence model attends over — padded to
128 tokens with the mask marking real tokens. Both are extracted over the
**ground-truth** transcript.

Labels stored per clip: the Big Five traits (`openness`, `conscientiousness`,
`extraversion`, `agreeableness`, `neuroticism`, and `emotional_stability`
`= 1 − neuroticism`) plus the `interview` variable, all in `[0, 1]`.

## Provenance and reproducibility

The file's HDF5 root attributes embed the **builder config verbatim**, the
**git commit SHA** of the code that produced it (with a `-dirty` suffix if built
from an uncommitted tree), and the source label CSV's SHA-256 — so any copy
traces back to the exact recipe. To rebuild from scratch and verify:

```bash
make pipeline-fi   # decode → extract features → build fi.h5 (requires raw FI)
```

## Citation

Features derived from the **ChaLearn First Impressions V2** dataset. If you use
this artifact, please cite the original dataset and the PersonalityLinMulT
paper:

> Fodor et al., *Multimodal Sentiment and Personality Perception Under Speech:
> A Comparison of Transformer-based Architectures.*
> [PMLR v173](https://proceedings.mlr.press/v173/fodor22a.html)

## Links

- **Code / models:** https://github.com/fodorad/PersonalityLinMulT
- **Documentation:** https://fodorad.github.io/PersonalityLinMulT/
- **Original dataset:** https://chalearnlap.cvc.uab.cat/dataset/24/description/
