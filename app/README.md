---
title: PersonalityLinMulT Big Five
emoji: 🎭
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.22.0
app_file: app.py
pinned: false
license: mit
short_description: Big Five apparent-personality prediction from audio or video
python_version: "3.12"
---

# PersonalityLinMulT — Big Five apparent-personality demo

Predict the Big Five personality traits (openness, conscientiousness,
extraversion, agreeableness, emotional stability) from a short clip, using
one of two **champion** models trained in
[**PersonalityLinMulT**](https://github.com/fodorad/PersonalityLinMulT) on
the ChaLearn First Impressions V2 dataset:

- **`wavlm_best-ccc`** — single-stream LinT, acoustic (WavLM) only, trained
  with CCC loss at batch size 256. Test `1-MAE=0.8918`, `CCC=0.5604`,
  `Pearson r=0.5609`, `std_ratio=1.0013` (near-perfect prediction variance
  vs. ground-truth variance).
- **`avt_best-ccc`** — LinMulT, full multimodal fusion of all 7
  acoustic/visual/textual features. Test `1-MAE=0.9074`, `CCC=0.6901`,
  `Pearson r=0.7086`, `std_ratio=0.9924` — beats the audio-only champion on
  every accuracy metric.

Pick a **model** and a **backend** from the two dropdowns, then upload a
clip. A video's audio track is extracted automatically (mono 44.1 kHz,
matching training).

- `wavlm_best-ccc` supports both `pytorch` (`torch`/`lightning`, manual
  feature extraction) and `onnx` (`onnxruntime` only, the full
  decode→WavLM→LinT pipeline in one call).
- `avt_best-ccc` is **ONNX only**: its full pipeline
  (`predict_from_media`) runs face detection/tracking, six ONNX feature
  extractors, and the trained LinMulT head, with an optional transcript
  field — leave it blank to transcribe with Whisper automatically, or check
  "skip text" to predict from audio/video alone.

Both download from the same public
[HF Hub model repo](https://huggingface.co/fodorad/personalitylinmult-fi-models),
so this Space and `pip install personalitylinmult[onnx]` share one loading
mechanism. The progress log below the upload box reports each stage's
wall-clock time — ONNX is typically **~6-8x faster end to end** once warmed
up for the audio-only champion, since WavLM (tens of millions of
parameters) dominates wall-clock time and its ONNX export is where the real
speedup shows.

> These are apparent-personality *perception* models: they predict how a panel of
> annotators rated a person from a short clip, not any ground truth about the
> person. Use accordingly.
