<div align="center">

<img src="https://raw.githubusercontent.com/fodorad/PersonalityLinMulT/main/docs/assets/logo.svg" alt="PersonalityLinMulT" width="360"/>

<br/>

**Transformer-based Big Five Personality Perception and Interview-Variable Estimation**

[![GitHub Release](https://img.shields.io/github/v/release/fodorad/PersonalityLinMulT?color=purple)](https://github.com/fodorad/PersonalityLinMulT/releases)
[![PyPI](https://img.shields.io/pypi/v/personalitylinmult?color=purple)](https://pypi.org/project/personalitylinmult/)
[![CI](https://github.com/fodorad/PersonalityLinMulT/workflows/CI/badge.svg)](https://github.com/fodorad/PersonalityLinMulT/actions)
[![Coverage](https://codecov.io/gh/fodorad/PersonalityLinMulT/branch/main/graph/badge.svg)](https://codecov.io/gh/fodorad/PersonalityLinMulT)
[![Docs](https://img.shields.io/badge/docs-online-blue?logo=githubpages)](https://fodorad.github.io/PersonalityLinMulT/)
<br/>
[![Python](https://img.shields.io/badge/python-3.13%2B-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![ONNX](https://img.shields.io/badge/ONNX%20-supported-6b6b6b?logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

</div>

---

PersonalityLinMulT builds on [LinMulT](https://github.com/fodorad/linmult). It is trained on the First Impressions V2 dataset for two separate regression tasks: perceived Big Five traits (Automatic Personality Perception), and the interview (hireability) variable.

* paper: **Multimodal Sentiment and Personality Perception Under Speech: A Comparison of Transformer-based Architectures** ([pdf](https://proceedings.mlr.press/v173/fodor22a/fodor22a.pdf), [website](https://proceedings.mlr.press/v173/fodor22a.html))

**Two promoted, deployed champions ship with this repository:** `wavlm_best-ccc` (acoustic only, LinT) and `avt_best-ccc` (acoustic + visual + textual fusion of all 7 features, LinMulT). Both are fully wired into every surface below — the public API, ONNX export, Docker/Cloud Run server, HF Space, Gradio demo and showcase notebook — end-to-end. The dataset (7 features across all three modalities), training code, and ablation matrix support any other single-feature or fusion combination the same way — `LinT` for any single feature, `LinMulT` for any fusion — so a visual-only or textual-only champion is trained and promoted the same way once a run is selected (see [Champion selection](#champion-selection)).

### What's in this repository

- **Training** — LinT (single-stream) and LinMulT (cross-modal fusion) over any subset of 7 features, configurable losses (elementwise, CCC, and a learned-uncertainty-weighted combination), MLflow-tracked, Optuna HPO, feature ablations.
- **Metrics** — 1-MAE, Pearson r, CCC, R², and `std_ratio` (prediction-vs-target variance ratio, catching regression-to-the-mean that 1-MAE alone cannot see).
- **Model lifecycle** — MLflow Model Registry with alias-based promotion (`@champion`, not deprecated stages), so a specific trained run becomes "the" deployable model through a deliberate, auditable step.
- **Public inference API** — `pip install personalitylinmult`, then `PersonalityModel.from_pretrained("wavlm_best-ccc")`: downloads a champion from the public HF Hub model repo and predicts, no MLflow or repo clone needed. An optional `[onnx]` extra swaps in `PersonalityModelONNX` for a PyTorch/Lightning-free runtime, including a full raw-audio-to-prediction pipeline on ONNX Runtime alone — measurably faster end-to-end (see [ONNX optimization](#onnx-optimization)).
- **Deployment** — a Docker image (`deploy/Dockerfile`) serving the ONNX path, runnable identically on `localhost` or on Google Cloud Run (true scale-to-zero, no idle billing), plus a Hugging Face Spaces Gradio demo with a live PyTorch-vs-ONNX backend switch.

# Setup

### Install package from PyPI for inference

```
pip install personalitylinmult
```

```python
from personalitylinmult import PersonalityModel

model = PersonalityModel.from_pretrained("wavlm_best-ccc")
scores = model.predict({"wavlm": wavlm_features})  # {trait: score in [0, 1]}
```

`model.feature_names` lists exactly which features a given checkpoint needs (a checkpoint is self-describing, so the same `PersonalityModel` class serves every architecture — no need to track LinT vs. LinMulT yourself). See [Pretrained models](#pretrained-models) below for which ids are actually available.

### Install package for training

```
git clone https://github.com/fodorad/PersonalityLinMulT
cd PersonalityLinMulT
make train        # training + the HDF5 builder
make preprocess   # only if re-extracting features from raw video
```

> Use `uv` (as the Makefile does) for the `preprocess` extra: it applies the
> `[tool.uv]` dependency overrides required to resolve the
> exordium/linmult/blinklinmult stack. Plain `pip` ignores those overrides and
> will fail.

#### Supported extras definitions:

| extras tag | description                                                                     |
| ---------- | ------------------------------------------------------------------------------ |
| train      | Lightning, MLflow, Hugging Face Hub, plotting, and the HDF5 builder            |
| preprocess | the feature-extraction stack (exordium, whisperx); needed only to rebuild data  |
| onnx       | `PersonalityModelONNX` — ONNX Runtime inference, no PyTorch/Lightning needed    |
| serve      | the REST serving app (`deploy/serve.py`) — FastAPI + uvicorn                    |
| demo       | the Gradio demo / Hugging Face Space                                           |
| dev        | linting, type-checking, testing, and pre-commit tooling                         |
| docs       | Sphinx documentation generation                                                 |

Reading a built HDF5 and running a model needs neither extra — the data layer
and the model are part of the core install.

# Data preparation

The First Impressions V2 (FI) dataset is not redistributed with this
repository due to its license. To reproduce preprocessing and feature
extraction:

1. Obtain the FI dataset from its official source under its license terms.
2. Place it at `data/raw/FI/` with the following layout (unmodified from the
   original release):
   ```
   data/raw/FI/
   ├── gt/       # annotation_*.pkl, transcription_*.pkl
   ├── train/
   ├── valid/
   └── test/
   ```
3. Extract the features, then build the training artifact:
   ```
   make preprocess-fi   # decode video/audio, write label.csv
   make features-fi     # run the feature extractors (GPU, hours)
   make hdf5-fi          # collect them into data/processed/FI/h5/fi.h5
   ```

### The HDF5 artifact

`make hdf5-fi` collects the features named in `config/FI/data/avt.yaml` into a
single HDF5 file. **Only the features that config lists are read or written** —
if it names WavLM and DINOv2, nothing else enters the file. Its root attributes
embed the builder config verbatim, the commit SHA, and the label CSV's hash, so
a built file is always traceable to what produced it.

Training reads that file and nothing else, so it is the reproducibility
boundary. The raw video tree and the intermediate feature directories are
~300 GB (`wavlm_baseplus` alone is 263 GB) and are never redistributed; only the
compact HDF5 is.

**Two ways to reproduce**, so you can trust the artifact or verify it:

```
make pull-fi-h5   # fast path: download the built fi.h5 from Hugging Face,
                  #   then train immediately -- no feature extraction
make pipeline-fi  # full path: rebuild fi.h5 from raw FI (hours), for verification
```

The features in `fi.h5` are **derived** (embeddings), not the ChaLearn First
Impressions videos, so the artifact is published to a public Hugging Face
dataset repo ([`fodorad/personalitylinmult-fi`](https://huggingface.co/datasets/fodorad/personalitylinmult-fi),
documented on the [dataset docs page](https://fodorad.github.io/PersonalityLinMulT/dataset.html)).
Obtain the raw videos from the official ChaLearn release if you want to run the full extraction. The h5's
embedded git SHA and builder config are what let you confirm a downloaded file
matches the recipe.

> DVC is intentionally not used: it cannot push to a Hugging Face remote, and
> its value is a paid cloud remote this project does not have. The Makefile is
> the build recipe; Hugging Face is the artifact store.

# Training

Two separate tasks over the same features:

```
make hdf5-fi              # build the artifact first
make train-fi-bigfive     # Big Five trait regression
make train-fi-interview   # interview-variable regression
make mlflow-ui            # monitor runs at http://localhost:5000
```

Big Five regression trains on `emotional_stability` (`= 1 - neuroticism`) so that
every trait points the same way; naming `neuroticism` as a target is rejected to 
avoid any misunderstanding.
The interview variable is a hireability judgement rather than a personality
trait, so it is trained as its own task with its own head.

### Metrics

The primary metric is per-trait **CCC** (Lin's concordance correlation),
summarised as the mean of the five independent trait scores. That mean
(`valid/mean_ccc`) selects checkpoints and drives early stopping.
1-MAE, Pearson *r*, R², and MSE are reported per trait alongside it.

CCC, not 1-MAE, is primary because **1-MAE is misleading exactly when it
matters most**: the FI trait distributions cluster tightly around their mean,
so a model that always predicts the population mean regardless of input
already scores a deceptively good 1-MAE — it never commits to an extreme
prediction, so it never pays a large per-sample penalty, even though it has
learned nothing about the input. This is regression-to-the-mean, and 1-MAE
alone cannot detect it: a model can improve 1-MAE while its predictions
collapse toward a narrow band around the mean, which is the opposite of
useful. CCC penalizes exactly this failure mode — it requires both correct
correlation with the target *and* the target's variance to be reproduced
(`std_ratio`, below, makes the variance-collapse component directly
visible). A model cannot "win" on CCC by hedging toward the mean the way it
can on 1-MAE. This project's own experiments confirmed the effect directly:
switching the loss/selection metric from an elementwise objective to CCC at
a large enough batch size raised `std_ratio` from ~0.85 to ~1.00 (predictions
as spread out as the real trait values) while 1-MAE barely moved.

| metric | meaning |
| ------ | ------- |
| `{split}/ccc/{trait}` | agreement including bias and scale — **primary** |
| `{split}/mean_ccc` | mean of the five trait scores — **model selection** |
| `{split}/one_minus_mae/{trait}` | per-trait 1 − MAE — reported, not selected on |
| `{split}/pearson_r/{trait}` | ranks clips correctly, ignoring bias |
| `{split}/r2/{trait}` | variance explained vs. predicting the mean |

### Analysis window

Clips are ~15.3 s. The HDF5 pads each feature to `max_time_dim` and the window is
applied by the dataloader from each feature's rate, so every window is a round
number of frames:

| feature | rate | 15 s | 10 s | 5 s |
|---|---|---|---|---|
| WavLM, emotion2vec+ | 50 Hz | 750 | 500 | 250 |
| DINOv2 | 25 Hz | 375 | 250 | 125 |
| RoBERTa (text) | — | 112 tokens, never windowed |

The rates are nominal (20 ms stride, 25 fps) rather than the measured 49.79/25.02
Hz; the difference is the few hundred ms by which clips overrun 15 s, which the
builder truncates and reports. Text is padded to 112 because the longest of the
10000 transcripts is 111 tokens. Sweeping the window needs **no rebuild**:

```
make train-fi-windows                                   # 15s, 10s, 5s
make train-fi-bigfive ARGS="--set data.window_seconds=10"
```

### Configuration

Every knob lives in a YAML file under `config/FI/`, parsed into typed dataclasses:

| directory | purpose |
| --------- | ------- |
| `features/` | one file per feature: where it lives, its dim, its rate, which layers |
| `data/` | which features enter the HDF5, the window, and the dataloader settings |
| `model/` | LinMulT architecture (`linmult.yaml`) and the MulT baseline (`mult.yaml`) |
| `train/` | task, targets, loss, optimizer, early stopping, MLflow |

Any value can be overridden without editing a file:

```
make train-fi-bigfive ARGS="--set train.optimizer.lr=1e-4 --set data.batch_size=32"
```

### Feature ablation

A predefined matrix (`personalitylinmult/train/ablation.py`) measures what each
feature and modality contributes: **7 unimodal baselines** — one per feature,
each a single-stream `LinT` — plus **3 within-modality** (all-acoustic,
all-visual, all-textual) and **1 full** multimodal `LinMulT` run. Text is fixed
on the ASR transcripts throughout, since that is what exists at deployment.

```
make ablation-smoke   # one batch per arm -- the pre-flight for a sweep
make ablation         # run the whole matrix, logging each arm to MLflow
```

Each arm is the shared `data`/`model`/`train` configs with the arm's overrides
applied, so editing the study is editing `ablation.py`, not a tree of config
files. A run that fails is recorded and skipped rather than aborting the sweep.
Unimodal arms use `LinT` (single-stream) and the rest use `LinMulT`; the family
is a config field (`model.family`), validated against the feature count.

# Pretrained models

`pip install personalitylinmult` and `PersonalityModel.from_pretrained(model_id)` downloads directly from the public [HF Hub model repo](https://huggingface.co/fodorad/personalitylinmult-fi-models) — no MLflow, no repo clone. Every id below follows `{features joined by "_", multiword names use "-"}_best-{metric}` (e.g. `wavlm_best-ccc`); the architecture (LinT vs. LinMulT) is never part of the id — a checkpoint is self-describing, so `from_pretrained` reads it off the downloaded checkpoint automatically.

| Model id | Modality | Architecture | Test 1-MAE | Test CCC | Test Pearson r | Test std_ratio | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `avt_best-ccc` | acoustic + visual + textual (all 7 features) | LinMulT | **0.9074** | **0.6901** | **0.7086** | 0.9924 | **available, deployed** — see [Deployments](#deployments) |
| `wavlm_best-ccc` | acoustic (WavLM) only | LinT | 0.8918 | 0.5604 | 0.5609 | **1.0013** | **available, deployed** — see [Deployments](#deployments) |

Both are promoted, real champions, and both are fully wired into every live surface (public API, ONNX export, Docker/Cloud Run server, HF Space, showcase notebook). Full multimodal fusion (`avt_best-ccc`) beats audio-only on every accuracy metric (1-MAE, CCC, Pearson r), confirming visual and textual signal genuinely add information beyond speech alone; audio-only edges it out on `std_ratio` (prediction-vs-target variance ratio — 1.0013 vs. 0.9924, both close to the ideal of 1.0). `avt_best-ccc`'s live-extraction path (`PersonalityModelONNX.predict_from_media`, ONNX-only — see [ONNX optimization](#onnx-optimization)) runs face detection/tracking plus six ONNX feature extractors, with an optional Whisper ASR step when no transcript is supplied.

**Reproducing each champion's training run:**

```
# avt_best-ccc (LinMulT, all 7 features, ccc+l1 loss, batch_size=16
# gradient-accumulated x16 -> effective batch 256)
make hdf5-fi   # uses config/FI/data/avt.yaml by default
make train-fi-bigfive \
  ARGS="--set train.loss=ccc+l1 --set train.accumulate_grad_batches=16"

# wavlm_best-ccc (LinT, WavLM only, ccc loss, batch_size=256)
make hdf5-fi FI_DATA_CONFIG=config/FI/data/wavlm.yaml
make train-fi-bigfive \
  FI_DATA_CONFIG=config/FI/data/wavlm.yaml FI_MODEL_CONFIG=config/FI/model/lint.yaml \
  ARGS="--set train.loss=ccc --set data.batch_size=256"
```

Both then follow the same promotion + publish steps — see [Champion selection](#champion-selection) below.

# Champion selection

A trained run only becomes "the" deployable model through a deliberate,
auditable promotion step — nothing is ever promoted automatically:

```
make register-models                          # wrap a trained checkpoint as an MLflow pyfunc model
make promote-champion MODEL=<name> VERSION=<n> # move the @champion alias to it
```

`@champion` replaces MLflow's deprecated Staging/Production stages — it is
scoped per registered-model *name*, so independent champions (acoustic-only,
visual-only, text-only, full AVT fusion) each carry their own `@champion` with
no conflict. Once promoted, publish it to the public HF Hub model repo so it
is reachable via `pip install personalitylinmult` alone, and add its row to
the [Pretrained models](#pretrained-models) table above:

```
make push-champion-model MODEL_ID=wavlm_best-ccc CKPT=<path/to/best.ckpt>
```

# ONNX optimization

An optional, PyTorch/Lightning-free inference path (`[onnx]` extra), applied
as a post-training, post-promotion step:

```
pip install personalitylinmult[onnx]
```

```python
from personalitylinmult.onnx import PersonalityModelONNX

model = PersonalityModelONNX.from_pretrained("wavlm_best-ccc")
scores = model.predict_from_audio("clip.wav")  # raw audio/video -> Big Five,
                                                # no torch/transformers/exordium
```

A separate class, not a `backend=` flag on `PersonalityModel` — installing
the plain package never pulls in `onnxruntime`. WavLM's own ONNX export
(shared across every WavLM-based champion) downloads on first use.

The full multimodal champion has its own full-pipeline method,
`predict_from_media`, since it needs face detection/tracking and (optionally)
Whisper ASR in addition to six ONNX feature extractors — those two steps stay
`torch`/`exordium`-based, everything else runs on ONNX Runtime alone:

```python
model = PersonalityModelONNX.from_pretrained("avt_best-ccc")
scores = model.predict_from_media("clip.mp4")               # transcript=None -> Whisper ASR
scores = model.predict_from_media("clip.mp4", transcript="hello world")  # explicit transcript
scores = model.predict_from_media("clip.mp4", transcript="")  # skip text (placeholder + all-False mask)
```

**Two findings, at two different scales, both real:**

- The trained LinT head alone (~20K params) is numerically exact under ONNX
  but **not faster** — onnxruntime's own session overhead dominates at this
  scale.
- The **full pipeline** (raw audio → WavLM → LinT), where WavLM's tens of
  millions of parameters actually benefit from ONNX Runtime's graph
  optimizations, is **~6-8x faster end-to-end** once warmed up.

That's why `deploy/serve.py` (below) runs the ONNX path exclusively, while
the Gradio demo offers both behind a switch so the difference is visible
live. Small, measured, bounded accuracy tradeoff from `ffmpeg`-based
resampling (avoiding a `torchaudio` dependency): ~0.001–0.004 per trait.

# Inference

```python
from personalitylinmult import PersonalityModel

model = PersonalityModel.from_pretrained("wavlm_best-ccc")
scores = model.predict({"wavlm": wavlm_features})  # {trait: score in [0, 1]}
```

Or the ONNX-only, torch-free equivalent — see [ONNX optimization](#onnx-optimization) above. Both classes are checkpoint-self-describing: `model.feature_names` and `model.traits` always match what the loaded champion actually needs and predicts, so the same code works for any promoted champion (acoustic, visual, textual, or full multimodal) without an architecture lookup.

# Notebook comparisons and demo

**[`notebooks/showcase.ipynb`](notebooks/showcase.ipynb)** — three parts:

1. Loads the audio-only champion via `PersonalityModel.from_pretrained`,
   extracts real features from a local FI test clip, predicts, and plots
   predicted vs. ground truth (the ground-truth annotation ships as a
   fixture; the raw clip itself is never redistributed — see
   [`docs/dataset.md`](docs/dataset.md)).
2. A **PyTorch vs. ONNX comparison** on a synthetic `ffmpeg`-generated fixture
   clip (no dataset license needed) — runs the full audio-only pipeline
   through both backends, times each, and prints a per-trait diff table.
3. The **full multimodal (AVT) champion**, `predict_from_media`, on the same
   kind of synthetic fixture (video this time) — ONNX only, since that
   champion's live pipeline is only implemented for `PersonalityModelONNX`.

**Hugging Face Space (Gradio)** — live at
**[huggingface.co/spaces/fodorad/personalitylinmult-fi](https://huggingface.co/spaces/fodorad/personalitylinmult-fi)**
(`app/app.py`; run `uv run --extra demo --extra onnx python app/app.py`
locally instead if you prefer). Pick a **model** (`wavlm_best-ccc` or
`avt_best-ccc`) and a **backend** (`pytorch` or `onnx`, audio-only champion
only — AVT is ONNX-only) from two dropdowns, upload a clip, and optionally
supply a transcript for the AVT champion (leave it blank to transcribe with
Whisper automatically, or check "skip text" to predict from audio/video
alone). The progress log reports each pipeline stage's wall-clock time — so
the ONNX speedup from the section above is something you can see live on
your own hardware, not just read in a table. Hosted on ZeroGPU (the only
free Gradio hardware tier) even though the app is entirely CPU-based — see
`app/app.py`'s module docstring for the no-op `@spaces.GPU` function that
satisfies ZeroGPU's startup requirement without ever actually requesting a
GPU.

# Deployments

The same small FastAPI server (`deploy/serve.py`, ONNX Runtime only — see
[ONNX optimization](#onnx-optimization)) runs identically whether started
directly on `localhost` via Docker, or deployed to Google Cloud Run. There is
no separate "local" variant of the image.

### Docker (local)

```
make build-serve-docker            # docker build -f deploy/Dockerfile ...
make run-serve-docker              # docker run -p 8080:8080 ...
curl http://localhost:8080/ping
curl -X POST http://localhost:8080/predict_audio -F file=@clip.wav
curl -X POST http://localhost:8080/predict_media -F file=@clip.mp4 -F transcript="hello world"
```

### Google Cloud Run

```
make gcloud-setup GCP_PROJECT=<id>    # one-time, per GCP project
make push-serve-gcloud GCP_PROJECT=<id> MODEL=wavlm_best-ccc
make deploy-serve-cloudrun GCP_PROJECT=<id> MODEL=wavlm_best-ccc
```

`--min-instances=0` (true scale-to-zero, no idle billing) and
`--max-instances=3` (caps worst-case spend under a spike). Chosen over
Fly.io/Render for a standing, non-trial free tier. Each deployed instance
serves exactly one champion, selected by the `MODEL_ID` environment
variable — `SERVE_MODEL_ID` defaults to `MODEL`, so `make
deploy-serve-cloudrun MODEL=<id>` sets both the image tag and the runtime
`MODEL_ID` together. Three endpoints exist on every instance —
`POST /invocations` (pre-extracted features), `POST /predict_audio` (raw
audio/video upload, audio-only champion), and `POST /predict_media` (raw
video upload + optional `transcript` field, full AVT champion) — but only
the endpoint matching the loaded champion's `feature_names` will actually
succeed; the other returns a clear 400, not a 500. Deliberately not
`mlflow models build-docker`: that image spawns a second, isolated
virtualenv at container startup purely to run the model server (defeating
any fix applied to the outer Python's site-packages), and its nginx layer
produced no diagnosable log output under Cloud Run specifically.
`deploy/serve.py` is a handful of lines instead — one process, no nginx,
`--host 0.0.0.0` explicit.

**Only `wavlm_best-ccc` is deployed to Cloud Run.** `avt_best-ccc` was
tried and pulled back: `predict_from_media` loads six ONNX extractor
sessions (~3.7 GB of weights) plus PyTorch-based face detection/tracking
(YOLO + AdaFace) simultaneously, and this genuinely exceeded 8 GiB of
resident memory in testing (OOM'd a 4 GiB instance at ~4.1 GiB used, then
an 8 GiB instance at ~8.3 GiB used and still climbing) — Cloud Run's next
tier up (16–32 GiB, `--cpu=8`) works, but the resulting per-request cost is
outside the free-tier budget this project targets, so it wasn't kept
running. `SERVE_MEMORY` (default `4Gi`) exists precisely for this — pass
`SERVE_MEMORY=16Gi` (or higher) if you want to deploy `avt_best-ccc` here
yourself and are fine paying for the bigger instance:

```
make push-serve-gcloud GCP_PROJECT=<id> MODEL=avt_best-ccc
make deploy-serve-cloudrun GCP_PROJECT=<id> MODEL=avt_best-ccc SERVE_MEMORY=16Gi
```

The AVT champion is still fully available elsewhere at no extra
infrastructure cost: the [HF Space](https://huggingface.co/spaces/fodorad/personalitylinmult-fi)
above runs it live on free ZeroGPU-tier hardware, and `PersonalityModel
.from_pretrained("avt_best-ccc")` / `PersonalityModelONNX
.from_pretrained("avt_best-ccc")` work identically from a plain
`pip install` — Cloud Run is just one deployment surface among several, not
the only way to reach this champion.

# Documentation

Full API reference (Sphinx + autoapi) and the dataset guide:
**[fodorad.github.io/PersonalityLinMulT](https://fodorad.github.io/PersonalityLinMulT/)**

| path | covers |
| --- | --- |
| [`docs/dataset.md`](docs/dataset.md) | the FI dataset, license, and the HDF5 artifact |

# Related projects

### exordium

Collection of preprocessing functions and deep learning methods. This repository contains revised codes for fine landmark detection (including face, eye region, iris and pupil landmarks), head pose estimation, and eye feature calculation.

* code: https://github.com/fodorad/exordium

### (2022) LinMulT

General-purpose Multimodal Transformer with Linear Complexity Attention Mechanism. This base model is further modified and trained for various tasks and datasets.

* code: https://github.com/fodorad/LinMulT

### (2023) BlinkLinMulT

LinMulT is trained for blink presence detection and eye state recognition tasks.
Our results demonstrate comparable or superior performance compared to state-of-the-art models on 2 tasks, using 7 public benchmark databases.

* paper: BlinkLinMulT: Transformer-based Eye Blink Detection ([pdf](https://adamfodor.com/pdf/2023_Fodor_Adam_MDPI_BlinkLinMulT.pdf), [website](https://www.mdpi.com/2313-433X/9/10/196))
* code: https://github.com/fodorad/BlinkLinMulT

# Citation - BibTex

If you found our research helpful or influential please consider citing:

### (2023) BlinkLinMulT for blink presence detection and eye state recognition

```
@Article{fodor2023blinklinmult,
  title = {BlinkLinMulT: Transformer-Based Eye Blink Detection},
  author = {Fodor, Ádám and Fenech, Kristian and Lőrincz, András},
  journal = {Journal of Imaging},
  volume = {9},
  year = {2023},
  number = {10},
  article-number = {196},
  url = {https://www.mdpi.com/2313-433X/9/10/196},
  PubMedID = {37888303},
  ISSN = {2313-433X},
  DOI = {10.3390/jimaging9100196}
}
```

### (2022) LinMulT for personality trait and sentiment estimation

```
@InProceedings{pmlr-v173-fodor22a,
  title = {Multimodal Sentiment and Personality Perception Under Speech: A Comparison of Transformer-based Architectures},
  author = {Fodor, {\'A}d{\'a}m and Saboundji, Rachid R. and Jacques Junior, Julio C. S. and Escalera, Sergio and Gallardo-Pujol, David and L{\H{o}}rincz, Andr{\'a}s},
  booktitle = {Understanding Social Behavior in Dyadic and Small Group Interactions},
  pages = {218--241},
  year = {2022},
  editor = {Palmero, Cristina and Jacques Junior, Julio C. S. and Clapés, Albert and Guyon, Isabelle and Tu, Wei-Wei and Moeslund, Thomas B. and Escalera, Sergio},
  volume = {173},
  series = {Proceedings of Machine Learning Research},
  month = {16 Oct},
  publisher = {PMLR},
  pdf = {https://proceedings.mlr.press/v173/fodor22a/fodor22a.pdf},
  url = {https://proceedings.mlr.press/v173/fodor22a.html}
}
```

# Contact

* Ádám Fodor (fodorad201@gmail.com) — [adamfodor.com](https://adamfodor.com)
