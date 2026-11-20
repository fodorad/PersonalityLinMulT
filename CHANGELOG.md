# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

From v3.0.0 onward, releases and version bumps are automated via
[release-please](https://github.com/googleapis/release-please) driven by
Conventional Commits.

---

## [3.0.0](https://github.com/fodorad/PersonalityLinMulT/compare/v2.1.4...v3.0.0) (2026-08-10)

Two promoted, deployed champions ship with this release: `wavlm_best-ccc`
(acoustic-only, `LinT`) and `avt_best-ccc` (acoustic + visual + textual fusion
of all 7 features, `LinMulT`). This is a major release that modernizes the
tooling, dependencies, and release process, and adds a complete production
story: public inference API, ONNX optimization, Docker/Cloud Run serving, and a
live Gradio demo.

### Architecture & models

- Refactor to a two-champion multimodal architecture: a single-stream acoustic
  champion (`LinT`, WavLM only) and a full cross-modal fusion champion
  (`LinMulT` over all 7 features), both trained, promoted, and deployed
  end-to-end ([#1](https://github.com/fodorad/PersonalityLinMulT/issues/1)).
- Checkpoints are self-describing: `model.feature_names` and `model.traits`
  are read off the checkpoint, so one `PersonalityModel` class serves every
  architecture — no LinT-vs-LinMulT bookkeeping.
- Feature ablation matrix (`ablation.py`): 7 unimodal `LinT` baselines, 3
  within-modality (all-acoustic, all-visual, all-textual), and 1 full
  multimodal `LinMulT` arm, each logged to MLflow and driven by a single
  config family.
- Two real champions ship with results: `avt_best-ccc` (test CCC 0.6901) and
  `wavlm_best-ccc` (test CCC 0.5604).

### Public inference API & ONNX

- Public API: `pip install personalitylinmult` +
  `PersonalityModel.from_pretrained(model_id)` — downloads a champion from the
  public HF Hub model repo; no MLflow, no repo clone.
- `[onnx]` extra: `PersonalityModelONNX` — PyTorch/Lightning-free runtime,
  including `predict_from_audio` (raw audio → Big Five) and
  `predict_from_media` (raw video → Big Five, with face detection/tracking and
  optional Whisper ASR).
- ~6–8× end-to-end speedup on the full ONNX pipeline (WavLM benefits from ONNX
  Runtime's graph optimizations); the LinT head alone is numerically exact but
  not faster. Measured, bounded accuracy tradeoff from `ffmpeg`-based
  resampling: ~0.001–0.004 per trait.
- WavLM's ONNX export is shared across every WavLM-based champion and
  downloads on first use.

### Deployment & serving

- Docker image (`deploy/Dockerfile`) + FastAPI server (`deploy/serve.py`,
  ONNX Runtime only) running identically on localhost and Google Cloud Run
  (true scale-to-zero, `--min-instances=0`, `--max-instances=3`).
- Three endpoints on every instance (`/invocations`, `/predict_audio`,
  `/predict_media`); the one matching the loaded champion's `feature_names`
  succeeds, the others return a clear 400.
- Hugging Face Spaces Gradio demo (ZeroGPU): model + backend
  (`pytorch`/`onnx`) dropdowns, live per-stage wall-clock timing, optional
  transcript or automatic Whisper ASR — the ONNX speedup is visible live.

### Training & experimentation

- CCC promoted to primary metric (per-trait, summarized as the mean of five),
  with `std_ratio` (prediction-vs-target variance ratio) making
  regression-to-the-mean visible — the failure mode 1-MAE alone cannot
  detect. Reported alongside 1-MAE, Pearson r, R², and MSE.
- MLflow Model Registry with alias-based promotion (`@champion`) replacing the
  deprecated Staging/Production stages — auditable, per-model-name champions.
- Analysis-window sweep without rebuilding the HDF5 (`make train-fi-windows`).
- Typed-dataclass YAML config under `config/FI/` with `--set` overrides
  (`make train-fi-bigfive ARGS="--set data.batch_size=32"`).
- `mlflow` added to the `train` extra for experiment tracking.

### Data & preprocessing

- HDF5 as the reproducibility boundary: root attributes embed the builder
  config verbatim, the commit SHA, and the label CSV hash; training reads only
  that file.
- `make pull-fi-h5` (fast path: download the built artifact from the public HF
  dataset repo `fodorad/personalitylinmult-fi`) and `make pipeline-fi` (full
  rebuild from raw FI).
- FI face-track preprocessing now uses exordium's `FaceClusterTracker` (IoU
  tracklets + AdaFace face-ID clustering) to recover a single protagonist
  across occlusions and scene cuts; output standardized as
  `protagonist_track_heuristic.csv`.
- `fi_predict_transcript.py`: whisperX speech-to-text over FI clips →
  `transcript_whisperx.csv`, with optional word-timestamp JSON (`--save-words`).

### Tooling & CI/CD

- Standardized DevOps tooling matching `exordium`/`linmult`: GitHub Actions
  CI/CD (ubuntu + macOS, Python 3.12/3.13), Codecov, Sphinx auto-generated API
  docs on GitHub Pages, release-please automation, Dependabot, pre-commit,
  `ruff`, and `ty`.
- Tag-driven versioning via `hatch-vcs` — the version derives from the git tag
  at build time instead of a hardcoded value in `pyproject.toml`.

### Dependencies & extras

- `linmult` dependency unpinned — the project always tracks the latest LinMulT
  release.
- PyTorch ecosystem pinned together in `train`: `torch==2.11.0`,
  `torchaudio==2.11.0`, `torchvision==0.26.0`, and `torchcodec==0.14.0`.
- Relaxed the remaining `train` pins (`lightning`, `matplotlib`,
  `opencv-python`, `pandas`, `tqdm`) to track compatible latest versions.
- New `dev` and `docs` optional-dependency extras.

### Breaking changes

- Two-champion architecture refactor ([#1](https://github.com/fodorad/PersonalityLinMulT/issues/1)).
- License changed from CC BY-NC 4.0 to MIT — free for commercial use.
- Minimum Python raised to 3.13 (from 3.12).
- The obsolete `all` extra and `requirements.txt` install path removed.

### Removed

- Tracked contents of `results/` (directory kept via `.gitkeep`).

### Fixed

- CI: run release-please in manifest mode so versioning continues from the
  manifest instead of restarting at 0.1.0 ([#12](https://github.com/fodorad/PersonalityLinMulT/issues/12)).

## [2.1.4] - 2026-06-04

### Added

- Trained model weights and experiment scripts for First Impressions (FI) and
  CMU-MOSEI.

[3.0.0]: https://github.com/fodorad/PersonalityLinMulT/releases/tag/v3.0.0
[2.1.4]: https://github.com/fodorad/PersonalityLinMulT/releases/tag/v2.1.4
