# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

From v3.0.0 onward, releases and version bumps are automated via
[release-please](https://github.com/googleapis/release-please) driven by
Conventional Commits.

---

## [3.0.0](https://github.com/fodorad/PersonalityLinMulT/compare/v2.1.4...v3.0.0) (2026-08-10)


### ⚠ BREAKING CHANGES

* refactor to two-champion multimodal architecture with ONNX deployment ([#1](https://github.com/fodorad/PersonalityLinMulT/issues/1))

### Features

* refactor to two-champion multimodal architecture with ONNX deployment ([#1](https://github.com/fodorad/PersonalityLinMulT/issues/1)) ([9ce770a](https://github.com/fodorad/PersonalityLinMulT/commit/9ce770a75051ca2b5ac435bfcbe8b0cd5dd7cd7f))


### Bug Fixes

* **ci:** run release-please in manifest mode so it versions from 2.1.4 ([#12](https://github.com/fodorad/PersonalityLinMulT/issues/12)) ([8eee224](https://github.com/fodorad/PersonalityLinMulT/commit/8eee224bf7bdc78af26eff172ef253ad64397fd6))

## [3.0.0] - 2026-06-28

This is a major release that modernizes the project's tooling, dependencies, and
release process. From this version on, versioning is fully automated.

### Changed

- **BREAKING: License changed from CC BY-NC 4.0 to MIT.** The project is now
  permissively licensed and free for commercial use.
- **BREAKING: Minimum Python raised to 3.13** (from 3.12). CI runs on Python
  3.14 across `ubuntu-latest` and `macos-latest`.
- Adopted standardized DevOps tooling matching `exordium`/`linmult`: GitHub
  Actions CI/CD, Codecov, Sphinx auto-generated API docs (GitHub Pages),
  release-please automation, Dependabot, pre-commit, `ruff`, and `ty`.
- Switched to tag-driven versioning via `hatch-vcs` — the version is derived
  from the git tag at build time instead of a hardcoded value in
  `pyproject.toml`.
- The `linmult` dependency is now unpinned so the project always tracks the
  latest LinMulT release.
- Pinned the PyTorch ecosystem together for the `train` extra:
  `torch==2.11.0`, `torchaudio==2.11.0`, `torchvision==0.26.0`, and
  `torchcodec==0.14.0` (kept in lockstep with `exordium`).
- Relaxed the remaining `train` pins (`lightning`, `matplotlib`,
  `opencv-python`, `pandas`, `tqdm`) to track compatible latest versions.
- FI face-track preprocessing (`fi_create_tracks.py`) now uses exordium 2.5's
  `FaceClusterTracker` (IoU tracklets + AdaFace face-ID clustering) to recover a
  single protagonist across occlusions and scene cuts, replacing the plain IoU
  tracker and geometric selection heuristic. Terminology standardized on
  "protagonist" (was "main character"/"participant"); the heuristic output is now
  `protagonist_track_heuristic.csv`.

### Added

- `fi_predict_transcript.py`: whisperX speech-to-text over the FI audio clips
  (via exordium 2.6.1's `text` extra), writing `transcript_whisperx.csv`
  (`video_id,text`) with optional per-clip word-timestamp JSON (`--save-words`).
- `mlflow` to the `train` extra for experiment tracking.
- `dev` and `docs` optional-dependency extras (linting/typing/testing and
  documentation tooling).
- Project meta files: `CHANGELOG.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
  `SECURITY.md`, issue/PR templates, and a `Makefile`.
- Bootstrap test scaffold under `tests/`.

### Removed

- Tracked contents of `results/` (the directory is kept via `.gitkeep`).
- The obsolete `all` extra and `requirements.txt` install path.

## [2.1.4] - 2026-06-04

### Added

- Trained model weights and experiment scripts for First Impressions (FI) and
  CMU-MOSEI.

[3.0.0]: https://github.com/fodorad/PersonalityLinMulT/releases/tag/v3.0.0
[2.1.4]: https://github.com/fodorad/PersonalityLinMulT/releases/tag/v2.1.4
