.PHONY: install dev install-docs train preprocess fix lint type-check test docs docs-serve docs-deploy check clean help \
        preprocess-fi vdets-fi tracks-fi transcript-fi \
        acoustic-wavlm-fi acoustic-emotion2vec-fi acoustic-fi \
        textual-xlm-roberta-fi textual-mmbert-fi textual-fi \
        visual-marlin-fi visual-dinov2-fi visual-emotieffnet-fi visual-farl-fi visual-fi \
        features-fi hdf5-fi hdf5-fi-smoke pipeline-fi push-fi-h5 pull-fi-h5 push-hf-card \
        push-champion-model pull-champion-model push-hf-model-card \
        train-fi-bigfive train-fi-interview train-fi-bigfive-mult \
        train-fi-a train-fi-wavlm train-fi-emotion2vec train-fi-audio-group lr-find-fi-wavlm \
        train-fi-windows train-fi-smoke ablation ablation-smoke mlflow-ui \
        register-models promote-champion serve-model \
        build-serve-docker run-serve-docker \
        gcloud-setup push-serve-gcloud deploy-serve-cloudrun \
        export-champion-onnx push-champion-onnx benchmark-onnx \
        export-wavlm-onnx push-wavlm-onnx export-dinov2-onnx push-dinov2-onnx \
        export-mmbert-onnx push-mmbert-onnx export-xlm-roberta-onnx push-xlm-roberta-onnx \
        export-emotieffnet-onnx push-emotieffnet-onnx export-farl-onnx push-farl-onnx \
        export-emotion2vec-onnx push-emotion2vec-onnx

help:
	@echo "Dev (modify files):  fix"
	@echo "Checks (read-only):  lint | type-check | test | docs | check"
	@echo "Setup:               install | dev | install-docs | train | preprocess"
	@echo "Docs:                docs-serve | docs-deploy"
	@echo "FI data pipeline:    preprocess-fi | vdets-fi | tracks-fi | transcript-fi |"
	@echo "                     acoustic-wavlm-fi | acoustic-emotion2vec-fi | acoustic-fi |"
	@echo "                     textual-xlm-roberta-fi | textual-mmbert-fi | textual-fi |"
	@echo "                     visual-marlin-fi | visual-dinov2-fi | visual-emotieffnet-fi |"
	@echo "                     visual-farl-fi | visual-fi |"
	@echo "                     features-fi | hdf5-fi | hdf5-fi-smoke | pipeline-fi"
	@echo "FI artifact (HF):    push-fi-h5 | pull-fi-h5"
	@echo "FI model weights (HF): push-champion-model MODEL_ID=... CKPT=... |"
	@echo "                     pull-champion-model MODEL_ID=... | push-hf-model-card"
	@echo "FI training:         train-fi-bigfive | train-fi-interview | train-fi-bigfive-mult |"
	@echo "                     train-fi-windows | train-fi-smoke | mlflow-ui"
	@echo "FI audio group:      train-fi-a | train-fi-wavlm | train-fi-emotion2vec | train-fi-audio-group"
	@echo "FI LR range test:    lr-find-fi-wavlm"
	@echo "FI ablation:         ablation-smoke | ablation"
	@echo "Model deployment:    register-models | promote-champion MODEL=... VERSION=... |"
	@echo "                     serve-model MODEL=..."
	@echo "Serve (Docker/ONNX): build-serve-docker | run-serve-docker"
	@echo "Cloud Run deploy:    gcloud-setup GCP_PROJECT=... | push-serve-gcloud GCP_PROJECT=... |"
	@echo "                     deploy-serve-cloudrun GCP_PROJECT=..."
	@echo "ONNX export:         export-champion-onnx CKPT=... | push-champion-onnx MODEL_ID=... |"
	@echo "                     benchmark-onnx CKPT=... ONNX_OUT=... | export-wavlm-onnx |"
	@echo "                     push-wavlm-onnx | export-dinov2-onnx | push-dinov2-onnx |"
	@echo "                     export-mmbert-onnx | push-mmbert-onnx |"
	@echo "                     export-xlm-roberta-onnx | push-xlm-roberta-onnx |"
	@echo "                     export-emotieffnet-onnx | push-emotieffnet-onnx |"
	@echo "                     export-farl-onnx | push-farl-onnx |"
	@echo "                     export-emotion2vec-onnx | push-emotion2vec-onnx"
	@echo "Cleanup:             clean"

# ── Setup ──────────────────────────────────────────────────────────────────────

install:
	uv sync

install-docs:
	uv sync --extra docs

# Everything needed to build the h5 and train: Lightning, MLflow, HF Hub, plots.
train:
	uv sync --extra train

# The feature-extraction stack (exordium, whisperx, ...). Only needed to rebuild
# data/processed/FI from raw video.
preprocess:
	uv sync --extra preprocess

dev:
	uv sync --all-extras

# ── Dev helpers (modify files) ─────────────────────────────────────────────────

fix:
	uv run ruff format .
	uv run ruff check --fix .

# ── Checks (read-only — mirrors GitHub CI) ─────────────────────────────────────

lint:
	uv run ruff check .
	uv run ruff format --check .

type-check:
	uv run ty check personalitylinmult

test:
	uv run coverage run -m unittest discover -s tests -v
	uv run coverage report
	uv run coverage html
	uv run coverage xml -o coverage.xml

docs:
	uv run sphinx-build -b html docs/ site/

check: lint type-check test docs

# ── Docs ───────────────────────────────────────────────────────────────────────

docs-serve:
	uv run sphinx-autobuild docs/ site/

docs-deploy:
	@echo "Docs are deployed automatically via GitHub Actions on push to main."

# ── FI data pipeline ───────────────────────────────────────────────────────────
# Reproduces data/processed/FI from data/raw/FI. Each step is resumable/
# idempotent (safe to re-run; already-processed files are skipped). Requires
# `make train` first. See README.md "Data" section for obtaining data/raw/FI.

preprocess-fi:
	uv run python personalitylinmult/preprocess/fi/preprocess.py

vdets-fi:
	uv run python personalitylinmult/preprocess/fi/create_vdets.py

tracks-fi: vdets-fi
	uv run python personalitylinmult/preprocess/fi/create_tracks.py

transcript-fi:
	uv run python personalitylinmult/preprocess/fi/create_transcript.py

# Defaults to layer 9 in fp16 (see acoustic_wavlm.py). Pass ARGS="--layers 6 9"
# to keep more; the dir name must then match config/FI/features/wavlm.yaml.
acoustic-wavlm-fi:
	uv run python personalitylinmult/preprocess/fi/acoustic_wavlm.py \
		--wavlm-dir data/processed/FI/wavlm_baseplus_l9 $(ARGS)

acoustic-emotion2vec-fi:
	uv run python personalitylinmult/preprocess/fi/acoustic_emotion2vec.py

acoustic-fi: acoustic-wavlm-fi acoustic-emotion2vec-fi

textual-xlm-roberta-fi:
	uv run python personalitylinmult/preprocess/fi/textual_xlm_roberta.py

textual-mmbert-fi:
	uv run python personalitylinmult/preprocess/fi/textual_mmbert.py

textual-fi: textual-xlm-roberta-fi textual-mmbert-fi

visual-marlin-fi: tracks-fi
	uv run python personalitylinmult/preprocess/fi/visual_marlin.py

visual-dinov2-fi:
	uv run python personalitylinmult/preprocess/fi/visual_dinov2.py

visual-emotieffnet-fi: tracks-fi
	uv run python personalitylinmult/preprocess/fi/visual_emotieffnet.py

visual-farl-fi: tracks-fi
	uv run python personalitylinmult/preprocess/fi/visual_farl.py

# MARLIN is excluded: it is a 16-frame video transformer and is an order of
# magnitude slower than the per-frame face encoders. Run `make visual-marlin-fi`
# explicitly when its features are actually needed.
visual-fi: visual-dinov2-fi visual-emotieffnet-fi visual-farl-fi

features-fi: acoustic-fi textual-fi visual-fi

# Collects the extracted features named in config/FI/data/avt.yaml into the
# single HDF5 training artifact. Only the features that config lists are read
# or written. This file is the reproducibility boundary: it is what gets
# published to Hugging Face (make push-fi-h5) and all that training reads.
hdf5-fi:
	uv run python -m personalitylinmult.data.builder --config $(FI_DATA_CONFIG) $(ARGS)

# A small, real dataset (10 clips per split) for smoke tests and CI.
hdf5-fi-smoke:
	uv run python -m personalitylinmult.data.builder \
		--config config/FI/data/smoke.yaml --limit 30 --overwrite

pipeline-fi: preprocess-fi features-fi hdf5-fi

# ── FI artifact hosting (Hugging Face) ──────────────────────────────────────────
# fi.h5 is the reproducibility boundary: it is published to a public HF dataset
# repo so anyone can pull it and train without re-running the multi-hour feature
# extraction, while `make pipeline-fi` remains the from-scratch rebuild. The h5
# embeds its git SHA + builder config, so a pulled file traces back to the code
# that made it. HF is used instead of DVC (which cannot push to an HF remote).
#
# HF_XET_HIGH_PERFORMANCE replaces the retired HF_HUB_ENABLE_HF_TRANSFER flag.
FI_H5 ?= data/processed/FI/h5/fi.h5
HF_DATASET_REPO ?= fodorad/personalitylinmult-fi

push-fi-h5:
	HF_XET_HIGH_PERFORMANCE=1 uv run hf upload $(HF_DATASET_REPO) \
		$(FI_H5) fi.h5 --repo-type dataset

pull-fi-h5:
	HF_XET_HIGH_PERFORMANCE=1 uv run hf download $(HF_DATASET_REPO) fi.h5 \
		--repo-type dataset --local-dir $(dir $(FI_H5))

# Publish the HF dataset card from its single source of truth (docs/dataset.md),
# prepending the frontmatter HF renders. docs/dataset.md stays canonical, so the
# GitHub docs page and the HF card cannot drift.
HF_CARD_BUILD ?= .build/hf_dataset_card.md
push-hf-card:
	mkdir -p $(dir $(HF_CARD_BUILD))
	cat docs/_hf_frontmatter.yaml docs/dataset.md > $(HF_CARD_BUILD)
	HF_XET_HIGH_PERFORMANCE=1 uv run hf upload $(HF_DATASET_REPO) \
		$(HF_CARD_BUILD) README.md --repo-type dataset

# ── FI model weights (Hugging Face) ─────────────────────────────────────────────
# The public distribution channel for champion checkpoints: `pip install
# personalitylinmult` alone (no MLflow, no repo clone) can load one via
# `PersonalityModel.from_pretrained(MODEL_ID)`, which downloads from this repo.
# One repo holds every champion as a separate file, named by MODEL_ID (see
# README.md's "Pretrained models" naming convention). Publishing is always a
# deliberate, separate step from promotion (`make promote-champion`) -- never
# automatic.
HF_MODEL_REPO ?= fodorad/personalitylinmult-fi-models
MODEL_ID ?=
CKPT ?=

push-champion-model:
	@test -n "$(MODEL_ID)" || (echo "MODEL_ID is required, e.g. MODEL_ID=wavlm_best-ccc" && exit 1)
	@test -n "$(CKPT)" || (echo "CKPT is required: path to the .ckpt to publish" && exit 1)
	HF_XET_HIGH_PERFORMANCE=1 uv run hf upload $(HF_MODEL_REPO) \
		$(CKPT) $(MODEL_ID).ckpt --repo-type model

pull-champion-model:
	@test -n "$(MODEL_ID)" || (echo "MODEL_ID is required, e.g. MODEL_ID=wavlm_best-ccc" && exit 1)
	HF_XET_HIGH_PERFORMANCE=1 uv run hf download $(HF_MODEL_REPO) $(MODEL_ID).ckpt \
		--repo-type model --local-dir .

# ── ONNX export ──────────────────────────────────────────────────────────────────
# A post-training, post-champion-selection step (after `make promote-champion`,
# before deployment): exports the trained model (LinT, single feature, or
# LinMulT, multiple features -- e.g. the AVT champion) to ONNX for the
# `[onnx]`-extra inference path (PersonalityModelONNX), then benchmarks it
# against the plain PyTorch path (speed) and against the recorded test-set
# metrics (accuracy drift). Requires `uv sync --extra onnx` (or `--extra
# onnx --extra train` to also export from a fresh MLflow-registered
# checkpoint) -- onnxruntime/onnx/onnxscript are never installed by default.
# See personalitylinmult/inference/onnx_export.py's module docstring.
ONNX_OUT ?= $(CKPT:.ckpt=.onnx)

export-champion-onnx:
	@test -n "$(CKPT)" || (echo "CKPT is required: path to the trained .ckpt to export" && exit 1)
	uv run python -c "\
from personalitylinmult.train.module import FiLightningModule; \
from personalitylinmult.inference.onnx_export import export_to_onnx; \
module = FiLightningModule.load_from_checkpoint('$(CKPT)'); \
export_to_onnx(module, '$(ONNX_OUT)')"

# Publishes both the .onnx graph and its .onnx.json metadata sidecar.
push-champion-onnx:
	@test -n "$(MODEL_ID)" || (echo "MODEL_ID is required, e.g. MODEL_ID=wavlm_best-ccc" && exit 1)
	@test -n "$(ONNX_OUT)" || (echo "ONNX_OUT is required: path to the exported .onnx" && exit 1)
	HF_XET_HIGH_PERFORMANCE=1 uv run hf upload $(HF_MODEL_REPO) \
		$(ONNX_OUT) $(MODEL_ID).onnx --repo-type model
	HF_XET_HIGH_PERFORMANCE=1 uv run hf upload $(HF_MODEL_REPO) \
		$(ONNX_OUT).json $(MODEL_ID).onnx.json --repo-type model

# Compares PyTorch vs. ONNX Runtime on the same champion: per-sample latency
# and test-set metric drift (1-MAE/CCC/std_ratio), against a real fi.h5 test
# split. See personalitylinmult/inference/onnx_benchmark.py.
benchmark-onnx:
	@test -n "$(CKPT)" || (echo "CKPT is required: path to the trained .ckpt" && exit 1)
	@test -n "$(ONNX_OUT)" || (echo "ONNX_OUT is required: path to the exported .onnx" && exit 1)
	uv run python -m personalitylinmult.inference.onnx_benchmark \
		--checkpoint $(CKPT) --onnx $(ONNX_OUT) --data config/FI/data/wavlm.yaml

# WavLM's own ONNX export: a separate, shared artifact (not tied to one
# champion id -- every wavlm-based champion uses the same backbone), needed
# for PersonalityModelONNX.predict_from_audio's full audio -> prediction
# pipeline to run without torch/transformers. ~380 MB; downloads WavLM
# base+ from HF once, at export time.
WAVLM_ONNX_OUT ?= .build/wavlm-base-plus.onnx

export-wavlm-onnx:
	mkdir -p $(dir $(WAVLM_ONNX_OUT))
	uv run python -c "\
from personalitylinmult.inference.onnx_export import export_wavlm_to_onnx; \
export_wavlm_to_onnx('$(WAVLM_ONNX_OUT)')"

push-wavlm-onnx:
	@test -f "$(WAVLM_ONNX_OUT)" || (echo "$(WAVLM_ONNX_OUT) not found -- run 'make export-wavlm-onnx' first" && exit 1)
	HF_XET_HIGH_PERFORMANCE=1 uv run hf upload $(HF_MODEL_REPO) \
		$(WAVLM_ONNX_OUT) wavlm-base-plus.onnx --repo-type model

# DINOv2's own ONNX export: a separate, shared artifact (not tied to one
# champion id), needed for the AVT champion's full media -> prediction
# pipeline. Downloads DINOv2 base from HF once, at export time.
DINOV2_ONNX_OUT ?= .build/dinov2-base.onnx

export-dinov2-onnx:
	mkdir -p $(dir $(DINOV2_ONNX_OUT))
	uv run python -c "\
from personalitylinmult.inference.onnx_export import export_dinov2_to_onnx; \
export_dinov2_to_onnx('$(DINOV2_ONNX_OUT)')"

push-dinov2-onnx:
	@test -f "$(DINOV2_ONNX_OUT)" || (echo "$(DINOV2_ONNX_OUT) not found -- run 'make export-dinov2-onnx' first" && exit 1)
	HF_XET_HIGH_PERFORMANCE=1 uv run hf upload $(HF_MODEL_REPO) \
		$(DINOV2_ONNX_OUT) dinov2-base.onnx --repo-type model

# mmBERT and XLM-RoBERTa (mpnet) ONNX exports: the two token-level text
# encoders behind the AVT champion's xlm_roberta_gt/mmbert_gt features.
# Shared artifacts, not tied to one champion id.
MMBERT_ONNX_OUT ?= .build/mmbert-base.onnx
XLM_ROBERTA_ONNX_OUT ?= .build/xlm-roberta-mpnet.onnx

export-mmbert-onnx:
	mkdir -p $(dir $(MMBERT_ONNX_OUT))
	uv run python -c "\
from personalitylinmult.inference.onnx_export import export_mmbert_to_onnx; \
export_mmbert_to_onnx('$(MMBERT_ONNX_OUT)')"

push-mmbert-onnx:
	@test -f "$(MMBERT_ONNX_OUT)" || (echo "$(MMBERT_ONNX_OUT) not found -- run 'make export-mmbert-onnx' first" && exit 1)
	HF_XET_HIGH_PERFORMANCE=1 uv run hf upload $(HF_MODEL_REPO) \
		$(MMBERT_ONNX_OUT) mmbert-base.onnx --repo-type model

export-xlm-roberta-onnx:
	mkdir -p $(dir $(XLM_ROBERTA_ONNX_OUT))
	uv run python -c "\
from personalitylinmult.inference.onnx_export import export_xlm_roberta_to_onnx; \
export_xlm_roberta_to_onnx('$(XLM_ROBERTA_ONNX_OUT)')"

push-xlm-roberta-onnx:
	@test -f "$(XLM_ROBERTA_ONNX_OUT)" || (echo "$(XLM_ROBERTA_ONNX_OUT) not found -- run 'make export-xlm-roberta-onnx' first" && exit 1)
	HF_XET_HIGH_PERFORMANCE=1 uv run hf upload $(HF_MODEL_REPO) \
		$(XLM_ROBERTA_ONNX_OUT) xlm-roberta-mpnet.onnx --repo-type model

# EmotiEffNet's own ONNX export: a separate, shared artifact (not tied to
# one champion id), needed for the AVT champion's full media -> prediction
# pipeline. Downloads and remaps the EmotiEffLib checkpoint once, at export
# time.
EMOTIEFFNET_ONNX_OUT ?= .build/emotieffnet-enet-b0-8-best-vgaf.onnx

export-emotieffnet-onnx:
	mkdir -p $(dir $(EMOTIEFFNET_ONNX_OUT))
	uv run python -c "\
from personalitylinmult.inference.onnx_export import export_emotieffnet_to_onnx; \
export_emotieffnet_to_onnx('$(EMOTIEFFNET_ONNX_OUT)')"

push-emotieffnet-onnx:
	@test -f "$(EMOTIEFFNET_ONNX_OUT)" || (echo "$(EMOTIEFFNET_ONNX_OUT) not found -- run 'make export-emotieffnet-onnx' first" && exit 1)
	HF_XET_HIGH_PERFORMANCE=1 uv run hf upload $(HF_MODEL_REPO) \
		$(EMOTIEFFNET_ONNX_OUT) emotieffnet-enet-b0-8-best-vgaf.onnx --repo-type model

# FaRL's own ONNX export: a separate, shared artifact (not tied to one
# champion id), needed for the AVT champion's full media -> prediction
# pipeline. Downloads and OpenAI-CLIP-to-HF-converts the FaRL checkpoint
# once, at export time.
FARL_ONNX_OUT ?= .build/farl-ep64.onnx

export-farl-onnx:
	mkdir -p $(dir $(FARL_ONNX_OUT))
	uv run python -c "\
from personalitylinmult.inference.onnx_export import export_farl_to_onnx; \
export_farl_to_onnx('$(FARL_ONNX_OUT)')"

push-farl-onnx:
	@test -f "$(FARL_ONNX_OUT)" || (echo "$(FARL_ONNX_OUT) not found -- run 'make export-farl-onnx' first" && exit 1)
	HF_XET_HIGH_PERFORMANCE=1 uv run hf upload $(HF_MODEL_REPO) \
		$(FARL_ONNX_OUT) farl-ep64.onnx --repo-type model

# emotion2vec+'s own ONNX export: a separate, shared artifact (not tied to
# one champion id), needed for the AVT champion's full media -> prediction
# pipeline. Downloads and remaps the emotion2vec+ seed checkpoint once, at
# export time.
EMOTION2VEC_ONNX_OUT ?= .build/emotion2vec-plus-seed.onnx

export-emotion2vec-onnx:
	mkdir -p $(dir $(EMOTION2VEC_ONNX_OUT))
	uv run python -c "\
from personalitylinmult.inference.onnx_export import export_emotion2vec_to_onnx; \
export_emotion2vec_to_onnx('$(EMOTION2VEC_ONNX_OUT)')"

push-emotion2vec-onnx:
	@test -f "$(EMOTION2VEC_ONNX_OUT)" || (echo "$(EMOTION2VEC_ONNX_OUT) not found -- run 'make export-emotion2vec-onnx' first" && exit 1)
	HF_XET_HIGH_PERFORMANCE=1 uv run hf upload $(HF_MODEL_REPO) \
		$(EMOTION2VEC_ONNX_OUT) emotion2vec-plus-seed.onnx --repo-type model

# Publish the HF model card, built from README.md itself (single source of
# truth -- no separate docs/model_card.md to keep in sync): the frontmatter
# HF renders, prepended to everything from "# Pretrained models" through the
# next top-level "# " heading, same pattern as push-hf-card.
HF_MODEL_CARD_BUILD ?= .build/hf_model_card.md
push-hf-model-card:
	mkdir -p $(dir $(HF_MODEL_CARD_BUILD))
	cat docs/_hf_model_frontmatter.yaml > $(HF_MODEL_CARD_BUILD)
	awk '/^# Pretrained models/{p=1} /^# Champion selection/{p=0} p' README.md >> $(HF_MODEL_CARD_BUILD)
	HF_XET_HIGH_PERFORMANCE=1 uv run hf upload $(HF_MODEL_REPO) \
		$(HF_MODEL_CARD_BUILD) README.md --repo-type model

# ── FI training ────────────────────────────────────────────────────────────────
# Two separate tasks over the same features: big-five trait regression (on
# emotional_stability, not neuroticism) and interview-variable regression.
# Requires `make hdf5-fi` first. Pass extra overrides via ARGS, e.g.
#   make train-fi-bigfive ARGS="--set data.window_seconds=10"
#
# Memory on Apple Silicon (MPS): unified memory is shared with the OS, so an
# oversized batch can OOM-kill the whole machine, not just the process -- MPS
# has no per-process cap the way CUDA's OOM does. PYTORCH_MPS_HIGH_WATERMARK_RATIO
# below makes MPS raise instead of exhausting all unified memory; 0.0 disables
# it entirely, so this project always sets an explicit ratio (< 1.0) instead.
# torch requires low <= high; its low-watermark default is 1.4 (soft, allows
# beyond recommendedMaxWorkingSetSize), so any high ratio below that is
# invalid unless PYTORCH_MPS_LOW_WATERMARK_RATIO is *also* set below it --
# both are set here together for exactly that reason.
# 0.9 leaves ~10% of unified memory outside PyTorch's own tracking for the
# rest of macOS -- if a run still takes the machine down at this ratio, drop
# both ratios back toward 0.7-0.8 for more headroom before reaching for a
# smaller batch or more accumulation.
# For a loss with per-batch statistics (e.g. "ccc") a large physical batch is
# what actually helps (stabler mean/var/covariance per step); if it will not
# fit even at a conservative ratio, use accumulate_grad_batches to reach the
# same *effective* batch size (data.batch_size * accumulate_grad_batches)
# while only ever holding one physical batch in memory, e.g.
#   make train-fi-bigfive ARGS="--set train.loss=ccc --set data.batch_size=64 \
#       --set train.accumulate_grad_batches=4"
# gives an effective batch of 256 at 64's memory footprint -- slower per epoch
# (4 forward passes per optimizer step) but the noisy-CCC fix without the OOM.

FI_DATA_CONFIG ?= config/FI/data/avt.yaml
FI_MODEL_CONFIG ?= config/FI/model/linmult.yaml
ARGS ?=
PYTORCH_MPS_HIGH_WATERMARK_RATIO ?= 0.9
PYTORCH_MPS_LOW_WATERMARK_RATIO ?= 0.8

train-fi-bigfive:
	PYTORCH_MPS_HIGH_WATERMARK_RATIO=$(PYTORCH_MPS_HIGH_WATERMARK_RATIO) \
	PYTORCH_MPS_LOW_WATERMARK_RATIO=$(PYTORCH_MPS_LOW_WATERMARK_RATIO) \
	uv run python -m personalitylinmult.train.cli \
		--data $(FI_DATA_CONFIG) --model $(FI_MODEL_CONFIG) \
		--train config/FI/train/bigfive.yaml $(ARGS)

train-fi-interview:
	uv run python -m personalitylinmult.train.cli \
		--data $(FI_DATA_CONFIG) --model $(FI_MODEL_CONFIG) \
		--train config/FI/train/interview.yaml $(ARGS)

# The MulT baseline: identical to LinMulT but for softmax attention.
train-fi-bigfive-mult:
	$(MAKE) train-fi-bigfive FI_MODEL_CONFIG=config/FI/model/mult.yaml

# ── Audio experiment group (Big Five) ───────────────────────────────────────────
# Three arms over the acoustic features in the SAME fi.h5 -- no rebuild:
#   train-fi-a          LinMulT, wavlm + emotion2vec (cross-modal fusion)
#   train-fi-wavlm      LinT,    wavlm only (single-stream)
#   train-fi-emotion2vec LinT,   emotion2vec only (single-stream)
# Each lands in its own MLflow experiment AND its own results/FI/bigfive/<experiment>
# subdir, so they keep separate checkpoints and each resumes independently.
# Experiment names follow {db}_{model}_{inputs} (inputs joined with "-" when
# there is more than one). `train-fi-audio-group` runs all three. Pass extra
# overrides via ARGS.
#
# Stop with Ctrl+C (loses only the in-progress epoch; each finished epoch is
# checkpointed). Resume the SAME run the next day by adding --resume, e.g.
#   make train-fi-a ARGS="--resume"
# which continues from last.ckpt and appends to the same MLflow run.
train-fi-a:
	$(MAKE) train-fi-bigfive FI_DATA_CONFIG=config/FI/data/a.yaml \
		ARGS="--set train.mlflow.experiment_name=fi_linmult_wavlm-emotion2vec $(ARGS)"

train-fi-wavlm:
	$(MAKE) train-fi-bigfive FI_DATA_CONFIG=config/FI/data/wavlm.yaml \
		FI_MODEL_CONFIG=config/FI/model/lint.yaml \
		ARGS="--set train.mlflow.experiment_name=fi_lint_wavlm $(ARGS)"

train-fi-emotion2vec:
	$(MAKE) train-fi-bigfive FI_DATA_CONFIG=config/FI/data/emotion2vec.yaml \
		FI_MODEL_CONFIG=config/FI/model/lint.yaml \
		ARGS="--set train.mlflow.experiment_name=fi_lint_emotion2vec $(ARGS)"

train-fi-audio-group: train-fi-a train-fi-wavlm train-fi-emotion2vec

# LR range test: probes a wide LR sweep and suggests a peak learning rate for
# a run, saving a loss-vs-lr plot under that run's own results dir (does not
# train the model). See personalitylinmult/train/lr_find.py.
lr-find-fi-wavlm:
	uv run python -m personalitylinmult.train.lr_find \
		--data config/FI/data/wavlm.yaml --model config/FI/model/lint.yaml \
		--train config/FI/train/bigfive.yaml \
		--set train.mlflow.experiment_name=fi_lint_wavlm $(ARGS)

# The analysis-window experiment. One h5 serves every window: only the
# dataloader's crop changes, so no rebuild is needed.
train-fi-windows:
	for w in 15 10 5; do \
		$(MAKE) train-fi-bigfive ARGS="--set data.window_seconds=$$w \
			--set train.mlflow.run_name=bigfive-window-$${w}s"; \
	done

train-fi-smoke:
	uv run python -m personalitylinmult.train.cli \
		--data config/FI/data/smoke.yaml --model $(FI_MODEL_CONFIG) \
		--train config/FI/train/smoke.yaml --fast-dev-run $(ARGS)

# ── FI feature ablation ─────────────────────────────────────────────────────────
# The ablation matrix (personalitylinmult/train/ablation.py): 7 unimodal LinT
# baselines + 3 within-modality LinMulT runs + 1 full LinMulT run, all on ASR
# text. Each run is the shared avt/model/train configs with the run's overrides
# applied -- editing the study is editing ablation.py, not this Makefile.
#
# Smoke first (one batch per arm, proves every arm's data->model->loss path),
# then the real sweep. Requires `make hdf5-fi` first.
ablation-smoke:
	uv run python -m personalitylinmult.train.run_ablation --fast-dev-run $(ARGS)

ablation:
	uv run python -m personalitylinmult.train.run_ablation $(ARGS)

mlflow-ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

# ── Model deployment (MLflow registry + serving) ────────────────────────────────
# Wrap the trained checkpoints as pyfunc models and register them (named
# {db}_{model}_{inputs}: fi_linmult_wavlm-emotion2vec, fi_lint_wavlm,
# fi_lint_emotion2vec), then serve one as a REST endpoint or bake it into a
# container. Run `register-models` once after training; promote a specific
# version to an alias (`promote-champion`) as a separate, deliberate step --
# registering a version never auto-promotes it. Serving/Docker resolve
# MODEL_URI via the alias, so promoting a new champion needs no other change.
#
# Aliases replace MLflow's deprecated Staging/Production stages: an alias is
# scoped to one registered-model name, so each model family (AVT, audio-only,
# ...) carries its own independent @champion with no conflict.
MODEL ?= fi_lint_wavlm-ccc-bs256
MODEL_ALIAS ?= champion
MODEL_URI ?= models:/$(MODEL)@$(MODEL_ALIAS)
SERVE_PORT ?= 5001

register-models:
	uv run python -m personalitylinmult.inference.register_models

# Promote an already-registered version to an alias, e.g.
#   make promote-champion MODEL=fi_lint_wavlm VERSION=3
VERSION ?=
promote-champion:
	uv run python -c "\
from personalitylinmult.inference.mlflow_pyfunc import set_champion; \
import mlflow; mlflow.set_tracking_uri('sqlite:///mlflow.db'); \
set_champion('$(MODEL)', '$(VERSION)', alias='$(MODEL_ALIAS)')"

# REST endpoint at http://127.0.0.1:$(SERVE_PORT)/invocations. POST a payload of
# {"inputs": [{"wavlm": [[...]], "emotion2vec": [[...]]}]}.
serve-model:
	uv run mlflow models serve -m "$(MODEL_URI)" -p $(SERVE_PORT) --env-manager uv

# ── REST serving (custom FastAPI, ONNX-only) ────────────────────────────────────
# `mlflow models build-docker` was tried and abandoned for Cloud Run: it
# spawns a second, isolated virtualenv at container startup to run the actual
# server (defeating any site-packages fix), its nginx layer failed silently
# under Cloud Run specifically, and mlflow's scoring server never binds
# 0.0.0.0 by default. deploy/serve.py replaces it: one process, no nginx, no
# extra virtualenv, ONNX Runtime only (no torch/lightning at runtime) -- see
# README.md's "Deployments" section for the full diagnosis. Two endpoints:
# POST /invocations (pre-extracted features) and POST /predict_audio (raw
# audio/video upload, full decode->WavLM->LinT pipeline).
#
# The image is identical whether run locally or deployed -- `docker run` it
# on your own machine to get the exact same server Cloud Run runs, no
# separate "local" variant.
SERVE_IMAGE ?= personalitylinmult-serve:onnx
SERVE_PORT_LOCAL ?= 8080
# Defaults to MODEL (the same champion id build-serve-docker/push-serve-gcloud/
# deploy-serve-cloudrun already key off) so `make deploy-serve-cloudrun
# MODEL=avt_best-ccc` deploys avt_best-ccc's image *and* runs it with
# MODEL_ID=avt_best-ccc -- one variable to set, not two kept in sync by hand.
# Override explicitly only if the image and the runtime MODEL_ID genuinely
# need to differ.
SERVE_MODEL_ID ?= $(MODEL)

build-serve-docker:
	docker build --platform linux/amd64 -f deploy/Dockerfile -t $(SERVE_IMAGE) .

run-serve-docker:
	docker run --rm -p $(SERVE_PORT_LOCAL):8080 -e MODEL_ID=$(SERVE_MODEL_ID) $(SERVE_IMAGE)

# ── Cloud Run deployment ─────────────────────────────────────────────────────────
# Serve the built image as a real, publicly reachable REST endpoint, scaled to
# zero when idle (no billed idle state -- forgetting to tear it down costs
# nothing, since zero instances run when there is no traffic). Chosen over
# Fly.io/Render for a standing, non-trial free tier and true scale-to-zero.
#
# One-time setup, once GCP_PROJECT exists (`gcloud auth login` first, then
# `gcloud projects create <id>`, `gcloud config set project <id>`, and link a
# billing account -- required even for free-tier usage, but the free tier
# itself never expires or silently converts to billing):
#   make gcloud-setup GCP_PROJECT=<id>
# Also set up a Cloud Billing budget alert (console.cloud.google.com/billing ->
# Budgets & alerts, free, e.g. alert at $1) as a one-time safety net.
#
# Per-champion flow: build-serve-docker -> push-serve-gcloud -> deploy-serve-cloudrun.
# Kept as three separate targets (not chained) so a failed build or push is
# caught before a deploy is attempted against a stale or missing image.
GCP_PROJECT ?=
GCP_REGION ?= europe-west1
AR_REPO ?= personalitylinmult
# 4Gi covers the audio-only champion (one WavLM ONNX graph). The AVT champion
# needs a lot more -- six extractor ONNX graphs plus PyTorch face detection/
# tracking models loaded at once genuinely OOM'd both a 4Gi instance (4108
# MiB used against a 4096 MiB limit) and an 8Gi instance (8321 MiB used and
# still climbing) in practice. avt_best-ccc is therefore NOT deployed to
# Cloud Run in this project -- the 16-32Gi tier that would fit costs real
# money per request, outside the free-tier budget this deploy targets (see
# README's Deployments section and deploy/serve.py's module docstring). It
# remains available via the HF Space (ZeroGPU, free) and the plain `pip
# install` API. Pass SERVE_MEMORY=16Gi (or higher) yourself if you want to
# deploy avt_best-ccc here anyway and are fine paying for the bigger instance.
SERVE_MEMORY ?= 4Gi
AR_IMAGE = $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT)/$(AR_REPO)/$(MODEL):onnx
# Cloud Run service names allow only lowercase alphanumerics and dashes (no
# underscores) -- MODEL names use underscores (e.g. fi_lint_wavlm-ccc-bs256),
# so the service name substitutes dashes for this one use, without touching
# MODEL itself (still used verbatim for the image tag/registered-model name).
CLOUDRUN_SERVICE = personalitylinmult-$(subst _,-,$(MODEL))

gcloud-setup:
	@test -n "$(GCP_PROJECT)" || (echo "GCP_PROJECT is required, e.g. GCP_PROJECT=my-project-id" && exit 1)
	gcloud config set project $(GCP_PROJECT)
	gcloud services enable run.googleapis.com artifactregistry.googleapis.com
	gcloud artifacts repositories create $(AR_REPO) \
		--repository-format=docker --location=$(GCP_REGION) \
		--description="PersonalityLinMulT model images" || true

push-serve-gcloud:
	@test -n "$(GCP_PROJECT)" || (echo "GCP_PROJECT is required, e.g. GCP_PROJECT=my-project-id" && exit 1)
	gcloud auth configure-docker $(GCP_REGION)-docker.pkg.dev --quiet
	docker tag $(SERVE_IMAGE) $(AR_IMAGE)
	docker push $(AR_IMAGE)

# --min-instances=0 is non-negotiable: the one setting that would otherwise
# turn "scale to zero" into "always on and billed". --max-instances=3 caps
# worst-case spend under a traffic spike, not idle time. --memory (see
# SERVE_MEMORY above) --cpu=2 is more than the audio-only champion strictly
# needs, but is kept generous rather than re-tuned per image (the default
# 512Mi caused an OOM crash loop under the earlier torch+lightning-based
# image; the AVT champion needs more still -- see SERVE_MEMORY's comment).
deploy-serve-cloudrun:
	@test -n "$(GCP_PROJECT)" || (echo "GCP_PROJECT is required, e.g. GCP_PROJECT=my-project-id" && exit 1)
	gcloud run deploy $(CLOUDRUN_SERVICE) \
		--image $(AR_IMAGE) \
		--region $(GCP_REGION) \
		--port=8080 \
		--memory=$(SERVE_MEMORY) --cpu=2 --timeout=300 \
		--min-instances=0 \
		--max-instances=3 \
		--set-env-vars MODEL_ID=$(SERVE_MODEL_ID) \
		--allow-unauthenticated

# ── Misc ───────────────────────────────────────────────────────────────────────

clean:
	rm -rf .venv coverage_html dist/ .pytest_cache/ site/ tmp/
	rm -f .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
