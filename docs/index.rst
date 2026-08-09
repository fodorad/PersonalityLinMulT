PersonalityLinMulT
===================

**Transformer-based Big Five Personality Perception and Interview-Variable Estimation**

PersonalityLinMulT builds on `LinMulT <https://github.com/fodorad/linmult>`_. It is
trained on the First Impressions V2 dataset for two separate regression tasks:
perceived Big Five traits (Automatic Personality Perception), and the interview
(hireability) variable.

* paper: **Multimodal Sentiment and Personality Perception Under Speech: A
  Comparison of Transformer-based Architectures**
  (`pdf <https://proceedings.mlr.press/v173/fodor22a/fodor22a.pdf>`_,
  `website <https://proceedings.mlr.press/v173/fodor22a.html>`_)

**Two promoted, deployed champions ship with this repository:**
``wavlm_best-ccc`` (acoustic only, LinT) and ``avt_best-ccc`` (acoustic +
visual + textual fusion of all 7 features, LinMulT). Both are fully wired
into every surface below -- the public API, ONNX export, Docker/Cloud Run
server, HF Space, Gradio demo and showcase notebook -- end-to-end. The
dataset (7 features across all three modalities), training code, and
ablation matrix support any other single-feature or fusion combination the
same way -- ``LinT`` for any single feature, ``LinMulT`` for any fusion --
so a visual-only or textual-only champion is trained and promoted the same
way once a run is selected (see `Champion selection`_).

What's in this repository
--------------------------

- **Training** -- LinT (single-stream) and LinMulT (cross-modal fusion) over
  any subset of 7 features, configurable losses (elementwise, CCC, and a
  learned-uncertainty-weighted combination), MLflow-tracked, Optuna HPO,
  feature ablations.
- **Metrics** -- CCC (primary), 1-MAE, Pearson r, R\ :sup:`2`, and
  ``std_ratio`` (prediction-vs-target variance ratio, catching
  regression-to-the-mean that 1-MAE alone cannot see).
- **Model lifecycle** -- MLflow Model Registry with alias-based promotion
  (``@champion``, not deprecated stages), so a specific trained run becomes
  "the" deployable model through a deliberate, auditable step.
- **Public inference API** -- ``pip install personalitylinmult``, then
  ``PersonalityModel.from_pretrained("wavlm_best-ccc")``: downloads a
  champion from the public HF Hub model repo and predicts, no MLflow or repo
  clone needed. An optional ``[onnx]`` extra swaps in
  ``PersonalityModelONNX`` for a PyTorch/Lightning-free runtime, including a
  full raw-audio-to-prediction pipeline on ONNX Runtime alone -- measurably
  faster end-to-end (see `ONNX optimization`_).
- **Deployment** -- a Docker image (``deploy/Dockerfile``) serving the ONNX
  path, runnable identically on ``localhost`` or on Google Cloud Run (true
  scale-to-zero, no idle billing), plus a Hugging Face Spaces Gradio demo
  with a live PyTorch-vs-ONNX backend switch.

Setup
-----

Install for inference:

.. code-block:: bash

   pip install personalitylinmult

.. code-block:: python

   from personalitylinmult import PersonalityModel

   model = PersonalityModel.from_pretrained("wavlm_best-ccc")
   scores = model.predict({"wavlm": wavlm_features})  # {trait: score in [0, 1]}

``model.feature_names`` lists exactly which features a given checkpoint
needs -- a checkpoint is self-describing, so the same ``PersonalityModel``
class serves every architecture. See `Pretrained models`_ below for which
ids are actually available.

Install for training:

.. code-block:: bash

   git clone https://github.com/fodorad/PersonalityLinMulT
   cd PersonalityLinMulT
   make train        # training + the HDF5 builder
   make preprocess   # only if re-extracting features from raw video

.. note::

   Use ``uv`` (as the Makefile does) for the ``preprocess`` extra: it applies
   the ``[tool.uv]`` dependency overrides required to resolve the
   exordium/linmult/blinklinmult stack. Plain ``pip`` ignores those
   overrides and will fail.

Data preparation
-----------------

The First Impressions V2 (FI) dataset is not redistributed with this
repository due to its license. Feature extraction writes per-clip files
under ``data/processed/FI``; the builder then collects **only the features
named in the data config** into a single HDF5 file, and training reads that
file and nothing else:

.. code-block:: bash

   make preprocess-fi   # decode video/audio, write label.csv
   make features-fi     # run the feature extractors (GPU, hours)
   make hdf5-fi          # collect them into data/processed/FI/h5/fi.h5

Its root attributes embed the builder config verbatim, the commit SHA, and
the label CSV's hash, so a built file is always traceable to what produced
it. The raw video tree and intermediate feature directories are never
redistributed; only the compact HDF5 is, published to a public Hugging Face
dataset repo (see :doc:`dataset`):

.. code-block:: bash

   make pull-fi-h5   # fast path: download the built fi.h5, train immediately
   make pipeline-fi  # full path: rebuild fi.h5 from raw FI, for verification

Training
--------

Two separate tasks over the same features:

.. code-block:: bash

   make hdf5-fi              # build the artifact first
   make train-fi-bigfive     # Big Five trait regression
   make train-fi-interview   # interview-variable regression
   make mlflow-ui            # monitor runs at http://localhost:5000

Big Five regression trains on ``emotional_stability`` (``= 1 - neuroticism``)
so that every trait points the same way; naming ``neuroticism`` as a target
is rejected. The interview variable is a hireability judgement rather than a
personality trait, so it is trained as its own task with its own head.

Metrics
~~~~~~~

The primary metric is per-trait **CCC** (Lin's concordance correlation),
summarised as the mean of the five independent trait scores. That mean
(``valid/mean_ccc``) selects checkpoints and drives early stopping. 1-MAE,
Pearson *r*, R\ :sup:`2`, and MSE are reported per trait alongside it.

CCC, not 1-MAE, is primary because **1-MAE is misleading exactly when it
matters most**: the FI trait distributions cluster tightly around their
mean, so a model that always predicts the population mean regardless of
input already scores a deceptively good 1-MAE -- it never commits to an
extreme prediction, so it never pays a large per-sample penalty, even though
it has learned nothing about the input. This is regression-to-the-mean, and
1-MAE alone cannot detect it. CCC penalizes exactly this failure mode -- it
requires both correct correlation with the target *and* the target's
variance to be reproduced (``std_ratio`` makes the variance-collapse
component directly visible). This project's own experiments confirmed the
effect directly: switching the loss/selection metric from an elementwise
objective to CCC at a large enough batch size raised ``std_ratio`` from
~0.85 to ~1.00 while 1-MAE barely moved.

.. list-table::
   :header-rows: 1

   * - metric
     - meaning
   * - ``{split}/ccc/{trait}``
     - agreement including bias and scale -- **primary**
   * - ``{split}/mean_ccc``
     - mean of the five trait scores -- **model selection**
   * - ``{split}/one_minus_mae/{trait}``
     - per-trait 1 - MAE -- reported, not selected on
   * - ``{split}/pearson_r/{trait}``
     - ranks clips correctly, ignoring bias
   * - ``{split}/r2/{trait}``
     - variance explained vs. predicting the mean

Feature ablation
~~~~~~~~~~~~~~~~~

A predefined matrix (``personalitylinmult/train/ablation.py``) measures what
each feature and modality contributes: **7 unimodal baselines** -- one per
feature, each a single-stream ``LinT`` -- plus **3 within-modality**
(all-acoustic, all-visual, all-textual) and **1 full** multimodal ``LinMulT``
run.

.. code-block:: bash

   make ablation-smoke   # one batch per arm -- the pre-flight for a sweep
   make ablation         # run the whole matrix, logging each arm to MLflow

Pretrained models
------------------

``pip install personalitylinmult`` and
``PersonalityModel.from_pretrained(model_id)`` downloads directly from the
public `HF Hub model repo <https://huggingface.co/fodorad/personalitylinmult-fi-models>`_
-- no MLflow, no repo clone. Every id below follows
``{features joined by "_", multiword names use "-"}_best-{metric}`` (e.g.
``wavlm_best-ccc``); the architecture (LinT vs. LinMulT) is never part of
the id -- a checkpoint is self-describing.

.. list-table::
   :header-rows: 1

   * - Model id
     - Modality
     - Architecture
     - Test 1-MAE
     - Test CCC
     - Test Pearson r
     - Test std_ratio
   * - ``avt_best-ccc``
     - acoustic + visual + textual (all 7 features)
     - LinMulT
     - **0.9074**
     - **0.6901**
     - **0.7086**
     - 0.9924
   * - ``wavlm_best-ccc``
     - acoustic (WavLM) only
     - LinT
     - 0.8918
     - 0.5604
     - 0.5609
     - **1.0013**

Both are promoted, real champions, and both are fully wired into every live
surface (public API, ONNX export, Docker/Cloud Run server, HF Space,
showcase notebook). Full multimodal fusion (``avt_best-ccc``) beats
audio-only on every accuracy metric (1-MAE, CCC, Pearson r), confirming
visual and textual signal genuinely add information beyond speech alone;
audio-only edges it out on ``std_ratio`` (both close to the ideal of 1.0).

Champion selection
-------------------

A trained run only becomes "the" deployable model through a deliberate,
auditable promotion step -- nothing is ever promoted automatically:

.. code-block:: bash

   make register-models                          # wrap a trained checkpoint as an MLflow pyfunc model
   make promote-champion MODEL=<name> VERSION=<n> # move the @champion alias to it

``@champion`` replaces MLflow's deprecated Staging/Production stages -- it
is scoped per registered-model *name*, so independent champions
(acoustic-only, visual-only, text-only, full AVT fusion) each carry their
own ``@champion`` with no conflict. Once promoted, publish it to the public
HF Hub model repo:

.. code-block:: bash

   make push-champion-model MODEL_ID=wavlm_best-ccc CKPT=<path/to/best.ckpt>

ONNX optimization
-------------------

An optional, PyTorch/Lightning-free inference path (``[onnx]`` extra),
applied as a post-training, post-promotion step:

.. code-block:: bash

   pip install personalitylinmult[onnx]

.. code-block:: python

   from personalitylinmult.onnx import PersonalityModelONNX

   model = PersonalityModelONNX.from_pretrained("wavlm_best-ccc")
   scores = model.predict_from_audio("clip.wav")  # raw audio/video -> Big Five,
                                                   # no torch/transformers/exordium

The full multimodal champion has its own full-pipeline method,
``predict_from_media``, since it needs face detection/tracking and
(optionally) Whisper ASR in addition to six ONNX feature extractors -- those
two steps stay ``torch``/``exordium``-based, everything else runs on ONNX
Runtime alone:

.. code-block:: python

   model = PersonalityModelONNX.from_pretrained("avt_best-ccc")
   scores = model.predict_from_media("clip.mp4")               # transcript=None -> Whisper ASR
   scores = model.predict_from_media("clip.mp4", transcript="hello world")  # explicit transcript
   scores = model.predict_from_media("clip.mp4", transcript="")  # skip text (placeholder + all-False mask)

The **full pipeline** (raw audio -> WavLM -> LinT), where WavLM's tens of
millions of parameters actually benefit from ONNX Runtime's graph
optimizations, is **~6-8x faster end-to-end** once warmed up -- why
``deploy/serve.py`` runs the ONNX path exclusively, while the Gradio demo
offers both behind a switch.

Notebook comparisons and demo
-------------------------------

``notebooks/showcase.ipynb`` -- three parts: (1) loads the audio-only
champion, extracts real features from a local FI test clip, and plots
predicted vs. ground truth; (2) a PyTorch vs. ONNX comparison on a synthetic
``ffmpeg``-generated fixture clip; (3) the full multimodal (AVT) champion's
``predict_from_media`` on a synthetic video fixture.

**Hugging Face Space (Gradio)** -- ``app/app.py``,
``uv run --extra demo --extra onnx python app/app.py`` to run locally. Pick
a model and backend from two dropdowns, upload a clip, and optionally
supply a transcript for the AVT champion.

Deployments
-----------

The same small FastAPI server (``deploy/serve.py``, ONNX Runtime only) runs
identically whether started directly on ``localhost`` via Docker, or
deployed to Google Cloud Run:

.. code-block:: bash

   make build-serve-docker            # docker build -f deploy/Dockerfile ...
   make run-serve-docker              # docker run -p 8080:8080 ...
   curl http://localhost:8080/ping
   curl -X POST http://localhost:8080/predict_audio -F file=@clip.wav
   curl -X POST http://localhost:8080/predict_media -F file=@clip.mp4 -F transcript="hello world"

.. code-block:: bash

   make gcloud-setup GCP_PROJECT=<id>    # one-time, per GCP project
   make push-serve-gcloud GCP_PROJECT=<id>
   make deploy-serve-cloudrun GCP_PROJECT=<id>

Each deployed instance serves exactly one champion, selected by the
``MODEL_ID`` environment variable; three endpoints exist on every instance
-- ``POST /invocations`` (pre-extracted features), ``POST /predict_audio``
(audio-only champion), and ``POST /predict_media`` (full AVT champion) --
but only the endpoint matching the loaded champion's ``feature_names`` will
actually succeed.

.. toctree::
   :maxdepth: 1
   :caption: Dataset

   dataset

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   autoapi/personalitylinmult/index

Related projects
------------------

exordium
~~~~~~~~

Collection of preprocessing functions and deep learning methods. This
repository contains revised codes for fine landmark detection (including
face, eye region, iris and pupil landmarks), head pose estimation, and eye
feature calculation.

* code: https://github.com/fodorad/exordium

(2022) LinMulT
~~~~~~~~~~~~~~~

General-purpose Multimodal Transformer with Linear Complexity Attention
Mechanism. This base model is further modified and trained for various
tasks and datasets.

* code: https://github.com/fodorad/LinMulT

(2023) BlinkLinMulT
~~~~~~~~~~~~~~~~~~~~

LinMulT is trained for blink presence detection and eye state recognition
tasks. Our results demonstrate comparable or superior performance compared
to state-of-the-art models on 2 tasks, using 7 public benchmark databases.

* paper: BlinkLinMulT: Transformer-based Eye Blink Detection
  (`pdf <https://adamfodor.com/pdf/2023_Fodor_Adam_MDPI_BlinkLinMulT.pdf>`_,
  `website <https://www.mdpi.com/2313-433X/9/10/196>`_)
* code: https://github.com/fodorad/BlinkLinMulT

Citation
--------

If you found our research helpful or influential please consider citing:

(2023) BlinkLinMulT for blink presence detection and eye state recognition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bibtex

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

(2022) LinMulT for personality trait and sentiment estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bibtex

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

Contact
-------

* Ádám Fodor (fodorad201@gmail.com) -- `adamfodor.com <https://adamfodor.com>`_
