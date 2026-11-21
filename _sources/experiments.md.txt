# Baselines, eGeMAPS and a WavLM probe

Small experiments around the `wavlm_best-ccc` champion. All Big Five, seed 42, one run per
row, test split. Every trained row uses the champion recipe (AdamW 5e-3, weight decay 0.05,
cosine schedule with 15% warmup, gradient clip 1.0, early stopping on `valid/mean_ccc`,
batch 256 effective). Only the loss or the input differs, as stated per row.

The eGeMAPS features and the experiment HDF5 are local, temporary artifacts: they are not
published and not part of the reproducible pipeline.

## Results

| Row | Input | Loss | 1-MAE | CCC | Pearson r | std_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| `avt_best-ccc` | all 7 features | ccc+l1 | 0.9074 | 0.6901 | 0.7086 | 0.9924 |
| `wavlm_best-ccc` | WavLM layer 9 | ccc | 0.8918 | 0.5604 | 0.5609 | 1.0013 |
| wavlm + eGeMAPS | WavLM + eGeMAPS LLD | ccc | 0.8856 | 0.5543 | 0.5600 | 1.0929 |
| eGeMAPS | eGeMAPS LLD | ccc | 0.8710 | 0.4908 | 0.5062 | 1.1519 |
| eGeMAPS | eGeMAPS LLD | l1 | 0.8973 | 0.4461 | 0.4932 | 0.6378 |
| random input | N(0, 1) noise, resampled every access | l1 | 0.8810 | 0.0005 | 0.0039 | 0.0657 |
| train mean | constant | none | 0.8814 | 0.0000 | n/a | 0.0000 |

* **Train-mean.** The constant prediction has zero variance, so Pearson r is
  0/0 (undefined) and CCC is exactly 0. Its 1-MAE of 0.8814 is the number any
  model has to beat, and it is why 1-MAE alone is a weak metric here.
* **Random input.** With an input unrelated to the target the network converges to
  the mean: 1-MAE 0.8810, CCC about 0, predictions with 6.6% of the target spread.
* **eGeMAPS, loss comparison.** The same input and model, trained with `l1` or `ccc`,
  lands in different places: `l1` gives the best 1-MAE (0.8973, above WavLM) but
  under-spread predictions (std_ratio 0.64), while `ccc` gives higher correlation
  (0.4908) but over-spread predictions (std_ratio 1.15, negative R2). The loss decides
  which part of the target distribution the model reproduces.
* **eGeMAPS vs WavLM.** Low-level prosodic and spectral descriptors carry
  signal (CCC 0.49) but clearly less than WavLM (0.56).
* **Fusion.** Adding eGeMAPS to WavLM (`LinMulT`, two streams) does not help: CCC 0.5543
  vs 0.5604 for WavLM alone, and 1-MAE 0.8856 vs 0.8918. With one seed per row the small
  drop is within run-to-run noise, so the safe reading is "no gain". Together with the probe
  below, this says the eGeMAPS information is largely contained in WavLM rather than
  complementary to it.

## Linear probe: can WavLM reconstruct eGeMAPS?

Ridge regression (closed form, penalty chosen on validation) from the 768-d WavLM
layer-9 frame to the 25 z-scored eGeMAPS LLDs. eGeMAPS (100 Hz) is average-pooled
2:1 to WavLM's 50 Hz; padding is masked, unvoiced frames are kept. Test split.

| Probe | Train clips | Mean R2 | Mean r |
| --- | --- | --- | --- |
| train mean (control) | all | -0.0001 | 0.000 |
| WavLM layer 9 → eGeMAPS | all | 0.380 | 0.586 |
| WavLM layer 9 → eGeMAPS | 1000 | 0.376 | 0.582 |
| random-init WavLM → eGeMAPS | 1000 | 0.278 | 0.487 |
| eGeMAPS → WavLM layer 9 (reverse) | all | 0.036 | 0.176 |

* WavLM linearly recovers a substantial part of eGeMAPS (R2 0.38), and the
  1000-clip subset gives the same number as the full train set.
* An untrained, randomly initialised WavLM already reaches 0.28, so the
  trained model's own contribution is the gap above that (about +0.10).
* The reverse direction is weak (R2 0.04): 25 hand-designed descriptors explain
  almost none of WavLM's 768 dimensions. WavLM contains much of eGeMAPS and a lot more.

### Per-LLD test R2

| LLD | WavLM (all train) | WavLM (1000 clips) | random-init (1000 clips) |
| --- | --- | --- | --- |
| `mfcc1_sma3` | 0.698 | 0.695 | 0.482 |
| `mfcc2_sma3` | 0.666 | 0.661 | 0.423 |
| `mfcc3_sma3` | 0.630 | 0.626 | 0.236 |
| `Loudness_sma3` | 0.561 | 0.556 | 0.558 |
| `mfcc4_sma3` | 0.557 | 0.554 | 0.333 |
| `alphaRatio_sma3` | 0.555 | 0.550 | 0.559 |
| `hammarbergIndex_sma3` | 0.543 | 0.538 | 0.460 |
| `F1amplitudeLogRelF0_sma3nz` | 0.520 | 0.518 | 0.413 |
| `F2amplitudeLogRelF0_sma3nz` | 0.514 | 0.513 | 0.406 |
| `F3amplitudeLogRelF0_sma3nz` | 0.506 | 0.504 | 0.398 |
| `HNRdBACF_sma3nz` | 0.498 | 0.495 | 0.362 |
| `slope500-1500_sma3` | 0.494 | 0.490 | 0.276 |
| `F0semitoneFrom27.5Hz_sma3nz` | 0.494 | 0.491 | 0.396 |
| `spectralFlux_sma3` | 0.360 | 0.355 | 0.345 |
| `logRelF0-H1-A3_sma3nz` | 0.360 | 0.357 | 0.251 |
| `slope0-500_sma3` | 0.317 | 0.307 | 0.475 |
| `F1bandwidth_sma3nz` | 0.275 | 0.270 | 0.175 |
| `shimmerLocaldB_sma3nz` | 0.206 | 0.203 | 0.126 |
| `F2bandwidth_sma3nz` | 0.201 | 0.198 | 0.069 |
| `F2frequency_sma3nz` | 0.118 | 0.110 | 0.049 |
| `F3frequency_sma3nz` | 0.109 | 0.101 | 0.034 |
| `F3bandwidth_sma3nz` | 0.099 | 0.096 | 0.020 |
| `F1frequency_sma3nz` | 0.077 | 0.072 | 0.043 |
| `logRelF0-H1-H2_sma3nz` | 0.076 | 0.074 | 0.041 |
| `jitterLocal_sma3nz` | 0.056 | 0.054 | 0.025 |
