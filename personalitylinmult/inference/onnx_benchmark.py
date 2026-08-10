r"""Compare a champion's PyTorch and ONNX Runtime paths: speed and accuracy.

Answers the two questions ONNX export exists to answer: is it actually
faster, and how much (if any) test-set accuracy does the export cost. Runs
both paths over the real test split from a built ``fi.h5``, using the exact
metrics suite training reports (1-MAE, Pearson r, CCC, R2, std_ratio) so the
comparison is apples-to-apples against the numbers already recorded for the
champion in README.md's "Pretrained models" table.

Usage::

    uv run python -m personalitylinmult.inference.onnx_benchmark \\
        --checkpoint results/FI/bigfive/fi_lint_wavlm-ccc-bs256/checkpoints/best.ckpt \\
        --onnx results/FI/bigfive/fi_lint_wavlm-ccc-bs256/checkpoints/best.onnx \\
        --data config/FI/data/wavlm.yaml
"""

from __future__ import annotations

import argparse
import logging
import time

import torch

logger = logging.getLogger(__name__)
"""Module-level logger."""


def _time_calls(fn, n: int) -> float:
    """Return the mean wall-clock seconds per call, over ``n`` calls.

    Args:
        fn: A zero-argument callable to time.
        n (int): Number of calls to average over.

    Returns:
        float: Mean seconds per call.
    """
    start = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - start) / n


def run_benchmark(
    checkpoint_path: str, onnx_path: str, data_path: str, n_timing_runs: int = 20
) -> dict:
    """Benchmark a champion's PyTorch vs. ONNX Runtime paths.

    Args:
        checkpoint_path (str): Path to the trained ``.ckpt``.
        onnx_path (str): Path to the exported ``.onnx`` (with a
            ``.onnx.json`` metadata sidecar alongside it).
        data_path (str): Path to the data config (e.g.
            ``config/FI/data/wavlm.yaml``) whose ``fi.h5``/test split this
            benchmark evaluates against.
        n_timing_runs (int): Forward passes averaged for the per-sample
            latency numbers (accuracy always runs over the full test split
            regardless of this value).

    Returns:
        dict: ``{"pytorch": {...}, "onnx": {...}}``, each with ``latency_ms``
        and the metrics suite (``mean_one_minus_mae``, ``mean_ccc``, etc.).
    """
    import onnxruntime as ort

    from personalitylinmult.data.collate import unpack_batch
    from personalitylinmult.data.datamodule import FiDataModule
    from personalitylinmult.train.config import DataConfig, load_yaml
    from personalitylinmult.train.metrics import TraitRegressionMetrics, clamp_predictions
    from personalitylinmult.train.module import FiLightningModule

    # map_location="cpu": a checkpoint saved from an MPS/CUDA run otherwise
    # keeps its weights on that device, which then mismatches CPU-resident
    # batches from the dataloader below.
    module = FiLightningModule.load_from_checkpoint(checkpoint_path, map_location="cpu")
    module.eval()
    if not module.is_lint:
        raise ValueError("ONNX benchmarking currently supports only single-feature (LinT) models.")

    # label_csv drives the h5 builder, not DataConfig -- see
    # ExperimentConfig.from_files, which strips it the same way.
    raw_data = load_yaml(data_path)
    raw_data.pop("label_csv", None)
    data_config = DataConfig.from_dict(raw_data)
    datamodule = FiDataModule(data_config, target_names=list(module.train_config.targets))
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    torch_metrics = TraitRegressionMetrics(list(module.train_config.targets))
    onnx_metrics = TraitRegressionMetrics(list(module.train_config.targets))

    first_batch: tuple[torch.Tensor, torch.Tensor] | None = None
    with torch.no_grad():
        for batch in test_loader:
            inputs, masks, _ = unpack_batch(batch, module.feature_names)
            if first_batch is None:
                first_batch = (inputs[0][:1], masks[0][:1])

            torch_preds = clamp_predictions(module(inputs, masks))
            torch_metrics.update(torch_preds, batch["targets"])  # ty: ignore[invalid-argument-type]

            x = inputs[0].numpy().astype("float32")
            mask = masks[0].numpy().astype(bool)
            (onnx_raw,) = session.run(["predictions"], {"x": x, "mask": mask})
            onnx_preds = clamp_predictions(torch.from_numpy(onnx_raw))
            onnx_metrics.update(onnx_preds, batch["targets"])  # ty: ignore[invalid-argument-type]

    if first_batch is None:
        raise ValueError(f"The test split at {data_path!r} has no clips.")
    torch_x, torch_mask = first_batch
    torch_latency = _time_calls(lambda: module([torch_x], [torch_mask]), n_timing_runs)
    onnx_x = torch_x.numpy().astype("float32")
    onnx_mask = torch_mask.numpy().astype(bool)
    onnx_latency = _time_calls(
        lambda: session.run(["predictions"], {"x": onnx_x, "mask": onnx_mask}), n_timing_runs
    )

    def _summarize(metrics: TraitRegressionMetrics, latency_s: float) -> dict:
        computed = metrics.compute()  # ty: ignore[missing-argument]
        return {
            "latency_ms": latency_s * 1000,
            "mean_one_minus_mae": float(computed["mean_one_minus_mae"]),
            "mean_pearson_r": float(computed["mean_pearson_r"]),
            "mean_ccc": float(computed["mean_ccc"]),
            "mean_r2": float(computed["mean_r2"]),
            "mean_std_ratio": float(computed["mean_std_ratio"]),
        }

    return {
        "pytorch": _summarize(torch_metrics, torch_latency),
        "onnx": _summarize(onnx_metrics, onnx_latency),
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="Path to the trained .ckpt.")
    parser.add_argument("--onnx", required=True, help="Path to the exported .onnx.")
    parser.add_argument("--data", required=True, help="Path to the data config.")
    parser.add_argument("--timing-runs", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    results = run_benchmark(args.checkpoint, args.onnx, args.data, n_timing_runs=args.timing_runs)

    speedup = results["pytorch"]["latency_ms"] / results["onnx"]["latency_ms"]
    mae_drift = results["onnx"]["mean_one_minus_mae"] - results["pytorch"]["mean_one_minus_mae"]

    metric_keys = (
        "latency_ms",
        "mean_one_minus_mae",
        "mean_pearson_r",
        "mean_ccc",
        "mean_r2",
        "mean_std_ratio",
    )
    print(f"{'metric':<22} {'pytorch':>12} {'onnx':>12}")
    for key in metric_keys:
        print(f"{key:<22} {results['pytorch'][key]:>12.4f} {results['onnx'][key]:>12.4f}")
    print(f"\nspeedup: {speedup:.2f}x")
    print(f"1-MAE drift (onnx - pytorch): {mae_drift:+.4f}")


if __name__ == "__main__":
    main()
