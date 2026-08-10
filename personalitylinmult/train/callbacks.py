"""Callbacks for FI training runs.

The old code kept a hand-rolled ``History`` class that maintained a nested dict,
wrote ``history.json``, and drove matplotlib itself. MLflow is that store now,
so what remains here is only what MLflow does not do on its own: epoch timing,
per-clip prediction dumps, and the diagnostic plots that make a run readable at
a glance.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import lightning as L
import matplotlib
import torch

if TYPE_CHECKING:
    from personalitylinmult.train.module import FiLightningModule

matplotlib.use("Agg")  # No display on a training host.
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)
"""Module-level logger."""


class TimeTrackingCallback(L.Callback):
    """Records wall-clock time per epoch and for the whole run.

    Args:
        output_dir (Path): Directory to write ``time.json`` into.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.epoch_times: list[dict[str, float]] = []
        self._run_start = 0.0
        self._epoch_start = 0.0

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Start the run timer.

        Args:
            trainer (L.Trainer): The trainer.
            pl_module (L.LightningModule): The module.
        """
        self._run_start = time.perf_counter()

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Start the epoch timer.

        Args:
            trainer (L.Trainer): The trainer.
            pl_module (L.LightningModule): The module.
        """
        self._epoch_start = time.perf_counter()

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Record the epoch's duration.

        Args:
            trainer (L.Trainer): The trainer.
            pl_module (L.LightningModule): The module.
        """
        elapsed = time.perf_counter() - self._epoch_start
        self.epoch_times.append({"epoch": trainer.current_epoch, "seconds": elapsed})
        pl_module.log("train/epoch_seconds", elapsed)

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Write the timing summary.

        Args:
            trainer (L.Trainer): The trainer.
            pl_module (L.LightningModule): The module.
        """
        total = time.perf_counter() - self._run_start
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "time.json"
        path.write_text(
            json.dumps(
                {
                    "total_seconds": total,
                    "epochs": self.epoch_times,
                    "mean_epoch_seconds": (
                        sum(e["seconds"] for e in self.epoch_times) / len(self.epoch_times)
                        if self.epoch_times
                        else 0.0
                    ),
                },
                indent=2,
            )
        )
        logger.info(f"Run took {total:.1f}s; timings written to {path}")


class PredictionWriter(L.Callback):
    """Writes per-clip test predictions next to their targets.

    This artifact is what lets significance testing and per-trait analysis run
    later without re-running the model.

    Args:
        output_dir (Path): Directory to write ``test_predictions.csv`` into.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.csv_path = self.output_dir / "test_predictions.csv"

    def on_test_epoch_end(  # ty: ignore[invalid-method-override]
        self, trainer: L.Trainer, pl_module: FiLightningModule
    ) -> None:
        """Dump predictions, targets, and clip ids to CSV.

        Args:
            trainer (L.Trainer): The trainer.
            pl_module (FiLightningModule): The module under test.
        """
        import pandas as pd

        preds, target = pl_module.test_metrics.predictions()
        if preds.shape[0] == 0:
            logger.warning("No test predictions to write.")
            return

        traits = pl_module.train_config.targets
        video_ids = pl_module.test_video_ids

        if len(video_ids) != preds.shape[0]:
            logger.warning(
                f"{len(video_ids)} clip ids for {preds.shape[0]} predictions; writing without ids."
            )
            video_ids = [""] * preds.shape[0]

        frame = pd.DataFrame({"video_id": video_ids})
        for i, trait in enumerate(traits):
            frame[f"pred_{trait}"] = preds[:, i].cpu().numpy()
            frame[f"true_{trait}"] = target[:, i].cpu().numpy()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(self.csv_path, index=False)
        logger.info(f"Wrote {len(frame)} test predictions to {self.csv_path}")


class PlotCallback(L.Callback):
    """Plots per-trait predicted-vs-true scatters at the end of testing.

    Args:
        output_dir (Path): Directory to write plots into.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)

    def on_test_epoch_end(  # ty: ignore[invalid-method-override]
        self, trainer: L.Trainer, pl_module: FiLightningModule
    ) -> None:
        """Write one scatter per trait.

        Args:
            trainer (L.Trainer): The trainer.
            pl_module (FiLightningModule): The module under test.
        """
        preds, target = pl_module.test_metrics.predictions()
        if preds.numel() == 0:
            return

        traits = pl_module.train_config.targets
        self.output_dir.mkdir(parents=True, exist_ok=True)

        n = len(traits)
        fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2), squeeze=False)
        for i, trait in enumerate(traits):
            axis = axes[0][i]
            y_true = target[:, i].cpu().numpy()
            y_pred = preds[:, i].cpu().numpy()
            axis.scatter(y_true, y_pred, s=6, alpha=0.35)
            axis.plot([0, 1], [0, 1], "k--", linewidth=1)
            score = 1.0 - float(torch.abs(preds[:, i] - target[:, i]).mean())
            axis.set_title(f"{trait}\n1-MAE={score:.3f}", fontsize=9)
            axis.set_xlabel("true")
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            if i == 0:
                axis.set_ylabel("predicted")

        fig.tight_layout()
        path = self.output_dir / "test_scatter.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        logger.info(f"Wrote per-trait scatter plots to {path}")
