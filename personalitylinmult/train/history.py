from pathlib import Path
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


class History:

    def __init__(self, log_dir: str | Path):
        self.history = {}
        self.log_dir = Path(log_dir)
        self.plot_dir = self.log_dir / 'visualization'


    def save(self):
        output_path = Path(self.log_dir) / "history.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.history, f, indent=4, default=self._convert_to_serializable)
        print(f"\nHistory saved to {output_path}")


    def save_test(self):
        output_path = Path(self.log_dir) / "history_test.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        history_test = {'test': self.history['test']}
        with open(output_path, "w") as f:
            json.dump(history_test, f, indent=4, default=self._convert_to_serializable)
        print(f"\nHistory saved to {output_path}")


    def _convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist() # Convert numpy arrays to lists
        if isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return obj.item() # Convert numpy scalars to Python scalars
        return obj


    def load(self, output_path: str):
        with open(output_path, "r") as f:
            self.history = json.load(f)
        print(f"\nHistory loaded from {output_path}")
        return self


    def update(self, phase, task, metric, value, epoch):
        if phase not in self.history:
            self.history[phase] = {}
        if task not in self.history[phase]:
            self.history[phase][task] = {}
        if metric not in self.history[phase][task]:
            self.history[phase][task][metric] = []
        self.history[phase][task][metric].append((epoch, value))


    def get_metric(self, phase, task, metric):
        return self.history.get(phase, {}).get(task, {}).get(metric, [])


    def get_best_epoch(self, phase: str, task: str, metric: str, mode: str = 'max'):
        """
        Get the epoch where the given metric is the best (lowest or highest).
        
        :param phase: Phase of training (e.g., 'train', 'valid').
        :param task: Task name.
        :param metric: Metric name.
        :param mode: If 'max', find the epoch where the metric is highest. If 'min', find lowest.
        :return: Best epoch and its corresponding value.
        """
        if mode not in {'max', 'min'}: raise ValueError('Argument mode should be in \{"max", "min"\}')

        data = self.get_metric(phase, task, metric)
        if not data:
            return None, None
        
        epochs, values = zip(*data)
        
        if mode == 'max':
            best_idx = int(torch.argmax(torch.tensor(values)))
        else:
            best_idx = int(torch.argmin(torch.tensor(values)))
        
        return epochs[best_idx], values[best_idx]


    def plot(self, task: str,
                   metric: str,
                   best_epoch_metric: str = None):
        """
        Plot a given metric for both training and validation phases, and optionally mark
        the best epoch of another metric.

        :param task: Task name.
        :param metric: Metric to plot.
        :param best_epoch_metric: Metric used to determine the best epoch.
                                  If provided, its best epoch will be marked on the plot.
                                  If `best_epoch_metric` is None or not found in history,
                                  only the current metric will be plotted.
        """

        # Determine whether to maximize or minimize based on the metric name
        mode = 'max' if any([elem.lower() in metric.lower() for elem in ['f1', 'corr', 'acc']]) else 'min'

        # Plot training and validation metrics
        for phase in ["train", "valid"]:
            data = self.get_metric(phase=phase, task=task, metric=metric)
            if data:
                epochs, values = zip(*data)
                plt.plot(epochs, values, "*-", label=f"{phase} {metric}")

                # Mark the best epoch for this metric in each phase
                best_metric_epoch, best_metric_value = self.get_best_epoch(
                    phase=phase, task=task, metric=metric, mode=mode
                )
                if best_metric_epoch is not None and phase == 'valid':
                    plt.plot(best_metric_epoch, best_metric_value, 'go', markersize=8,
                             label=f"Best {metric} ({phase}, Epoch {best_metric_epoch})")

        # Mark the best epoch for another selected metric (if provided)
        if best_epoch_metric:
            mode = 'max' if any([elem.lower() in best_epoch_metric.lower() for elem in ['f1', 'corr', 'acc']]) else 'min'
            for phase in ["train", "valid"]:
                data = self.get_metric(phase=phase, task=task, metric=metric)
                if data:
                    epochs, values = zip(*data)
                    best_other_epoch, _ = self.get_best_epoch(
                        phase=phase, task=task, metric=best_epoch_metric, mode=mode
                    )
                    if best_other_epoch is not None and phase == 'valid':
                        plt.plot(best_other_epoch, values[epochs.index(best_other_epoch)], 'b*', markersize=10,
                                 label=f"Best {best_epoch_metric} ({phase}, Epoch {best_other_epoch})")

        plt.title(f"{metric} over Epochs for {task}")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend(loc="best")
        plt.grid(True)
        plt.tight_layout()

        output_file = self.plot_dir / f'plot_{metric}.png'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file)
        plt.close()


    def plot_ncm(self, phase, task, metric, output_dir, n_classes: int):
        """Plot the Normalized Confusion Matrix (NCM)"""
        # Extract confusion matrices for the given phase and metric
        confusion_matrices = [epoch_ncm_tuple[1] for epoch_ncm_tuple in self.history[phase][task]["NormalizedConfusionMatrix"]]

        if phase == "test":
            # For test phase, expect only one entry
            if len(confusion_matrices) != 1:
                raise ValueError("Test history should contain exactly one entry for confusion matrix.")
            best_confusion_matrix = confusion_matrices[0]
            if best_confusion_matrix is None:
                raise ValueError("The test confusion matrix histroy is missing or invalid.")
        else:
            # For other phases like "valid", find the best confusion matrix index based on the specified metric
            if not confusion_matrices or all(cm is None for cm in confusion_matrices):
                raise ValueError("No valid confusion matrix history found for the given phase and metric.")

            metrics = [epoch_metric_tuple[1] for epoch_metric_tuple in self.history[phase][task][metric]]
            best_valid_index = np.nanargmax(metrics)  # Find the index of the best value

            # Select the best confusion matrix
            best_confusion_matrix = confusion_matrices[best_valid_index]
            if best_confusion_matrix is None:
                raise ValueError("The best confusion matrix history is missing or invalid.")

        # Reshape the confusion matrix
        best_confusion_matrix = np.array(best_confusion_matrix).reshape(n_classes, n_classes)

        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            best_confusion_matrix,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            cbar=True,
            linewidths=0.5,
            xticklabels=[f"Class {i}" for i in range(n_classes)],
            yticklabels=[f"Class {i}" for i in range(n_classes)]
        )

        plt.xlabel("Predicted Labels", fontsize=14)
        plt.ylabel("True Labels", fontsize=14)
        plt.title(f"Normalized Confusion Matrix ({phase.capitalize()}, {task}, {metric})", fontsize=16)

        output_path = output_dir / f'{phase}_{task}_{metric}_ncm.png'
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()