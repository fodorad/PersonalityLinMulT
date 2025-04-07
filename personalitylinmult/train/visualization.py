from pathlib import Path
import matplotlib.pyplot as plt
import torch
import numpy as np


def plot_fi_histograms(config, gt, preds, output_dir):
    """Plots histograms for each trait comparing ground truth (GT) and predictions (preds)."""

    traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    if "target_id" in config:
        traits = [traits[config["target_id"]]]
    
    # Clamp values to [0, 1]
    gt = np.clip(gt, 0, 1)
    preds = np.clip(preds, 0, 1)

    output_dir = Path(output_dir) / 'visualization'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot histograms for each trait
    for i, trait in enumerate(traits):
        plt.figure(figsize=(10, 6))

        plt.hist(gt[:, i], bins=15, alpha=0.5, label=f'{trait} GT', color='green')
        plt.hist(preds[:, i], bins=15, alpha=0.5, label=f'{trait} Predictions', color='blue')
        
        # Add labels and title
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.title(f'{trait} - Ground Truth vs Predictions')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)

        # Save the plot
        plot_path = output_dir / f'{trait.lower()}_hist.png'
        plt.savefig(str(plot_path))
        plt.close()

        print(f"Saved {trait} histogram to {str(output_dir)}")