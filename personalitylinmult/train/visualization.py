from pathlib import Path
import matplotlib.pyplot as plt
import torch
import numpy as np


def plot_fi_metrics(model, output_dir):
    """Saves the loss and 1 - MAE plots to the given directory."""

    if output_dir is None:
        print('The output_dir argument of plot_metrics is None. Skipping...')
        return

    (Path(output_dir) / 'visualization').mkdir(parents=True, exist_ok=True)
    
    # Step 1: Find the epoch with the highest validation 1-MAE
    best_epoch = int(torch.argmax(torch.tensor(model.valid_metrics))) # Get the index of the highest validation 1-MAE value
    best_val_1_mae = model.valid_metrics[best_epoch] # Get the best validation 1-MAE value
    best_val_r2 = model.valid_r2[best_epoch] # Get the best validation R^2 value
    num_epochs = len(model.valid_metrics)
    
    # Loss plot
    plt.figure(figsize=(12, 6))
    plt.plot(model.train_losses, "*-", label='Training Loss')
    plt.plot(model.valid_losses, "*-", label='Validation Loss')
    plt.plot(best_epoch, model.valid_losses[best_epoch], 'ro', markersize=10, label=f'Best Epoch {best_epoch}')
    plt.xticks(range(num_epochs))
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = str(Path(output_dir) / 'visualization' / 'plot_loss.png')
    plt.savefig(plot_path)
    plt.close()

    # 1-MAE plot
    plt.figure(figsize=(12, 6))
    plt.plot(model.train_metrics, "*-", label='Training 1-MAE')
    plt.plot(model.valid_metrics, "*-", label='Validation 1-MAE')
    plt.plot(best_epoch, best_val_1_mae, 'ro', markersize=10, label=f'Best Epoch {best_epoch}')
    plt.xticks(range(num_epochs))
    plt.title('1-MAE Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('1-MAE')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = str(Path(output_dir) / 'visualization' / 'plot_1-mae.png')
    plt.savefig(plot_path)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(model.train_metrics, "*-", label='Training 1-MAE')
    plt.plot(model.valid_metrics, "*-", label='Validation 1-MAE')
    plt.plot(best_epoch, best_val_1_mae, 'ro', markersize=10, label=f'Best Epoch {best_epoch}')
    plt.xticks(range(num_epochs))
    plt.title('1-MAE Over Epochs')
    plt.xlabel('Epochs')
    plt.ylim([0.88, 0.930])
    plt.yticks(np.arange(0.88, 0.931, 0.002))
    plt.ylabel('1-MAE')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = str(Path(output_dir) / 'visualization' / 'plot_1-mae_zoomed.png')
    plt.savefig(plot_path)
    plt.close()

    # R^2 plot
    plt.figure(figsize=(12, 6))
    plt.plot(model.valid_r2, "*-", label='Validation R^2')
    plt.plot(best_epoch, best_val_r2, 'ro', markersize=10, label=f'Best Epoch {best_epoch}')
    plt.xticks(range(num_epochs))
    plt.title('R^2 Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('R^2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = str(Path(output_dir) / 'visualization' / 'plot_r2.png')
    plt.savefig(plot_path)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(model.valid_r2, "*-", label='Validation R^2')
    plt.plot(best_epoch, best_val_r2, 'ro', markersize=10, label=f'Best Epoch {best_epoch}')
    plt.xticks(range(num_epochs))
    plt.ylim([0.0, 1.0])
    plt.yticks(np.arange(0.0, 1.01, 0.05))
    plt.title('R^2 Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('R^2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = str(Path(output_dir) / 'visualization' / 'plot_r2_zoomed.png')
    plt.savefig(plot_path)
    plt.close()

    print(f"Plots saved to {str(Path(output_dir) / 'visualization')}")


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


def plot_sentiment_metrics(model, output_dir):
    """Saves the loss and metrics plots to the given directory."""

    if output_dir is None:
        print('The output_dir argument of plot_metrics is None. Skipping...')
        return

    (Path(output_dir) / 'visualization').mkdir(parents=True, exist_ok=True)
    
    # Step 1: Find the epoch with the lowest validation mae
    best_epoch = int(torch.argmin(torch.tensor(model.valid_mae))) # Get the index of the highest validation 1-MAE value
    best_val_mae = model.valid_mae[best_epoch] 
    best_val_corr = model.valid_corr[best_epoch] 
    best_val_acc_2 = model.valid_acc_2[best_epoch] 
    best_val_acc_7 = model.valid_acc_7[best_epoch] 
    best_val_f1_2 = model.valid_f1_2[best_epoch] 
    best_val_f1_7 = model.valid_f1_7[best_epoch] 
    num_epochs = len(model.valid_mae)
    
    # Loss plot
    plt.figure(figsize=(12, 6))
    plt.plot(model.train_losses, "*-", label='Training Loss')
    plt.plot(model.valid_losses, "*-", label='Validation Loss')
    plt.plot(best_epoch, model.valid_losses[best_epoch], 'ro', markersize=10, label=f'Best Epoch {best_epoch}')
    plt.xticks(range(num_epochs))
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = str(Path(output_dir) / 'visualization' / 'plot_loss.png')
    plt.savefig(plot_path)
    plt.close()

    # MAE plot
    plt.figure(figsize=(12, 6))
    plt.plot(model.train_mae, "*-", label='Training MAE')
    plt.plot(model.valid_mae, "*-", label='Validation MAE')
    plt.plot(best_epoch, best_val_mae, 'ro', markersize=10, label=f'Best Epoch {best_epoch}')
    plt.xticks(range(num_epochs))
    plt.title('MAE Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = str(Path(output_dir) / 'visualization' / 'plot_mae.png')
    plt.savefig(plot_path)
    plt.close()

    # CORR plot
    plt.figure(figsize=(12, 6))
    plt.plot(model.valid_corr, "*-", label='Validation CORR')
    plt.plot(best_epoch, best_val_corr, 'ro', markersize=10, label=f'Best Epoch {best_epoch}')
    plt.xticks(range(num_epochs))
    plt.title('Corr Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Corr')
    plt.ylim([0,1])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = str(Path(output_dir) / 'visualization' / 'plot_corr.png')
    plt.savefig(plot_path)
    plt.close()

    # Acc2 plot
    plt.figure(figsize=(12, 6))
    plt.plot(model.valid_acc_2, "*-", label='Validation Acc_2')
    plt.plot(best_epoch, best_val_acc_2, 'ro', markersize=10, label=f'Best Epoch {best_epoch}')
    plt.xticks(range(num_epochs))
    plt.title('Acc2 Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Acc2')
    plt.ylim([0,1])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = str(Path(output_dir) / 'visualization' / 'plot_acc_2.png')
    plt.savefig(plot_path)
    plt.close()

    # Acc7 plot
    plt.figure(figsize=(12, 6))
    plt.plot(model.valid_acc_7, "*-", label='Validation Acc_7')
    plt.plot(best_epoch, best_val_acc_7, 'ro', markersize=10, label=f'Best Epoch {best_epoch}')
    plt.xticks(range(num_epochs))
    plt.title('Acc7 Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Acc7')
    plt.ylim([0,1])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = str(Path(output_dir) / 'visualization' / 'plot_acc_7.png')
    plt.savefig(plot_path)
    plt.close()

    # F1_7 plot
    plt.figure(figsize=(12, 6))
    plt.plot(model.valid_f1_7, "*-", label='Validation F1_7')
    plt.plot(best_epoch, best_val_f1_7, 'ro', markersize=10, label=f'Best Epoch {best_epoch}')
    plt.xticks(range(num_epochs))
    plt.title('F1_7 Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('F1_7')
    plt.ylim([0,1])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = str(Path(output_dir) / 'visualization' / 'plot_F1_7.png')
    plt.savefig(plot_path)
    plt.close()

    # F1_2 plot
    plt.figure(figsize=(12, 6))
    plt.plot(model.valid_f1_2, "*-", label='Validation F1_2')
    plt.plot(best_epoch, best_val_f1_2, 'ro', markersize=10, label=f'Best Epoch {best_epoch}')
    plt.xticks(range(num_epochs))
    plt.title('F1_2 Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('F1_2')
    plt.ylim([0,1])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = str(Path(output_dir) / 'visualization' / 'plot_F1_2.png')
    plt.savefig(plot_path)
    plt.close()