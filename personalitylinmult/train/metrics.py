import numpy as np
from pathlib import Path
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score


def racc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return 1 - mean_absolute_error(y_true, y_pred)


def calculate_app_metrics(preds_np: np.ndarray, targets_np: np.ndarray, config: dict = None, output_path: str | Path = None) -> dict:
    traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"] # Neuroticism here is already Emotional Stability, but I keep the name
    if config is not None and 'target_id' in config:
        traits = [traits[config['target_id']]]

    assert preds_np.shape[1] == len(traits), f"Expected {len(traits)} traits, got {preds_np.shape[1]}."

    # Calculate racc=1-MAE for each trait
    trait_racc = {trait[0]: racc(targets_np[:, i], preds_np[:, i]) for i, trait in enumerate(traits)}

    # Calculate the average racc=1-MAE and r2 across all traits
    mean_racc = np.array(list(trait_racc.values())).mean()
    r2 = r2_score(targets_np, preds_np)

    metrics = trait_racc | {
        'racc': mean_racc,
        'r2': r2
    }

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame({
            metric_name: [metric_value]
            for metric_name, metric_value 
            in metrics.items()
        })
        df.to_csv(str(output_path), index=False)
        print(f'Metrics saved to {output_path}')

    return metrics


def calculate_sentiment_metrics(preds_np: np.ndarray, targets_np: np.ndarray, output_path: str | Path = None) -> dict:
    # 7-class metrics
    # Round predictions to the nearest integer for classification into 7 classes (-3 to 3)
    preds_7_class = np.round(preds_np).astype(int)
    targets_7_class = np.round(targets_np).astype(int)

    acc_7 = accuracy_score(targets_7_class, preds_7_class)
    f1_7 = f1_score(targets_7_class, preds_7_class, average='weighted')

    # Binary metrics
    # Convert to binary: Sentiment > 0 is positive, <= 0 is negative
    preds_binary = (preds_np > 0).astype(int)
    targets_binary = (targets_np > 0).astype(int)

    acc_2 = accuracy_score(targets_binary, preds_binary)
    f1_2 = f1_score(targets_binary, preds_binary, average='weighted')

    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(targets_np, preds_np)

    # Pearson Correlation (Corr)
    corr, _ = pearsonr(preds_np, targets_np)

    # Save results to a CSV file
    metrics = {
        'acc_7': acc_7,
        'acc_2': acc_2,
        'f1_7': f1_7,
        'f1_2': f1_2,
        'mae': mae,
        'corr': corr
    }

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame({
            metric_name: [metric_value]
            for metric_name, metric_value 
            in metrics.items()
        })
        df.to_csv(str(output_path), index=False)
        print(f'Metrics saved to {output_path}')
    
    return metrics