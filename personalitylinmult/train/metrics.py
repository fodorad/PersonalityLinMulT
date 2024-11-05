import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error


def calculate_sentiment_metrics(preds_np, targets_np):
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
    return {
        'acc_7': acc_7,
        'acc_2': acc_2,
        'f1_7': f1_7,
        'f1_2': f1_2,
        'mae': mae,
        'corr': corr
    }