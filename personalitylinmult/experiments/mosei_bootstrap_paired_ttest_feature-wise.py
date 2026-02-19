import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
import os

# ── Configuration ────────────────────────────────────────────────────────────
DB_NAME = "MOSEI"
RESULTS_DIR = f"results/{DB_NAME}/"      # root folder containing model subdirectories
N_BOOTSTRAP = 10_000                     # number of bootstrap iterations
RANDOM_SEED = 42
# Compare OOWFR_LinMulT (proposed method) against all other configurations
BASELINE = "OOWFR_LinMulT"
COMPARISONS = [
    "OOB_MulT",
    "OOB_LinMulT",
    "WFR_MulT",
    "WFR_LinMulT",
    "OOWFR_MulT",
]
PAIRS = [(comp, BASELINE) for comp in COMPARISONS]

# Bonferroni correction for multiple comparisons (5 comparisons)
N_COMPARISONS = len(PAIRS)
ALPHA = 0.05
ALPHA_CORRECTED = ALPHA / N_COMPARISONS
# ─────────────────────────────────────────────────────────────────────────────


def load_predictions(model_dir):
    """Load sentiment predictions from a model directory.
    Returns tuple of (predictions, ground_truth) arrays.
    """
    base = os.path.join(RESULTS_DIR, model_dir)
    df = pd.read_csv(os.path.join(base, "test_predictions.csv"))
    return df["Prediction"].values, df["GroundTruth"].values


def compute_mae(preds, gt, indices):
    """Compute mean absolute error for given sample indices."""
    return np.mean(np.abs(gt[indices] - preds[indices]))


def compute_corr(preds, gt, indices):
    """Compute Pearson correlation for given sample indices."""
    r, _ = pearsonr(preds[indices], gt[indices])
    return r


def compute_acc7(preds, gt, indices):
    """Compute 7-class accuracy (round predictions to nearest integer)."""
    pred_classes = np.round(preds[indices]).astype(int)
    gt_classes = np.round(gt[indices]).astype(int)
    return np.mean(pred_classes == gt_classes)


def compute_f1_7(preds, gt, indices):
    """Compute 7-class F1 score (macro average)."""
    pred_classes = np.round(preds[indices]).astype(int)
    gt_classes = np.round(gt[indices]).astype(int)
    return f1_score(gt_classes, pred_classes, average='macro', zero_division=0)


def compute_acc2(preds, gt, indices):
    """Compute 2-class accuracy (positive/zero vs negative)."""
    pred_classes = (preds[indices] >= 0).astype(int)
    gt_classes = (gt[indices] >= 0).astype(int)
    return np.mean(pred_classes == gt_classes)


def compute_f1_2(preds, gt, indices):
    """Compute 2-class F1 score."""
    pred_classes = (preds[indices] >= 0).astype(int)
    gt_classes = (gt[indices] >= 0).astype(int)
    return f1_score(gt_classes, pred_classes, zero_division=0)


def bootstrap_metrics(preds, gt, n_bootstrap, seed):
    """Run bootstrap resampling and return arrays of all 6 metric values."""
    rng = np.random.default_rng(seed)
    n_samples = len(preds)
    mae_vals, corr_vals = [], []
    acc7_vals, f1_7_vals = [], []
    acc2_vals, f1_2_vals = [], []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_samples, size=n_samples)  # resample with replacement
        mae_vals.append(compute_mae(preds, gt, idx))
        corr_vals.append(compute_corr(preds, gt, idx))
        acc7_vals.append(compute_acc7(preds, gt, idx))
        f1_7_vals.append(compute_f1_7(preds, gt, idx))
        acc2_vals.append(compute_acc2(preds, gt, idx))
        f1_2_vals.append(compute_f1_2(preds, gt, idx))

    return (np.array(mae_vals), np.array(corr_vals),
            np.array(acc7_vals), np.array(f1_7_vals),
            np.array(acc2_vals), np.array(f1_2_vals))


def format_pvalue(pval):
    """Format p-value, handling underflow cases."""
    if pval == 0:  # Underflow to 0
        return "< 1.0e-300"
    formatted = f"{pval:.3e}"
    if formatted == "0.000e+00":  # Still rounds to 0
        return "< 1.0e-300"
    return formatted


def format_model_name(model_name):
    """Format model name for display: OOWFR_LinMulT -> LinMulT (OOWFR)."""
    if "_" not in model_name:
        return model_name
    parts = model_name.rsplit("_", 1)  # Split from right to separate feature set and model
    feature_set, model = parts[0], parts[1]
    return f"{model} ({feature_set})"


# Metric direction: True = higher is better, False = lower is better
METRIC_DIRECTIONS = {
    "MAE": False,        # Lower is better
    "Correlation": True, # Higher is better
    "Acc_7": True,       # Higher is better
    "F1_7": True,        # Higher is better
    "Acc_2": True,       # Higher is better
    "F1_2": True,        # Higher is better
}


def determine_direction(mean_diff, metric_name, is_significant):
    """Determine which model is better based on metric direction and sign.
    Returns: 'Comparison', 'Proposed Method', or '-' (no significant difference)
    """
    if not is_significant:
        return "-"

    higher_is_better = METRIC_DIRECTIONS.get(metric_name, True)

    if higher_is_better:
        return "Comparison" if mean_diff > 0 else "Proposed Method"
    else:
        return "Proposed Method" if mean_diff > 0 else "Comparison"


def run_paired_test(vals_a, vals_b, metric_name, label_a, label_b):
    """Run paired t-test and return results as a dictionary."""
    diff = vals_a - vals_b
    t_stat, p_val = stats.ttest_1samp(diff, popmean=0)
    mean_diff = np.mean(diff)
    ci_low, ci_high = np.percentile(diff, [2.5, 97.5])
    p_val_bonf = p_val * N_COMPARISONS  # Bonferroni correction
    p_val_bonf = min(p_val_bonf, 1.0)   # Cap at 1.0
    is_significant = p_val < ALPHA_CORRECTED
    direction = determine_direction(mean_diff, metric_name, is_significant)

    result = {
        "Proposed Method": BASELINE,
        "Comparison": label_a,
        "Metric": metric_name,
        "Mean Difference": mean_diff,
        "CI Lower": ci_low,
        "CI Upper": ci_high,
        "p-value": p_val,
        "Bonferroni p-value": p_val_bonf,
        "Direction": direction,
        "Decision": "H_a" if is_significant else "H_0"
    }

    return result


def main():
    rng_seed = RANDOM_SEED
    print(f"\n{'='*80}")
    print(f"BOOTSTRAP PAIRED T-TEST: {BASELINE} vs All Other Configurations")
    print(f"Dataset: MOSEI - Sentiment Prediction")
    print(f"{'='*80}")
    print(f"Number of comparisons: {N_COMPARISONS}")
    print(f"Significance level (uncorrected): α = {ALPHA:.4f}")
    print(f"Significance level (Bonferroni-corrected): α = {ALPHA_CORRECTED:.4f}")
    print(f"Bootstrap iterations: {N_BOOTSTRAP:,}")
    print(f"{'='*80}\n")

    results = []

    for model_a, model_b in PAIRS:
        print(f"\n{'-'*80}")
        print(f"Comparison: {model_a} vs {model_b}")
        print(f"{'-'*80}")

        preds_a, gt_a = load_predictions(model_a)
        preds_b, gt_b = load_predictions(model_b)

        # Also print the actual full-test-set metrics for reference
        metrics_a = pd.read_csv(os.path.join(RESULTS_DIR, model_a, "metrics_test.csv"))
        metrics_b = pd.read_csv(os.path.join(RESULTS_DIR, model_b, "metrics_test.csv"))
        print(f"\n  Full test-set metrics:")
        for metric in ['mae', 'corr', 'acc_7', 'f1_7', 'acc_2', 'f1_2']:
            val_a = metrics_a[metric].values[0]
            val_b = metrics_b[metric].values[0]
            print(f"    {metric:12s}: {model_a} = {val_a:.4f}  |  {model_b} = {val_b:.4f}")

        print(f"\n  Running {N_BOOTSTRAP} bootstrap iterations...")
        (mae_a, corr_a, acc7_a, f1_7_a, acc2_a, f1_2_a) = bootstrap_metrics(preds_a, gt_a, N_BOOTSTRAP, rng_seed)
        (mae_b, corr_b, acc7_b, f1_7_b, acc2_b, f1_2_b) = bootstrap_metrics(preds_b, gt_b, N_BOOTSTRAP, rng_seed)

        print("\n  Bootstrap paired t-test results:")
        # Test all 6 metrics
        metric_pairs = [
            (mae_a, mae_b, "MAE"),
            (corr_a, corr_b, "Correlation"),
            (acc7_a, acc7_b, "Acc_7"),
            (f1_7_a, f1_7_b, "F1_7"),
            (acc2_a, acc2_b, "Acc_2"),
            (f1_2_a, f1_2_b, "F1_2"),
        ]

        for vals_a, vals_b, metric_name in metric_pairs:
            result = run_paired_test(vals_a, vals_b, metric_name, model_a, model_b)
            results.append(result)

    # Create summary table
    df_results = pd.DataFrame(results)

    # Create display table with selected columns
    display_df = df_results[[
        "Proposed Method", "Comparison", "Metric",
        "Mean Difference", "CI Lower", "CI Upper",
        "p-value", "Bonferroni p-value",
        "Direction", "Decision"
    ]].copy()

    # Format model names for display
    display_df["Proposed Method"] = display_df["Proposed Method"].apply(format_model_name)
    display_df["Comparison"] = display_df["Comparison"].apply(format_model_name)

    # Format numeric columns
    display_df["Mean Difference"] = display_df["Mean Difference"].apply(lambda x: f"{x:+.4f}")
    # Format p-values with underflow handling
    display_df["p-value"] = display_df["p-value"].apply(format_pvalue)
    display_df["Bonferroni p-value"] = display_df["Bonferroni p-value"].apply(format_pvalue)

    # Combine CI into single column
    display_df["CI 95%"] = display_df.apply(
        lambda row: f"[{row['CI Lower']:+.2e}, {row['CI Upper']:+.2e}]",
        axis=1
    )

    # Select final columns
    display_df = display_df[[
        "Proposed Method", "Comparison", "Metric",
        "Mean Difference", "CI 95%",
        "p-value", "Bonferroni p-value",
        "Direction", "Decision"
    ]]

    print(f"\n\n{'='*120}")
    print("SUMMARY TABLE: Bootstrap Paired T-Test Results")
    print(f"{'='*120}")
    print(display_df.to_string(index=False))

    # Save to CSV
    output_csv = os.path.join(RESULTS_DIR, f"{DB_NAME}_bootstrap_paired_ttest_summary_feature-wise.csv")
    df_results.to_csv(output_csv, index=False)
    print(f"\n✓ Results saved to: {output_csv}")


if __name__ == "__main__":
    main()
