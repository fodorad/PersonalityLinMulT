import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
import os

# ── Configuration ────────────────────────────────────────────────────────────
DB_NAME = "FI"
RESULTS_DIR = f"results/{DB_NAME}/"      # root folder containing model subdirectories
N_BOOTSTRAP = 10_000                     # number of bootstrap iterations
RANDOM_SEED = 42
TRAITS = ["O", "C", "E", "A", "N"]

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

# Bonferroni correction for multiple comparisons
N_COMPARISONS = len(PAIRS)
ALPHA = 0.05
ALPHA_CORRECTED = ALPHA / N_COMPARISONS
# ─────────────────────────────────────────────────────────────────────────────


def load_predictions(model_dir):
    """Load all trait predictions from a model directory.
    Returns a dict with trait names as keys and (preds, gt) arrays as values.
    """
    base = os.path.join(RESULTS_DIR, model_dir)
    data = {}
    for i, trait in enumerate(TRAITS):
        df = pd.read_csv(os.path.join(base, f"test_predictions_{i}.csv"))
        data[trait] = (df["Prediction"].values, df["GroundTruth"].values)
    return data


def compute_racc(data, indices):
    """Compute mean Pearson correlation (Racc) over traits for given sample indices."""
    correlations = []
    for trait in TRAITS:
        preds, gt = data[trait]
        r, _ = pearsonr(preds[indices], gt[indices])
        correlations.append(r)
    return np.mean(correlations)


def compute_r2(data, indices):
    """Compute R^2 (coefficient of determination) pooled across all traits."""
    all_preds, all_gt = [], []
    for trait in TRAITS:
        preds, gt = data[trait]
        all_preds.append(preds[indices])
        all_gt.append(gt[indices])
    all_preds = np.concatenate(all_preds)
    all_gt = np.concatenate(all_gt)
    ss_res = np.sum((all_gt - all_preds) ** 2)
    ss_tot = np.sum((all_gt - np.mean(all_gt)) ** 2)
    return 1 - ss_res / ss_tot


def bootstrap_metrics(data, n_bootstrap, seed):
    """Run bootstrap resampling and return arrays of Racc and R2 values."""
    rng = np.random.default_rng(seed)
    n_samples = len(data[TRAITS[0]][0])
    racc_vals, r2_vals = [], []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_samples, size=n_samples)  # resample with replacement
        racc_vals.append(compute_racc(data, idx))
        r2_vals.append(compute_r2(data, idx))
    return np.array(racc_vals), np.array(r2_vals)


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
    "R_acc": True,  # Higher is better
    "R^2": True,    # Higher is better
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
    print(f"{'='*80}")
    print(f"Number of comparisons: {N_COMPARISONS}")
    print(f"Significance level (uncorrected): α = {ALPHA:.4f}")
    print(f"Significance level (Bonferroni-corrected): α = {ALPHA_CORRECTED:.4f}")
    print(f"Bootstrap iterations: {N_BOOTSTRAP:,}")
    print(f"{'='*80}\n")

    results = []

    for model_a, model_b in PAIRS:
        print(f"\n{'-'*80}")
        print(f"Comparison: {model_a} vs {model_b} (baseline)")
        print(f"{'-'*80}")

        data_a = load_predictions(model_a)
        data_b = load_predictions(model_b)

        # Also print the actual full-test-set metrics for reference
        metrics_a = pd.read_csv(os.path.join(RESULTS_DIR, model_a, "metrics_test.csv"))
        metrics_b = pd.read_csv(os.path.join(RESULTS_DIR, model_b, "metrics_test.csv"))
        print(f"\n  Full test-set Racc:  {model_a} = {metrics_a['racc'].values[0]:.4f}"
              f"   {model_b} = {metrics_b['racc'].values[0]:.4f}")
        print(f"  Full test-set R2:    {model_a} = {metrics_a['r2'].values[0]:.4f}"
              f"   {model_b} = {metrics_b['r2'].values[0]:.4f}")

        print(f"\n  Running {N_BOOTSTRAP} bootstrap iterations...")
        racc_a, r2_a = bootstrap_metrics(data_a, N_BOOTSTRAP, rng_seed)
        racc_b, r2_b = bootstrap_metrics(data_b, N_BOOTSTRAP, rng_seed)

        print("\n  Bootstrap paired t-test results:")
        racc_result = run_paired_test(racc_a, racc_b, "R_acc", model_a, model_b)
        r2_result = run_paired_test(r2_a, r2_b, "R^2", model_a, model_b)

        results.append(racc_result)
        results.append(r2_result)

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