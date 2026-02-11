from pathlib import Path
import numpy as np
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt


def two_sided_mann_whitney_u_test(baseline_file: str, proposed_file: str, output_dir: str = None):
    """Two-sided Mann-Whitney U Test.

    The Mann-Whitney U test, also known as the Wilcoxon rank-sum test, 
    is a non-parametric statistical test used to compare two independent groups. 
    It evaluates whether the distributions of the two groups differ in terms of central tendency 
    (e.g., medians) or overall distribution shape

    Hypotheses:
        Null Hypothesis (H_0): The two groups come from the same population (no difference in distributions).
        Alternative Hypothesis (H_a): The two groups come from populations with different distributions.
    """
    baseline_data = pd.read_csv(baseline_file)
    proposed_data = pd.read_csv(proposed_file)

    baseline_predictions = baseline_data['Prediction'].values
    proposed_predictions = proposed_data['Prediction'].values

    # Perform a paired two-sided t-test
    mwu_result = pg.mwu(x=baseline_predictions, 
                        y=proposed_predictions, 
                        alternative='two-sided')

    p_value = mwu_result['p-val'].values[0]
    if p_value < 0.05:
        print("Reject H_0: There is a significant difference between the baseline and proposed model predictions.")
    else:
        print("Fail to reject H_0: There is no significant difference between the baseline and proposed model predictions.")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = str(output_dir / 'ts_mwu_test.csv')
        mwu_result.to_csv(output_file, index=False)


def _convert_to_method_name(filepath: str, plot: bool = False):
    name_parts = Path(filepath).parent.name.split('_')
    if 'FI' in filepath:
        trait_ind = int(Path(filepath).stem.split('_')[-1])
        if plot:
            traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', r"$\mathregular{\overline{N}euroticism}$"]
        else:
            traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', "Neuroticism"]
        name = f'{name_parts[-1]} ({name_parts[0]}): {traits[trait_ind]}'
    else:
        name = f'{name_parts[-1]} ({name_parts[0]})'
    return name


def draw_qq_plots(files: list[str], output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    datasets = {}
    for file in files:
        data = pd.read_csv(file)
        name = _convert_to_method_name(file, plot=True)
        predictions = data['Prediction'].values
        datasets[name] = predictions

    if len(datasets) > 6:
        figsize = (8, 18)
    else:
        figsize = (8, 12)

    plt.figure(figsize=figsize)
    for i, (name, data) in enumerate(datasets.items(), start=1):
        plt.subplot(len(datasets)//2, 2, i)
        ax = pg.qqplot(data, dist='norm')
        ax.set_xlim(-3.9, 3.9)
        ax.set_ylim(-3.9, 3.9)
        plt.title(f'{name}')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def two_sided_paired_t_test(baseline_file: str, proposed_file: str, alpha: float = 0.05, output_dir: str = None, verbose: bool = False):
    """ Two-sided paired t-test.
    To determine whether there is a significant difference between two groups, without specifying the direction of the difference.
    Paired, because both baseline and proposed model predictions are generated for the same input sample.

    Hypotheses:
        Null hypothesis (H_0): The means of the two groups are equal (μ_1 = μ_2).
        Alternative hypothesis (H_a): The means of the two groups are not equal (μ_1 ≠ μ_2).
    """
    baseline_name = _convert_to_method_name(baseline_file)
    proposed_name = _convert_to_method_name(proposed_file)

    baseline_data = pd.read_csv(baseline_file)
    proposed_data = pd.read_csv(proposed_file)

    baseline_predictions = baseline_data['Prediction'].values
    proposed_predictions = proposed_data['Prediction'].values

    # Perform a paired two-sided t-test
    result = pg.ttest(x=baseline_predictions, 
                      y=proposed_predictions, 
                      paired=True, 
                      alternative='two-sided')

    p_value = result['p-val'].values[0]
    if verbose:
        if p_value < alpha:
            print("Reject H_0: There is a significant difference between the baseline and proposed model predictions.")
        else:
            print("Fail to reject H_0: There is no significant difference between the baseline and proposed model predictions.")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = str(output_dir / 'two_sided_paired_t_test.csv')
        result.to_csv(output_file, index=False)

    return result, p_value


def mosi_qq_plots():
    print('[MOSI] Q-Q Plots')

    proposed_file = 'results/MOSI/OOWFR_LinMulT/test_predictions.csv'
    baseline_files = [
        'results/MOSI/OOWFR_MulT/test_predictions.csv',
        'results/MOSI/OOB_LinMulT/test_predictions.csv',
        'results/MOSI/OOB_MulT/test_predictions.csv', 
        'results/MOSI/WFR_LinMulT/test_predictions.csv',
        'results/MOSI/WFR_MulT/test_predictions.csv',
    ]

    draw_qq_plots([proposed_file] + baseline_files, 'results/significance/MOSI/QQ_plots.png')


def mosei_qq_plots():
    print('[MOSEI] Q-Q Plots')

    proposed_file = 'results/MOSEI/OOWFR_LinMulT/test_predictions.csv'
    baseline_files = [
        'results/MOSEI/OOWFR_MulT/test_predictions.csv',
        'results/MOSEI/OOB_LinMulT/test_predictions.csv',
        'results/MOSEI/OOB_MulT/test_predictions.csv', 
        'results/MOSEI/WFR_LinMulT/test_predictions.csv',
        'results/MOSEI/WFR_MulT/test_predictions.csv',
    ]

    draw_qq_plots([proposed_file] + baseline_files, 'results/significance/MOSEI/QQ_plots.png')


def fi_qq_plots():
    print('[FI] Q-Q Plots')

    files = []
    for trait in range(5):
        files.append(f'results/FI/OOWFR_LinMulT/test_predictions_{trait}.csv')
        files.append(f'results/FI/OOWFR_MulT/test_predictions_{trait}.csv')

    draw_qq_plots(files, 'results/significance/FI/QQ_plots.png')


def mosi_significance(output_dir: str = 'results/significance/MOSI'):
    print('[MOSI] Two-Sided Paired T-Test')

    proposed_file = 'results/MOSI/OOWFR_LinMulT/test_predictions.csv'
    baseline_files = [
        'results/MOSI/OOWFR_MulT/test_predictions.csv',
        'results/MOSI/OOB_LinMulT/test_predictions.csv',
        'results/MOSI/OOB_MulT/test_predictions.csv', 
        'results/MOSI/WFR_LinMulT/test_predictions.csv',
        'results/MOSI/WFR_MulT/test_predictions.csv',
    ]

    proposed_name = _convert_to_method_name(proposed_file)
    baseline_names = [_convert_to_method_name(baseline_file) for baseline_file in baseline_files]

    results = []
    p_values = []
    for baseline_file in baseline_files:
        result, p_value = two_sided_paired_t_test(
            baseline_file,
            proposed_file,
            output_dir=f'results/significance/MOSI/{Path(proposed_file).parent.name}_vs_{Path(baseline_file).parent.name}'
        )
        results.append(result)
        p_values.append(p_value)

    # Apply Bonferroni correction
    reject_null_hypotheses, corrected_p_values = pg.multicomp(p_values, method='bonf')

    # print("Reject null hypotheses:", reject_null_hypotheses) # [bool,...]
    # print("Corrected p-values:", corrected_p_values) # [p_val,...]

    # update tables with the decision
    dfs = []
    for result, corrected_p_value, reject_H_0, baseline_name in zip(results, corrected_p_values, reject_null_hypotheses, baseline_names):
        result['model_1'] = proposed_name
        result['model_2'] = baseline_name
        result['bonf_p-val'] = corrected_p_value
        result['decision'] = 'H_a' if reject_H_0 else 'H_0'
        dfs.append(result)
    df = pd.concat(dfs, ignore_index=True)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = str(output_dir / 'two_sided_paired_t_test.csv')
        df.to_csv(output_file, index=False)

    return df


def mosei_significance(output_dir: str = 'results/significance/MOSEI'):
    print('[MOSEI] Two-Sided Paired T-Test')

    proposed_file = 'results/MOSEI/OOWFR_LinMulT/test_predictions.csv'
    baseline_files = [
        'results/MOSEI/OOWFR_MulT/test_predictions.csv',
        'results/MOSEI/OOB_LinMulT/test_predictions.csv',
        'results/MOSEI/OOB_MulT/test_predictions.csv', 
        'results/MOSEI/WFR_LinMulT/test_predictions.csv',
        'results/MOSEI/WFR_MulT/test_predictions.csv',
    ]

    proposed_name = _convert_to_method_name(proposed_file)
    baseline_names = [_convert_to_method_name(baseline_file) for baseline_file in baseline_files]

    results = []
    p_values = []
    for baseline_file in baseline_files:
        result, p_value = two_sided_paired_t_test(
            baseline_file,
            proposed_file,
            output_dir=f'results/significance/MOSEI/{Path(proposed_file).parent.name}_vs_{Path(baseline_file).parent.name}'
        )
        results.append(result)
        p_values.append(p_value)

    # Apply Bonferroni correction
    reject_null_hypotheses, corrected_p_values = pg.multicomp(p_values, method='bonf')

    # print("Reject null hypotheses:", reject_null_hypotheses) # [bool,...]
    # print("Corrected p-values:", corrected_p_values) # [p_val,...]

    # update tables with the decision
    dfs = []
    for result, corrected_p_value, reject_H_0, baseline_name in zip(results, corrected_p_values, reject_null_hypotheses, baseline_names):
        result['model_1'] = proposed_name
        result['model_2'] = baseline_name
        result['bonf_p-val'] = corrected_p_value
        result['decision'] = 'H_a' if reject_H_0 else 'H_0'
        dfs.append(result)
    df = pd.concat(dfs, ignore_index=True)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = str(output_dir / 'two_sided_paired_t_test.csv')
        df.to_csv(output_file, index=False)

    return df



def fi_significance(output_dir: str = 'results/significance/FI'):
    print('[FI] Two-Sided Paired T-Test')

    result_dfs = []
    for trait in range(5):
        proposed_file = f'results/FI/OOWFR_LinMulT/test_predictions_{trait}.csv'

        baseline_files = [
            f'results/FI/OOB_MulT/test_predictions_{trait}.csv',
            f'results/FI/OOB_LinMulT/test_predictions_{trait}.csv',
            f'results/FI/WFR_MulT/test_predictions_{trait}.csv',
            f'results/FI/WFR_LinMulT/test_predictions_{trait}.csv',
            f'results/FI/OOWFR_MulT/test_predictions_{trait}.csv',
        ]

        proposed_name = _convert_to_method_name(proposed_file)
        baseline_names = [_convert_to_method_name(baseline_file) for baseline_file in baseline_files]

        results = []
        p_values = []
        for baseline_file in baseline_files:
            result, p_value = two_sided_paired_t_test(
                baseline_file,
                proposed_file,
                output_dir=Path(output_dir) / str(trait) / f'{Path(proposed_file).parent.name}_vs_{Path(baseline_file).parent.name}'
            )
            results.append(result)
            p_values.append(p_value)

        # Apply Bonferroni correction
        reject_null_hypotheses, corrected_p_values = pg.multicomp(p_values, method='bonf')

        # print("Reject null hypotheses:", reject_null_hypotheses) # [bool,...]
        # print("Corrected p-values:", corrected_p_values) # [p_val,...]

        # update tables with the decision
        dfs = []
        for result, corrected_p_value, reject_H_0, baseline_name in zip(results, corrected_p_values, reject_null_hypotheses, baseline_names):
            result['model_1'] = proposed_name
            result['model_2'] = baseline_name
            result['bonf_p-val'] = corrected_p_value
            result['decision'] = 'H_a' if reject_H_0 else 'H_0'
            dfs.append(result)
        df = pd.concat(dfs, ignore_index=True)

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = str(output_dir / str(trait) / 'two_sided_paired_t_test.csv')
            df.to_csv(output_file, index=False)

        result_dfs.append(df)

    result_df = pd.concat(result_dfs, ignore_index=True)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = str(output_dir / 'two_sided_paired_t_test.csv')
        result_df.to_csv(output_file, index=False)

    return result_df


if __name__ == "__main__":
    mosi_qq_plots()
    mosei_qq_plots()
    fi_qq_plots()
    mosi_significance()
    mosei_significance()
    fi_significance()