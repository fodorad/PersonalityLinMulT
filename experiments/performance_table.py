import json
import os
from pathlib import Path
import numpy as np
from prettytable import PrettyTable


def get_table(base_dir: str, task: str, metric_names: list[str]):
    results = {}

    for root, _, files in os.walk(base_dir):

        for file in files:

            if file == "history_test.json":

                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)

                experiment_dir = Path(os.path.relpath(root, base_dir))
                experiment_name = experiment_dir.parts[0]

                test_data = data.get('test', {}).get(task, {})

                if experiment_name not in results:
                    results[experiment_name] = {metric_name.lower(): -np.inf for metric_name in metric_names}
                
                for metric_name in metric_names:
                    metric_value = test_data.get(metric_name, [[None, -np.inf]])[0][1]

                    if metric_value == -np.inf: continue
                
                    record = results[experiment_name]
                    if '_TW' in experiment_name:
                        if metric_name.lower() in ['racc', 'r2']:
                            if record[metric_name.lower()] == -np.inf:
                                record[metric_name.lower()] = []
                            record[metric_name.lower()].append(metric_value)
                        else:
                            record[metric_name.lower()] = metric_value
                    else:
                        record[metric_name.lower()] = metric_value

                    results[experiment_name] = record

    if not results:
        print("No history_test.json files found!")
        return

    desired_order = [
        "OOB_MulT",
        "OOB_LinMulT",
        "WFR_MulT",
        "WFR_LinMulT",
        "OOWFR_MulT",
        "OOWFR_LinMulT",
        "OOWFR_MulT_TW",
        "OOWFR_LinMulT_TW",
    ]

    results = sorted(
        results.items(),
        key=lambda x: desired_order.index(x[0]) if x[0] in desired_order else float('inf')
    )

    table = PrettyTable()
    table.field_names = ['Experiment'] + metric_names
    
    for experiment_name, metrics in results:
        row_values = []
        for metric_name in metric_names:
            metric_value = metrics[metric_name.lower()]
            if isinstance(metric_value, list):
                metric_value = np.array(metric_value).mean()
            row_values.append(round(metric_value, 3))     
        table.add_row([experiment_name]+row_values)

    print(table)


if __name__ == "__main__":
    get_table("results/MOSI", 'sentiment', ['acc_7', 'acc_2', 'f1_2', 'mae', 'corr'])
    get_table("results/MOSEI", 'sentiment', ['acc_7', 'acc_2', 'f1_2', 'mae', 'corr'])
    get_table("results/FI", 'app', ['O', 'C', 'E', 'A', 'N', 'racc', 'r2'])