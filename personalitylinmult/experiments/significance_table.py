import os
from pathlib import Path
import pandas as pd
from prettytable import PrettyTable


def format_value(value):

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if abs(value) < 1e-3 or abs(value) > 1e+3:
            return f"{value:.3e}" # Scientific notation
        else:
            return f"{value:.3f}" # Rounded to 3 decimals

    return value # Return non-numerical values as is


def get_table(base_dir, test_name: str):
    filepath = Path(base_dir) / f'{test_name}.csv'
    df = pd.read_csv(filepath)

    column_names = list(df.columns)
    [column_names.remove(elem) for elem in ['model_1', 'model_2', 'bonf_p-val', 'decision']]    
    field_names = ['model_1', 'model_2'] + column_names + ['bonf_p-val', 'decision']

    table = PrettyTable()
    table.field_names = field_names

    for _, row in df.iterrows():
        table.add_row([format_value(row[col]) for col in field_names])

    print(table)


if __name__ == "__main__":
    get_table('results/significance/MOSI', 'two_sided_paired_t_test')
    get_table('results/significance/MOSEI', 'two_sided_paired_t_test')
    get_table('results/significance/FI', 'two_sided_paired_t_test')