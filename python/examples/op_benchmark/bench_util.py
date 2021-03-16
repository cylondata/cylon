import os
import numpy as np
import pandas as pd


def add_record_to_stats_file(file_path: str, record: str):
    if os.path.exists(file_path):
        with open(file_path, "a+") as fp:
            fp.write(record + "\n")
    else:
        with open(file_path, "w") as fp:
            fp.write(record + "\n")


def get_random_data_column(num_rows: int, duplication_factor: float):
    gen_record_size = int(num_rows * duplication_factor)
    return np.random.randint(gen_record_size, size=num_rows)


def get_dataframe(num_rows: int, num_cols: int, duplication_factor: float):
    return pd.DataFrame({'data{}'.format(i): get_random_data_column(num_rows, duplication_factor)
                         for i in range(num_cols)})
