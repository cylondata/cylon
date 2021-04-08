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


def get_random_data_column(num_rows: int, duplication_factor: float, with_null: bool = False,
                           null_per: float = 0.9, stringify: bool = False):
    if with_null:
        null_row_count = int(num_rows * null_per)
        gen_record_size = int(num_rows * duplication_factor)
        null_col = [None] * null_row_count
        data_col = np.random.randint(gen_record_size, size=num_rows - null_row_count).tolist()
        null_col = null_col + data_col
        return null_col
    else:
        gen_record_size = int(num_rows * duplication_factor)
        return np.random.randint(gen_record_size, size=num_rows)


def get_dataframe(num_rows: int, num_cols: int, duplication_factor: float, with_null: bool = False,
                  null_per: float = 0.9, stringify: bool = False):
    if with_null:
        pdf = pd.DataFrame({'data{}'.format(i): get_random_data_column(num_rows=num_rows,
                                                                       duplication_factor=duplication_factor,
                                                                       with_null=with_null,
                                                                       null_per=null_per,
                                                                       stringify=stringify)
                            for i in range(num_cols)})
        pdf = pdf.sample(frac=1)
        if stringify:
            return pdf.astype('str')
        return pdf
    else:
        pdf = pd.DataFrame({'data{}'.format(i): get_random_data_column(num_rows=num_rows,
                                                                       duplication_factor=duplication_factor,
                                                                       stringify=stringify)
                            for i in range(num_cols)})
        if stringify:
            return pdf.astype('str')
        return pdf
