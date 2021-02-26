import operator
import os

from pandas import DataFrame
from pycylon import CylonContext
from pycylon import Table
from pycylon.io import CSVReadOptions
from pycylon.io import read_csv
import pyarrow as pa
import numpy as np
import pandas as pd

import time

ctx = CylonContext(config=None, distributed=False)
num_rows = 1_000_000
num_columns = 2


def get_random_array():
    return np.random.randn(num_rows)


def get_pandas_dataframe():
    return pd.DataFrame({'data{}'.format(i): get_random_array()
                         for i in range(num_columns)})


def perf_test_pdf_values():

    pdf = get_pandas_dataframe()

    tb1 = Table.from_pandas(ctx, pdf)

    tb1.set_index(tb1.column_names[0], drop=True)
    pdf.set_index(pdf.columns[0], drop=True, inplace=True)

    t1 = time.time()
    p_dict = tb1.to_pydict()
    t2 = time.time()
    values = pdf.values
    t3 = time.time()
    p_np = tb1.to_numpy()
    t4 = time.time()
    print(f"Pandas Value Conversion Time: : {t2 - t1}")
    print(f"PyCylon Value Conversion Time[dict]: : {t3 - t2}")
    print(f"PyCylon Value Conversion Time[numpy]: : {t4 - t3}")


def perf_test():
    pdf = get_pandas_dataframe()

    tb1 = Table.from_pandas(ctx, pdf)

    tb1.set_index(tb1.column_names[0], drop=True)
    pdf.set_index(pdf.columns[0], drop=True, inplace=True)

    t1 = time.time()
    for idx, row in pdf.iterrows():
        idx_t = idx + 1
        row_t = row // 2
    t2 = time.time()
    for idx, row in tb1.iterrows():
        idx_t = idx + 1
        row_t = row // 2
    t3 = time.time()
    print(f"Pandas Iterrow Time : {t2 - t1}")
    print(f"PyClon Iterrow Time : {t3 - t2}")


perf_test_pdf_values()
perf_test()
