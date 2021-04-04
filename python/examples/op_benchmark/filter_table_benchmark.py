##
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##


import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import numpy as np
from pycylon import Table
from pycylon import CylonContext
from bench_util import get_dataframe
import time
import argparse

"""
Run benchmark:

>>> python python/examples/op_benchmark/filter_benchmark.py --start_size 1_000_000 \
                                        --step_size 1_000_000 \
                                        --end_size 10_000_000 \
                                        --num_cols 2 \
                                        --stats_file /tmp/filter_bench.csv \
                                        --repetitions 1 \
                                        --duplication_factor 0.9
"""


def fixed_filter_bench():
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    num_rows = 10_000_000
    data = np.random.randn(num_rows)

    df = pd.DataFrame({'data{}'.format(i): data
                       for i in range(2)})

    np_key = np.random.randint(0, 100, size=num_rows)
    np_all = df.to_numpy()

    df['key'] = np_key

    rb = pa.record_batch(df)
    t = pa.Table.from_pandas(df)

    ct = Table.from_pandas(ctx, df)

    print(ct.shape, df.shape)
    pdf_time = []
    ct_time = []
    rep = 1

    t1 = time.time()
    ct_filter = ct['key'] > 5
    t2 = time.time()
    df_filter = df['key'] > 5
    t3 = time.time()
    ct_res = ct[ct_filter]
    t4 = time.time()
    df_res = df[df_filter]
    t5 = time.time()
    np_filter = np_key > 5
    t6 = time.time()
    np_res = np_all[np_filter]
    t7 = time.time()

    print(f"PyCylon filter time :  {t2 - t1} s")
    print(f"Pandas filter time: {t3 - t2} s")
    print(f"Numpy filter time: {t6 - t5} s")
    print(f"PyCylon assign time: {t4 - t3} s")
    print(f"Pandas assign time: {t5 - t4} s")
    print(f"Numpy assign time: {t7 - t6} s")

    artb = t

    artb_filter = ct_filter.to_arrow().combine_chunks()
    artb_array_filter = artb_filter.columns[0].chunks[0]
    t_ar_s = time.time()
    artb = artb.combine_chunks()
    from pyarrow import compute
    res = []
    for chunk_arr in artb.itercolumns():
        res.append(chunk_arr.filter(artb_array_filter))
    t_ar_e = time.time()
    res_t = pa.Table.from_arrays(res, ct.column_names)
    t_ar_e_2 = time.time()
    print(f"PyArrow Filter Time : {t_ar_e - t_ar_s}")
    print(f"PyArrow Table Creation : {t_ar_e_2 - t_ar_e}")


def filter_op(num_rows: int, num_cols: int, duplication_factor: float):
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    ctx.add_config("compute_engine", "numpy")
    pdf = get_dataframe(num_rows=num_rows, num_cols=num_cols, duplication_factor=duplication_factor)
    filter_column  = pdf.columns[0]
    filter_column_data = pdf[pdf.columns[0]]
    random_index = np.random.randint(low=0, high=pdf.shape[0])
    filter_value = filter_column_data.values[random_index]
    tb = Table.from_pandas(ctx, pdf)

    cylon_filter_cr_time = time.time()
    tb_filter = tb[filter_column] > filter_value
    cylon_filter_cr_time = time.time() - cylon_filter_cr_time

    cylon_filter_time = time.time()
    tb_filtered = tb[tb_filter]
    cylon_filter_time = time.time() - cylon_filter_time

    pandas_filter_cr_time = time.time()
    pdf_filter = pdf[filter_column] > filter_value
    pandas_filter_cr_time = time.time() - pandas_filter_cr_time

    pandas_filter_time = time.time()
    pdf_filtered = pdf[pdf_filter]
    pandas_filter_time = time.time() - pandas_filter_time

    return pandas_filter_cr_time, pandas_filter_time, cylon_filter_cr_time, cylon_filter_time


def bench_filter_op(start: int, end: int, step: int, num_cols: int, repetitions: int, stats_file: str,
                    duplication_factor: float):
    all_data = []
    schema = ["num_records", "num_cols", "pandas_filter_cr", "cylon_filter_cr", "pandas_filter",
              "cylon_filter", "speed up filter cr", "speed up filter"]
    assert repetitions >= 1
    assert start > 0
    assert step > 0
    assert num_cols > 0
    for records in range(start, end + step, step):
        times = []
        for idx in range(repetitions):
            pandas_filter_cr_time, pandas_filter_time, cylon_filter_cr_time, cylon_filter_time = filter_op(
                num_rows=records, num_cols=num_cols,
                duplication_factor=duplication_factor)
            times.append([pandas_filter_cr_time, pandas_filter_time, cylon_filter_cr_time, cylon_filter_time])
        times = np.array(times).sum(axis=0) / repetitions
        print(f"Filter Op : Records={records}, Columns={num_cols}"
              f"Pandas Filter Creation Time : {times[0]}, Cylon Filter Creation Time : {times[2]}, "
              f"Pandas Filter Time : {times[1]}, Cylon Filter Time : {times[3]}")
        all_data.append(
            [records, num_cols, times[0], times[2], times[1], times[3], times[0] / times[2], times[1] / times[3]])
    pdf = pd.DataFrame(all_data, columns=schema)
    print(pdf)
    pdf.to_csv(stats_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--start_size",
                        help="initial data size",
                        type=int)
    parser.add_argument("-e", "--end_size",
                        help="end data size",
                        type=int)
    parser.add_argument("-d", "--duplication_factor",
                        help="random data duplication factor",
                        type=float)
    parser.add_argument("-s", "--step_size",
                        help="Step size",
                        type=int)
    parser.add_argument("-c", "--num_cols",
                        help="number of columns",
                        type=int)
    parser.add_argument("-t", "--filter_size",
                        help="number of values per filter",
                        type=int)
    parser.add_argument("-r", "--repetitions",
                        help="number of experiments to be repeated",
                        type=int)
    parser.add_argument("-f", "--stats_file",
                        help="stats file to be saved",
                        type=str)

    args = parser.parse_args()
    print(f"Start Data Size : {args.start_size}")
    print(f"End Data Size : {args.end_size}")
    print(f"Step Data Size : {args.step_size}")
    print(f"Data Duplication Factor : {args.duplication_factor}")
    print(f"Number of Columns : {args.num_cols}")
    print(f"Number of Repetitions : {args.repetitions}")
    print(f"Stats File : {args.stats_file}")
    bench_filter_op(start=args.start_size,
                    end=args.end_size,
                    step=args.step_size,
                    num_cols=args.num_cols,
                    repetitions=args.repetitions,
                    stats_file=args.stats_file,
                    duplication_factor=args.duplication_factor)
