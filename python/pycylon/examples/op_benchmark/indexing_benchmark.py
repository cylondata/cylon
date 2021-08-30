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

import time
import pandas as pd
import numpy as np
import pycylon as cn
from pycylon import Table
from pycylon.io import CSVWriteOptions
from pycylon.indexing.index import IndexingType
import argparse
from bench_util import get_dataframe

"""
Run benchmark:

>>> python python/examples/op_benchmark/indexing_benchmark.py --start_size 1_000_000 \
                                        --step_size 1_000_000 \
                                        --end_size 10_000_000 \
                                        --num_cols 2 \
                                        --stats_file /tmp/indexing_bench.csv \
                                        --unique_factor 0.1 \
                                        --repetitions 1
"""


def indexing_op(num_rows: int, num_cols: int, unique_factor: float):
    from pycylon.indexing.index import IndexingType
    ctx: cn.CylonContext = cn.CylonContext(config=None, distributed=False)
    pdf = get_dataframe(num_rows=num_rows, num_cols=num_cols, unique_factor=unique_factor)
    filter_column = pdf.columns[0]
    filter_column_data = pdf[pdf.columns[0]]
    random_index = np.random.randint(low=0, high=pdf.shape[0])
    filter_value = filter_column_data.values[random_index]
    filter_values = filter_column_data.values.tolist()[0:pdf.shape[0] // 2]
    tb = Table.from_pandas(ctx, pdf)
    cylon_indexing_time = time.time()
    tb.set_index(filter_column, indexing_type=IndexingType.LINEAR, drop=True)
    cylon_indexing_time = time.time() - cylon_indexing_time
    pdf_indexing_time = time.time()
    pdf.set_index(filter_column, drop=True, inplace=True)
    pdf_indexing_time = time.time() - pdf_indexing_time

    cylon_filter_time = time.time()
    tb_filter = tb.loc[filter_values]
    cylon_filter_time = time.time() - cylon_filter_time

    pandas_filter_time = time.time()
    pdf_filtered = pdf.loc[filter_values]
    pandas_filter_time = time.time() - pandas_filter_time

    print(tb_filter.shape, pdf_filtered.shape)

    return pandas_filter_time, cylon_filter_time, pdf_indexing_time, cylon_indexing_time


def bench_indexing_op(start: int, end: int, step: int, num_cols: int, repetitions: int, stats_file: str,
                      unique_factor: float):
    all_data = []
    schema = ["num_records", "num_cols", "pandas_loc", "cylon_loc", "speed up loc", "pandas_indexing", "cylon_indexing",
              "speed up indexing"]
    assert repetitions >= 1
    assert start > 0
    assert step > 0
    assert num_cols > 0
    for records in range(start, end + step, step):
        times = []
        for idx in range(repetitions):
            pandas_filter_time, cylon_filter_time, pdf_indexing_time, cylon_indexing_time = indexing_op(
                num_rows=records, num_cols=num_cols,
                unique_factor=unique_factor)
            times.append([pandas_filter_time, cylon_filter_time, pdf_indexing_time, cylon_indexing_time])
        times = np.array(times).sum(axis=0) / repetitions
        print(
            f"Loc Op : Records={records}, Columns={num_cols}, Pandas Loc Time : {times[0]}, "
            f"Cylon Loc Time : {times[1]}, "
            f"Pandas Indexing Time : {times[2]}, Cylon Indexing Time : {times[3]}")
        all_data.append(
            [records, num_cols, times[0], times[1], times[0] / times[1], times[2], times[3], times[2] / times[3]])
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
    parser.add_argument("-d", "--unique_factor",
                        help="random data unique factor",
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
    print(f"Data Unique Factor : {args.unique_factor}")
    print(f"Number of Columns : {args.num_cols}")
    print(f"Number of Repetitions : {args.repetitions}")
    print(f"Stats File : {args.stats_file}")
    bench_indexing_op(start=args.start_size,
                      end=args.end_size,
                      step=args.step_size,
                      num_cols=args.num_cols,
                      repetitions=args.repetitions,
                      stats_file=args.stats_file,
                      unique_factor=args.unique_factor)
