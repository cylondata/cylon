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

import pyarrow as pa
import numpy as np
import pandas as pd
import pycylon as cn
from pycylon import CylonContext
from pycylon import Table
from bench_util import get_dataframe
import time
import argparse

"""
Run benchmark:

>>> python python/examples/op_benchmark/compute_benchmark.py --start_size 1_000_000_000 \
                                        --step_size 10_000_000 \
                                        --end_size 30_000_000 \
                                        --num_cols 2 \
                                        --filter_size 1_000_000 \
                                        --stats_file /tmp/isin_bench_v1.csv \
                                        --repetitions 1 \
                                        --duplication_factor 0.9
"""


def isin_op(num_rows: int, num_cols: int, filter_size: int, duplication_factor: float):
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    ctx.add_config("compute_engine", "arrow")
    df = get_dataframe(num_rows=num_rows, num_cols=num_cols, duplication_factor=duplication_factor)

    ct = Table.from_pandas(ctx, df)

    cmp_data = np.random.randn(filter_size)

    cmp_data = cmp_data.tolist()

    pandas_time = time.time()
    df.isin(cmp_data)
    pandas_time = time.time() - pandas_time

    cylon_time = time.time()
    ct.isin(cmp_data)
    cylon_time = time.time() - cylon_time

    pandas_eval_time = time.time()
    pd.eval('df.isin(cmp_data)')
    pandas_eval_time = time.time() - pandas_eval_time

    return pandas_time, cylon_time, pandas_eval_time


def bench_isin(start: int, end: int, step: int, num_cols: int, filter_size: int, repetitions: int, stats_file: str,
               duplication_factor: float):
    all_data = []
    schema = ["num_records", "num_cols", "filter_size", "pandas", "cylon", "pandas_eval", "speed up", "speed up [eval]"]
    assert repetitions >= 1
    assert start > 0
    assert step > 0
    assert num_cols > 0
    assert filter_size > 0
    for records in range(start, end + step, step):
        print(f"Isin Op : Records={records}, Columns={num_cols}, Filter Size={filter_size}")
        times = []
        for idx in range(repetitions):
            pandas_time, cylon_time, pandas_eval_time = isin_op(num_rows=records, num_cols=num_cols,
                                                                filter_size=filter_size,
                                                                duplication_factor=duplication_factor)
            times.append([pandas_time, cylon_time, pandas_eval_time])
        times = np.array(times).sum(axis=0) / repetitions
        print(f"Isin Op : Records={records}, Columns={num_cols}, Filter Size={filter_size}, "
              f"Pandas Time : {times[0]}, Cylon Time : {times[1]}, Pandas [eval] Time : {times[2]}")
        all_data.append(
            [records, num_cols, filter_size, times[0], times[1], times[2], times[0] / times[1], times[2] / times[1]])
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
    bench_isin(start=args.start_size,
               end=args.end_size,
               step=args.step_size,
               num_cols=args.num_cols,
               filter_size=args.filter_size,
               repetitions=args.repetitions,
               stats_file=args.stats_file,
               duplication_factor=args.duplication_factor)

