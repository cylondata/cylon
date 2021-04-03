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

>>> python python/examples/op_benchmark/table_creation_benchmark.py --start_size 1_000_000 \
                                        --step_size 1_000_000 \
                                        --end_size 10_000_000 \
                                        --num_cols 2 \
                                        --stats_file /tmp/table_creation_bench.csv \
                                        --repetitions 1 \
                                        --duplication_factor 0.9
"""


def tb_creation_op(num_rows: int, num_cols: int, duplication_factor: float):
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    ctx.add_config("compute_engine", "numpy")
    pdf = get_dataframe(num_rows=num_rows, num_cols=num_cols, duplication_factor=duplication_factor)
    # data_row_list
    data_row = [np.random.randint(num_rows)] * num_rows
    # data row np array
    data_row_ar = np.array(data_row)

    data_set = [data_row for i in range(num_cols)]
    data_set_ar = [data_row_ar for i in range(num_cols)]

    column_names = ["data_" + str(i) for i in range(num_cols)]

    t_pandas = time.time()
    tb = Table.from_pandas(ctx, pdf)
    t_pandas = time.time() - t_pandas

    t_list = time.time()
    tb1 = Table.from_list(ctx, column_names, data_set)
    t_list = time.time() - t_list

    t_numpy = time.time()
    tb2 = Table.from_numpy(ctx, column_names, data_set_ar)
    t_numpy = time.time() - t_numpy

    return t_pandas, t_numpy, t_list


def bench_tb_creation_op(start: int, end: int, step: int, num_cols: int, repetitions: int, stats_file: str,
                         duplication_factor: float):
    all_data = []
    schema = ["num_records", "num_cols", "from_pandas", "from_list", "from_numpy"]
    assert repetitions >= 1
    assert start > 0
    assert step > 0
    assert num_cols > 0
    for records in range(start, end + step, step):
        times = []
        for idx in range(repetitions):
            t_pandas, t_numpy, t_list = tb_creation_op(
                num_rows=records, num_cols=num_cols,
                duplication_factor=duplication_factor)
            times.append([t_pandas, t_numpy, t_list])
        times = np.array(times).sum(axis=0) / repetitions
        print(f"Table Creation Op : Records={records}, Columns={num_cols}"
              f"From Pandas  : {times[0]}, From List : {times[1]}, "
              f"From Numpy : {times[2]}")
        all_data.append(
            [records, num_cols, times[0], times[1], times[2]])
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
    bench_tb_creation_op(start=args.start_size,
                         end=args.end_size,
                         step=args.step_size,
                         num_cols=args.num_cols,
                         repetitions=args.repetitions,
                         stats_file=args.stats_file,
                         duplication_factor=args.duplication_factor)
