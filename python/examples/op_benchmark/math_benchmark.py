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
import time
import argparse
from bench_util import get_dataframe
from operator import add, sub, mul, truediv

"""
Run benchmark:

>>> python python/examples/op_benchmark/math_benchmark.py --start_size 1_000_000 \
                                        --step_size 1_000_000 \
                                        --end_size 10_000_000 \
                                        --num_cols 2 \
                                        --stats_file /tmp/filter_bench.csv \
                                        --repetitions 1 \
                                        --duplication_factor 0.9 \
                                        --op add
"""


def math_op_base():
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    num_rows = 10_000_000
    data = np.random.randn(num_rows)

    df = pd.DataFrame({'data{}'.format(i): data
                       for i in range(100)})

    np_key = np.random.randint(0, 100, size=num_rows)
    np_all = df.to_numpy()

    df['key'] = np_key

    rb = pa.record_batch(df)
    t = pa.Table.from_pandas(df)

    ct = Table.from_pandas(ctx, df)

    t1 = time.time()
    np_key + 1
    t2 = time.time()
    ct['key'] + 1
    t3 = time.time()
    df['key'] + 1
    t4 = time.time()
    artb = ct.to_arrow().combine_chunks()
    ar_key = ct['key'].to_arrow().combine_chunks().columns[0].chunks[0]
    pc.add(ar_key, 1)
    t5 = time.time()

    print(f"Numpy Time: {t2 - t1} s")
    print(f"PyCylon Time: {t3 - t2} s")
    print(f"Pandas Time: {t4 - t3} s")
    print(f"PyArrow Time: {t5 - t4} s")


def math_op(num_rows: int, num_cols: int, duplication_factor: float, op=add):
    ctx: CylonContext = CylonContext(config=None, distributed=False, compute_engine='numpy')
    ctx.add_config("compute_engine", "numpy")

    pdf = get_dataframe(num_rows=num_rows, num_cols=num_cols, duplication_factor=duplication_factor)
    filter_column = pdf.columns[0]
    filter_column_data = pdf[pdf.columns[0]]
    random_index = np.random.randint(low=0, high=pdf.shape[0])
    math_value = filter_column_data.values[random_index]
    tb = Table.from_pandas(ctx, pdf)

    print("Table Context Compute Engine: ", tb.context.compute_engine)

    cylon_math_op_time = time.time()
    tb_filter = op(tb[filter_column], math_value)
    cylon_math_op_time = time.time() - cylon_math_op_time

    pandas_math_op_time = time.time()
    pdf_filter = op(pdf[filter_column], math_value)  # pdf[filter_column] > filter_value
    pandas_math_op_time = time.time() - pandas_math_op_time

    pandas_eval_math_op_time = time.time()
    pdf_filter = pd.eval("op(pdf[filter_column], math_value)")
    pandas_eval_math_op_time = time.time() - pandas_eval_math_op_time

    return pandas_math_op_time, pandas_eval_math_op_time, cylon_math_op_time


def bench_math_op(start: int, end: int, step: int, num_cols: int, repetitions: int, stats_file: str,
                  duplication_factor: float, op=None):
    all_data = []
    schema = ["num_records", "num_cols", "pandas_math_op", "pandas_eval_math_op", "cylon_math_op",
              "speed up math op", "speed up math op [eval]"]
    assert repetitions >= 1
    assert start > 0
    assert step > 0
    assert num_cols > 0
    for records in range(start, end + step, step):
        times = []
        for idx in range(repetitions):
            pandas_math_op_time, pandas_eval_math_op_time, cylon_math_op_time = math_op(
                num_rows=records, num_cols=num_cols,
                duplication_factor=duplication_factor, op=op)
            times.append([pandas_math_op_time, pandas_eval_math_op_time, cylon_math_op_time])
        times = np.array(times).sum(axis=0) / repetitions
        print(f"Filter Op : Records={records}, Columns={num_cols}"
              f"Pandas Math Op Time : {times[0]}, Pandas Math Op Time : {times[1]}, "
              f"PyCylon Math Op Time : {times[2]}")
        all_data.append(
            [records, num_cols, times[0], times[1], times[2], times[0] / times[2], times[1] / times[2]])
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
    parser.add_argument("-r", "--repetitions",
                        help="number of experiments to be repeated",
                        type=int)
    parser.add_argument("-f", "--stats_file",
                        help="stats file to be saved",
                        type=str)
    parser.add_argument("-o", "--op",
                        help="operator",
                        type=str)

    args = parser.parse_args()
    print(f"Start Data Size : {args.start_size}")
    print(f"End Data Size : {args.end_size}")
    print(f"Step Data Size : {args.step_size}")
    print(f"Data Duplication Factor : {args.duplication_factor}")
    print(f"Number of Columns : {args.num_cols}")
    print(f"Number of Repetitions : {args.repetitions}")
    print(f"Stats File : {args.stats_file}")
    op_type = args.op

    ops = {'add': add, 'sub': sub, 'mul': mul, 'div': truediv}

    op = ops[op_type]

    bench_math_op(start=args.start_size,
                  end=args.end_size,
                  step=args.step_size,
                  num_cols=args.num_cols,
                  repetitions=args.repetitions,
                  stats_file=args.stats_file,
                  duplication_factor=args.duplication_factor,
                  op=op)
