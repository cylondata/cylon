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
import cupy as cp
import cudf
import util


def merge_test(data_size, ncols, number_of_tables):
    data_start = 0
    data_step = 30

    total_rows = util.get_rows(data_size, ncols)
    rows_per_table = total_rows / number_of_tables

    tables = []
    for i in range(number_of_tables):
        tbl = util.create_sorted_cudf_df(ncols, rows_per_table, start=data_start, step=data_step)
        data_start += ncols * rows_per_table * (data_step / 3)
        tables.append(tbl)
#        print(" \ntable: " + str(i) + "\n")
#        print(tbl)

    t0 = time.time()
    merged = cudf.merge_sorted(tables)
    t1 = time.time()

#    print("\nmerged table: \n")
#    print(merged)

    delay = (t1 - t0) * 1000
#    print("merge delay: " + str(delay))
    return delay


def sort_or_quantile_test(ncols, data_size, dtype="int64", task_type="sort"):
    if task_type != "sort" and task_type != "quantile":
        raise ValueError("task_type can be either 'sort' or 'quantile'")

    total_rows = util.get_rows(data_size, ncols)
    if dtype == "int64":
        df = util.create_random_data_df(ncols, total_rows)
    elif dtype == "str":
        df = util.create_random_str_df(ncols, total_rows);

    print("memory usage: ", df.memory_usage())

    if task_type == "quantile":
        quantile_points = cp.linspace(0, 1, 100)

    t0 = time.time()
    if task_type == "sort":
        result_df = df.sort_values(by=df.columns)
    elif task_type == "quantile":
        result_df = df._quantiles(quantile_points, "NEAREST")
    t1 = time.time()

#    print("\nresult table: \n")
#    print(result_df)

    delay = (t1 - t0) * 1000
#    print("sort delay: " + str(delay))
    del df, result_df
    return delay


#####################################################

data_size = "100MB"
ncols = 1
number_of_tables = 20

tasks = ["sort", "quantile"]
types = ["int64", "str"]

for ttype in tasks:
    for dtype in types:
        print(f"testing {ttype} and {dtype}")
        delay = sort_or_quantile_test(ncols, data_size, dtype, ttype)
        print(f"{ttype} {dtype} delay: ", delay)

# for number_of_tables in range(5, 30, 5):
#     duration = merge_test(data_size, ncols, number_of_tables)
#     print(number_of_tables, " merge duration: ", duration)
