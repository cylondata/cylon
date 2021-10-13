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

'''
utils for tests
'''


import cupy as cp
import cudf


def create_sorted_cudf_df(ncols, nrows, start=0, step=1):
    df_local = cudf.DataFrame()
    data_start = start
    for i in range(ncols):
        df_local["col-" + str(i)] = cp.arange(start=data_start, stop=data_start + nrows * step, step=step, dtype="int64")
        data_start += nrows * step
    return df_local


def create_random_data_df(ncols, nrows, low=0, high=1000000000000):
    df_local = cudf.DataFrame()
    for i in range(ncols):
        df_local["col-" + str(i)] = cp.random.randint(low=low, high=high, size=nrows, dtype="int64")
    return df_local


def random_data_df(nrows, col_lows=[0, 100], col_highs=[100, 200]):
    df_local = cudf.DataFrame()
    for i in range(len(col_lows)):
        df_local["col-" + str(i)] = cp.random.randint(low=col_lows[i], high=col_highs[i], size=nrows, dtype="int64")
    return df_local


import string
import random


def random_str(size=6, chars=string.ascii_uppercase + string.digits):
    """
    generate a random string with given size and char list
    source: https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits?rq=1
    """
    return ''.join(random.choice(chars) for _ in range(size))


def create_random_str_df(ncols, nrows, min_digits=2, max_digits=7):
    df_local = cudf.DataFrame()
    for i in range(ncols):
        df_local["col-" + str(i)] = [random_str(min_digits + i % (max_digits - min_digits)) for i in range(nrows)]
    return df_local
