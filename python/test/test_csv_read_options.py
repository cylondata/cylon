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
Run test:
>> pytest -q python/test/test_csv_read_options.py
'''

from pandas import DataFrame
from pycylon import CylonContext
from pycylon import Table
from pycylon.io import CSVReadOptions
from pycylon.io import read_csv
import pyarrow as pa
import numpy as np
import pandas as pd


def test_check_csv_read_opts():
    csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30)


def test_read_csv_with_use_cols():
    ctx = CylonContext(config=None, distributed=False)
    use_cols = ['a', 'b']
    csv_read_options = CSVReadOptions() \
        .use_threads(True) \
        .block_size(1 << 30) \
        .use_cols(use_cols)
    table_path = '/tmp/duplicate_data_0.csv'
    tb1: Table = read_csv(ctx, table_path, csv_read_options)
    pdf = pd.read_csv(table_path, usecols=use_cols)
    assert tb1.column_names == use_cols == pdf.columns.tolist()


def test_read_csv_with_skiprows():
    ctx = CylonContext(config=None, distributed=False)
    csv_read_options = CSVReadOptions() \
        .use_threads(True) \
        .block_size(1 << 30) \
        .skip_rows(1)
    table_path = '/tmp/duplicate_data_0.csv'
    tb1: Table = read_csv(ctx, table_path, csv_read_options)
    pdf = pd.read_csv(table_path, skiprows=1)
    print(tb1)
    print("-" * 80)
    print(pdf)
    assert tb1.to_pandas().values.tolist() == pdf.values.tolist()


def test_read_csv_with_na_values():
    ctx = CylonContext(config=None, distributed=False)
    csv_read_options = CSVReadOptions() \
        .use_threads(True) \
        .block_size(1 << 30) \
        .na_values(['na', 'none'])
    table_path = 'data/input/null_data.csv'
    tb1: Table = read_csv(ctx, table_path, csv_read_options)
    pdf = pd.read_csv(table_path, na_values=['na', 'none'])
    print(tb1)
    print("-" * 80)
    print(pdf)
    tb1 = tb1.fillna(0)
    pdf = pdf.fillna(0)
    assert tb1.to_pandas().values.tolist() == pdf.values.tolist()





