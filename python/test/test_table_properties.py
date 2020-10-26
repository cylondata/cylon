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

"""
Run test:
>> pytest -q python/test/test_table_properties.py
"""

import os
from pycylon import Table
from pycylon import CylonContext
from pycylon.io import CSVReadOptions
from pycylon.io import read_csv
from pycylon.data import ComparisonOp
from pandas import DataFrame
import pyarrow as pa
from typing import Tuple

'''
Run test:
>> pytest -q python/test/test_table_properties.py
'''

'''
Test Cases for Comparison Operations
-------------------------------------

Comparison on DataFrame
------------------------

Case 1: Compare based on a column (each value in a column is checked against the comparison 
value)
Case 2: Compare based on the whole table (all values in the table is checked against the 
comparison value)

Comparison Operators
--------------------

1. ==   -> ComparisonOp.EQ
2. !=   -> ComparisonOp.NE    
3. <    -> ComparisonOp.LT
4. >    -> ComparisonOp.GT
5. <=   -> ComparisonOp.LE
6. >=   -> ComparisonOp.GE

'''


def test_properties():
    ctx: CylonContext = CylonContext(config=None, distributed=False)

    table1_path = '/tmp/user_usage_tm_1.csv'
    table2_path = '/tmp/user_usage_tm_2.csv'

    assert os.path.exists(table1_path) and os.path.exists(table2_path)

    csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30)

    tb: Table = read_csv(ctx, table1_path, csv_read_options)

    pdf = tb.to_pandas()

    def generate_filter_and_result(op: ComparisonOp, column: str, input, comparison_value):
        if op == ComparisonOp.EQ:
            if column:
                filter = input[column] == comparison_value
                return filter, input[filter]
            else:
                filter = input == comparison_value
                return filter, input[filter]

        elif op == ComparisonOp.NE:
            if column:
                filter = input[column] != comparison_value
                return filter, input[filter]
            else:
                filter = input != comparison_value
                return filter, input[filter]

        elif op == ComparisonOp.LT:
            if column:
                filter = input[column] < comparison_value
                return filter, input[filter]
            else:
                filter = input < comparison_value
                return filter, input[filter]

        elif op == ComparisonOp.GT:
            if column:
                filter = input[column] > comparison_value
                return filter, input[filter]
            else:
                filter = input > comparison_value
                return filter, input[filter]

        elif op == ComparisonOp.LE:
            if column:
                filter = input[column] <= comparison_value
                return filter, input[filter]
            else:
                filter = input <= comparison_value
                return filter, input[filter]

        elif op == ComparisonOp.GE:
            if column:
                filter = input[column] >= comparison_value
                return filter, input[filter]
            else:
                filter = input >= comparison_value
                return filter, input[filter]
        else:
            raise ValueError("Unsupported Comparison Operation")

    def do_comparison_on_pdf_and_tb(tb_filter: Table, tb_result: Table, pdf_filter: DataFrame,
                                    pdf_result: DataFrame, is_full_table=False):

        if is_full_table:
            assert tb_filter.to_pandas().values.tolist() == pdf_filter.values.tolist()
            assert tb_result.to_pandas().fillna(0).values.tolist() == pdf_result.fillna(
                0).values.tolist()
        else:
            assert tb_filter.to_pandas().values.flatten().tolist() == pdf_filter.values.tolist()
            assert tb_result.to_pandas().values.tolist() == pdf_result.values.tolist()

    ops = [ComparisonOp.EQ, ComparisonOp.NE, ComparisonOp.LT, ComparisonOp.GT, ComparisonOp.LE,
           ComparisonOp.GE]
    value = 519.12
    columns = ['monthly_mb', None]
    is_full_table_flags = [False, True]

    for column, is_full_table in zip(columns, is_full_table_flags):
        for op in ops:
            tb_filter_all, tb_filter_all_result = generate_filter_and_result(op, column, tb, value)

            pdf_filter_all, pdf_filter_all_result = generate_filter_and_result(op, column, pdf,
                                                                               value)

            do_comparison_on_pdf_and_tb(tb_filter=tb_filter_all, tb_result=tb_filter_all_result,
                                        pdf_filter=pdf_filter_all, pdf_result=pdf_filter_all_result,
                                        is_full_table=is_full_table)
