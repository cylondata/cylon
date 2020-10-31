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

import operator
import os

from pandas import DataFrame
from pycylon import CylonContext
from pycylon import Table
from pycylon.io import CSVReadOptions
from pycylon.io import read_csv
from pyarrow import Table as PTable

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

1. ==   -> operator.__eq__
2. !=   -> operator.__ne__    
3. <    -> operator.__lt__
4. >    -> operator.__gt__
5. <=   -> operator.__le__
6. >=   -> operator.__ge__

'''


def test_properties():
    ctx: CylonContext = CylonContext(config=None, distributed=False)

    table1_path = '/tmp/user_usage_tm_1.csv'
    table2_path = '/tmp/user_usage_tm_2.csv'

    assert os.path.exists(table1_path) and os.path.exists(table2_path)

    csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30)

    tb: Table = read_csv(ctx, table1_path, csv_read_options)

    pdf = tb.to_pandas()

    def generate_filter_and_result(op, column: str, input, comparison_value):
        if column:
            filter = op(input[column], comparison_value)
            return filter, input[filter]
        else:
            filter = op(input, comparison_value)
            return filter, input[filter]

    def do_comparison_on_pdf_and_tb(tb_filter: Table, tb_result: Table, pdf_filter: DataFrame,
                                    pdf_result: DataFrame, is_full_table):

        if is_full_table:
            assert tb_filter.to_pandas().values.tolist() == pdf_filter.values.tolist()
            assert tb_result.to_pandas().fillna(0).values.tolist() == pdf_result.fillna(
                0).values.tolist()
        else:
            assert tb_filter.to_pandas().values.flatten().tolist() == pdf_filter.values.tolist()
            assert tb_result.to_pandas().values.tolist() == pdf_result.values.tolist()

    ops = [operator.__eq__, operator.__ne__, operator.__lt__, operator.__gt__, operator.__le__,
           operator.__ge__]
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


def test_filter():
    ctx: CylonContext = CylonContext(config=None, distributed=False)

    table1_path = '/tmp/user_usage_tm_1.csv'
    table2_path = '/tmp/user_usage_tm_2.csv'

    assert os.path.exists(table1_path) and os.path.exists(table2_path)

    csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30)

    tb: Table = read_csv(ctx, table1_path, csv_read_options)

    column_name = 'monthly_mb'

    ops = [operator.__or__, operator.__and__]
    or_limits = [600, 5000, 15000]
    and_limits = [0, 5000, 1000]
    comp_op_or = [operator.__gt__, operator.__le__, operator.__gt__]
    comp_op_and = [operator.__gt__, operator.__le__, operator.__gt__]
    limits = [or_limits, and_limits]
    comp_ops = [comp_op_or, comp_op_and]

    for op, limit, comp_op in zip(ops, limits, comp_ops):
        print("Op ", op)
        tb_cond_1 = comp_op[0](tb[column_name], limit[0])
        tb_cond_2 = comp_op[1](tb[column_name], limit[1])
        tb_cond_3 = comp_op[2](tb[column_name], limit[2])

        res_1_op = op(tb_cond_1, tb_cond_2)
        res_2_op = op(res_1_op, tb_cond_3)

        res_1 = tb[res_1_op]
        res_2 = tb[res_2_op]

        column_pdf_1 = res_1[column_name].to_pandas()
        column_pdf_2 = res_2[column_name].to_pandas()

        column_1 = column_pdf_1[column_name]
        for col in column_1:
            assert op(comp_op[0](col, limit[0]), comp_op[1](col, limit[1]))

        column_2 = column_pdf_2[column_name]
        for col in column_2:
            assert op(op(comp_op[0](col, limit[0]), comp_op[1](col, limit[1])),
                      comp_op[2](col, limit[2]))


def test_additional_properties():
    ctx: CylonContext = CylonContext(config=None, distributed=False)

    table1_path = '/tmp/user_usage_tm_1.csv'

    assert os.path.exists(table1_path)

    csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30)

    tb: Table = read_csv(ctx, table1_path, csv_read_options)

    tb_new = tb.drop(['outgoing_sms_per_month'])

    print(tb_new)






test_additional_properties()