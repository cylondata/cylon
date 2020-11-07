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
import pyarrow as pa
import numpy as np

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


def test_drop():
    ctx: CylonContext = CylonContext(config=None, distributed=False)

    table1_path = '/tmp/user_usage_tm_1.csv'

    assert os.path.exists(table1_path)

    csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30)

    tb: Table = read_csv(ctx, table1_path, csv_read_options)

    drop_column = 'outgoing_sms_per_month'

    tb_new = tb.drop([drop_column])

    assert not tb_new.column_names.__contains__(drop_column)


def test_fillna():
    col_names = ['col1', 'col2']
    data_list_numeric = [[1, 2, None, 4, 5], [6, 7, 8, 9, None]]
    fill_value = 0
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    cn_tb_numeric = Table.from_list(ctx, col_names, data_list_numeric)

    cn_tb_numeric_fillna = cn_tb_numeric.fillna(fill_value)

    data_list = list(cn_tb_numeric_fillna.to_pydict().values())
    for col in data_list:
        assert not col.__contains__(None)
        assert col.__contains__(fill_value)


def test_where():
    col_names = ['col1', 'col2']
    data_list_numeric = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    cn_tb = Table.from_list(ctx, col_names, data_list_numeric)

    cn_tb_where = cn_tb.where(cn_tb > 3)

    print(cn_tb_where)

    cn_tb_where_with_other = cn_tb.where(cn_tb > 3, 100)

    print(cn_tb_where_with_other)

    print(cn_tb > 3)


def test_rename():
    col_names = ['col1', 'col2', 'col3', 'col4']
    data_list_numeric = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15],
                         [16, 17, 18, 19, 20]]
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    cn_tb = Table.from_list(ctx, col_names, data_list_numeric)

    prev_col_names = cn_tb.column_names
    # with dictionary
    columns = {'col1': 'col-1', 'col3': 'col-3'}
    cn_tb.rename(columns)

    new_col_names = cn_tb.column_names

    for key in columns:
        value = columns[key]
        assert prev_col_names.index(key) == new_col_names.index(value)

    # with list
    cn_tb_list = Table.from_list(ctx, col_names, data_list_numeric)
    prev_col_names = cn_tb_list.column_names
    new_column_names = ['col-1', 'col-2', 'col-3', 'col-4']
    cn_tb_list.rename(new_column_names)

    assert cn_tb_list.column_names == new_column_names


def test_invert():
    # Bool Invert Test

    data_list = [[False, True, False, True, True], [False, True, False, True, True]]
    pdf = DataFrame(data_list)
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    cn_tb = Table.from_pandas(ctx, pdf)

    invert_cn_tb = ~cn_tb
    invert_pdf = ~pdf

    assert invert_cn_tb.to_pandas().values.tolist() == invert_pdf.values.tolist()


def test_neg():
    npr = np.array([[1, 2, 3, 4, 5, -6, -7], [-1, -2, -3, -4, -5, 6, 7]])
    pdf = DataFrame(npr)
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    cn_tb: Table = Table.from_pandas(ctx, pdf)
    neg_cn_tb: Table = -cn_tb
    neg_pdf = -pdf
    assert neg_cn_tb.to_pandas().values.tolist() == neg_pdf.values.tolist()


def test_setitem():
    npr = np.array([[1, 2, 3, 4, 5], [-1, -2, -3, -4, -5]])
    pdf = DataFrame(npr)
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    cn_tb: Table = Table.from_pandas(ctx, pdf)
    # replacing an existing column
    cn_tb['0'] = cn_tb['4']
    assert cn_tb['0'].to_pandas().values.tolist() == cn_tb['4'].to_pandas().values.tolist()
    # adding a new column at the end
    cn_tb['5'] = cn_tb['4']
    assert cn_tb['5'].to_pandas().values.tolist() == cn_tb['4'].to_pandas().values.tolist()


def test_math_ops_for_scalar():
    npr = np.array([[20, 2, 3, 4, 5], [10, -20, -30, -40, -50], [10.2, 13.2, 16.4, 12.2, 10.8]])
    pdf = DataFrame(npr)
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    cn_tb: Table = Table.from_pandas(ctx, pdf)

    from operator import add, sub, mul, truediv
    ops = [add, sub, mul, truediv]

    for op in ops:
        cn_tb_1 = cn_tb
        pdf_1 = pdf
        # test column division
        cn_tb_1['0'] = op(cn_tb_1['0'], 2)
        pdf_1[0] = op(pdf_1[0], 2)

        assert pdf_1.values.tolist() == cn_tb_1.to_pandas().values.tolist()

        # test table division
        cn_tb_2 = cn_tb
        pdf_2 = pdf

        cn_tb_2 = op(cn_tb_2, 2)
        pdf_2 = op(pdf, 2)

        assert pdf_2.values.tolist() == cn_tb_2.to_pandas().values.tolist()


def test_math_i_ops_for_scalar():
    """
    TODO: Enhance Test case and functionality
        Check the following case : https://github.com/cylondata/cylon/issues/229
    >>> from operator import __iadd__
    >>> assert __iadd__(cylon_table, value) == (cylon_table += value)
    >>> Failure ...
    """
    npr = np.array([[20, 2, 3, 4, 5], [10, -20, -30, -40, -50], [12.2, 13.2, 16.4, 12.2, 10.8]])
    pdf = DataFrame(npr)
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    cn_tb: Table = Table.from_pandas(ctx, pdf)

    cn_tb_1 = cn_tb
    pdf_1 = pdf
    # test column addition

    cn_tb_1['0'] += 2
    pdf_1[0] += 2

    assert pdf_1.values.tolist() == cn_tb_1.to_pandas().values.tolist()

    cn_tb_1['0'] -= 2
    pdf_1[0] -= 2

    assert pdf_1.values.tolist() == cn_tb_1.to_pandas().values.tolist()

    cn_tb_1['0'] *= 2
    pdf_1[0] *= 2

    assert pdf_1.values.tolist() == cn_tb_1.to_pandas().values.tolist()

    cn_tb_1['0'] /= 2
    pdf_1[0] /= 2

    assert pdf_1.values.tolist() == cn_tb_1.to_pandas().values.tolist()

    # test table division
    cn_tb_2 = cn_tb
    pdf_2 = pdf

    cn_tb_2 += 2
    pdf += 2

    assert pdf_2.values.tolist() == cn_tb_2.to_pandas().values.tolist()

    cn_tb_2 -= 2
    pdf -= 2

    assert pdf_2.values.tolist() == cn_tb_2.to_pandas().values.tolist()

    cn_tb_2 *= 2
    pdf *= 2

    assert pdf_2.values.tolist() == cn_tb_2.to_pandas().values.tolist()

    cn_tb_2 /= 2
    pdf /= 2

    assert pdf_2.values.tolist() == cn_tb_2.to_pandas().values.tolist()
