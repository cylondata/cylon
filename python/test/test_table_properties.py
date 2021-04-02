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
import pycylon as cn
import pyarrow as pa
import numpy as np
import pandas as pd

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


def test_string_type_filters():
    ctx: CylonContext = CylonContext()

    tb: Table = Table.from_pydict(ctx, {"A": ['a', 'b', 'c', 'ab', 'a'],
                                        "B": [1, 2, 3, 4, 5]})
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
    value = "a"
    columns = ["A"]
    is_full_table_flags = [False]

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
    index_values = [0, 1, 2, 3, 4]
    cn_tb.set_index(index_values)
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
    cn_tb_list.set_index(index_values)
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

    cn_tb['6'] = 1  # create new column
    assert np.array_equal(cn_tb['6'].to_pandas().values.flatten(), np.full(cn_tb.row_count, 1))

    cn_tb['6'] = 1.0  # replace column
    assert np.array_equal(cn_tb['6'].to_pandas().values.flatten(), np.full(cn_tb.row_count, 1.0))

    cn_tb['6'] = 'aaa'  # replace column
    assert np.array_equal(cn_tb['6'].to_pandas().values.flatten(), np.full(cn_tb.row_count, 'aaa'))


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


def test_math_ops_for_table_values():
    pdf = DataFrame({'0': [1, 2, 3, 4], '1': [5, 6, 7, 9], '2': [1., 2., 3., 4.]})
    ctx: CylonContext = CylonContext()
    cn_tb: Table = Table.from_pandas(ctx, pdf)

    from operator import add, sub, mul, truediv
    ops = [add]#, sub, mul, truediv]

    for op in ops:
        # test column division
        cn_res = op(cn_tb['0'], cn_tb['0'])
        pd_res = op(pdf['0'], pdf['0'])

        # pandas series.values returns an array, whereas dataframe.values list of lists. Hence it
        # needs to be flattened to compare
        assert pd_res.values.tolist() == cn_res.to_pandas().values.flatten().tolist()

        # test table division
        cn_res2 = op(cn_tb, cn_tb['0'])
        pd_res2 = getattr(pdf, op.__name__)(pdf['0'], axis=0)

        assert pd_res2.values.tolist() == cn_res2.to_pandas().values.tolist()


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


def test_i_bitwise_ops():
    # TODO: Improve test and functionality: https://github.com/cylondata/cylon/issues/229
    npr = np.array([[20, 2, 3, 4, 5], [10, -20, -30, -40, -50], [36.2, 13.2, 16.4, 12.2, 10.8]])
    pdf = DataFrame(npr)
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    cn_tb: Table = Table.from_pandas(ctx, pdf)

    a = cn_tb['0'] > 10
    b = cn_tb['1'] > 2
    a_pdf = pdf[0] > 10
    b_pdf = pdf[1] > 2

    d = a & b
    a &= b
    d_pdf = a_pdf & b_pdf
    a_pdf &= b_pdf

    assert d.to_pandas().values.tolist() == a.to_pandas().values.tolist()
    assert a.to_pandas().values.flatten().tolist() == a_pdf.values.tolist()

    ## OR

    a = cn_tb['0'] > 10
    b = cn_tb['1'] > 2
    a_pdf = pdf[0] > 10
    b_pdf = pdf[1] > 2

    d = a | b
    a |= b
    d_pdf = a_pdf | b_pdf
    a_pdf |= b_pdf

    assert d.to_pandas().values.tolist() == a.to_pandas().values.tolist()
    assert a.to_pandas().values.flatten().tolist() == a_pdf.values.tolist()


def test_add_prefix():
    npr = np.array([[20.2, 2.0, 3.2, 4.3, 5.5], [10, -20, -30, -40, -50], [36.2, 13.2, 16.4, 12.2,
                                                                           10.8]])
    pdf = DataFrame(npr)
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    cn_tb: Table = Table.from_pandas(ctx, pdf)
    prefix = "item_"
    cn_tb_with_prefix = cn_tb.add_prefix(prefix)
    pdf_with_prefix = pdf.add_prefix(prefix)

    assert pdf_with_prefix.columns.tolist() == cn_tb_with_prefix.column_names


def test_add_suffix():
    npr = np.array([[20.2, 2.0, 3.2, 4.3, 5.5], [10, -20, -30, -40, -50], [36.8, 13.2, 16.4, 12.2,
                                                                           10.8]])
    pdf = DataFrame(npr)
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    cn_tb: Table = Table.from_pandas(ctx, pdf)
    suffix = "item_"
    cn_tb_with_suffix = cn_tb.add_suffix(suffix)
    pdf_with_suffix = pdf.add_suffix(suffix)

    assert pdf_with_suffix.columns.tolist() == cn_tb_with_suffix.column_names


def test_empty_table():
    from pycylon.data.table import EmptyTable
    from pycylon.index import RangeIndex
    import pandas as pd
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    empt_tb = EmptyTable(ctx, RangeIndex(data=range(0, 0)))

    assert empt_tb.to_pandas().values.tolist() == pd.DataFrame().values.tolist()


def test_iterrows():
    ctx = CylonContext(config=None, distributed=False)
    csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30)
    table_path = '/tmp/duplicate_data_0.csv'
    tb1: Table = read_csv(ctx, table_path, csv_read_options)
    pdf: pd.DataFrame = tb1.to_pandas()

    tb1.set_index(tb1.column_names[0], drop=True)
    pdf.set_index(pdf.columns[0], drop=True, inplace=True)

    for p, c in zip(pdf.iterrows(), tb1.iterrows()):
        idx_p = p[0]
        row_p = p[1].tolist()
        idx_c = c[0]
        row_c = c[1]
        assert idx_p == idx_c
        assert row_p == row_c


def test_concat_table():
    """
        For Cylon concat operation:

        We can check for indexing column if default the index array contains [0,num_records-1)
        If indexed, the indexed column will be compared.

        We can use existing join ops.

        Algorithm
        =========

        axis=1 (regular join op considering a column)
        ----------------------------------------------

        1. If indexed or not, do a reset_index op (which will add the new column as 'index' in both
        tables)
        2. Do the regular join by considering the 'index' column
        3. Set the index by 'index' in the resultant table

        axis=0 (stacking tables or similar to merge function)
        -----------------------------------------------------
        assert: column count must match
        the two tables are stacked upon each other in order
        The index is created by concatenating two indices
    """
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    columns = ['c1', 'c2', 'c3']
    dataset_1 = [[1, 2, 3, 4, 5], [20, 30, 40, 50, 51], [33, 43, 53, 63, 73]]
    dataset_2 = [[1, 20, 3, 4, 50], [20, 30, 40, 50, 51], [33, 43, 53, 63, 73]]
    dataset_3 = [[1, 20, 3, 40, 50, 60], [21, 31, 41, 51, 50, 70], [32, 42, 52, 62, 72, 82]]

    tb1 = Table.from_list(ctx, columns, dataset_1)
    tb1 = tb1.add_prefix('d1_')

    tb2 = Table.from_list(ctx, columns, dataset_2)
    tb2 = tb2.add_prefix('d2_')

    tb3 = Table.from_list(ctx, columns, dataset_3)
    tb3 = tb3.add_prefix('d3_')

    tb4 = Table.from_list(ctx, columns, dataset_3)
    tb4 = tb4.add_prefix('d1_')

    pdf1 = tb1.to_pandas()
    pdf2 = tb2.to_pandas()
    pdf3 = tb3.to_pandas()
    pdf4 = tb4.to_pandas()

    print(tb1)
    print("-" * 80)
    print(tb2)

    tb1.set_index(tb1.column_names[0], drop=True)
    tb2.set_index(tb2.column_names[0], drop=True)
    tb3.set_index(tb3.column_names[0], drop=True)

    print("*" * 80)
    print("Indexed table")
    print(tb1)
    print("*" * 80)
    print("Reset_Index table")
    tb1.reset_index()
    print(tb1)
    print("*" * 80)

    pdf1.set_index(pdf1.columns[0], drop=True, inplace=True)
    pdf2.set_index(pdf2.columns[0], drop=True, inplace=True)
    pdf3.set_index(pdf3.columns[0], drop=True, inplace=True)

    print("=" * 80)
    print("axis=1")
    print("=" * 80)
    res_pdf_1 = pd.concat([pdf1, pdf2], join='inner', axis=1)
    print(res_pdf_1)
    print("-" * 80)
    res_pdf_2 = pd.concat([pdf1, pdf3], join='inner', axis=1)
    print(res_pdf_2)
    print("-" * 80)

    print("=" * 80)
    print("axis=0")
    print("=" * 80)
    res_pdf_1 = pd.concat([pdf1, pdf2], join='inner', axis=0)
    print(res_pdf_1)
    print("-" * 80)
    res_pdf_2 = pd.concat([pdf1, pdf3], join='inner', axis=0)
    print(res_pdf_2)
    print("-" * 80)
    res_pdf_3 = pd.concat([pdf1, pdf4], join='inner', axis=0)
    print(res_pdf_3)
    print("-" * 80)
    print("Multi Table Concat 1")
    res_pdf_4 = pd.concat([pdf1, pdf2, pdf3], join='inner', axis=1)
    print(res_pdf_4)
    print("Multi Table Concat 2")
    res_pdf_5 = pd.concat([pdf2, pdf3, pdf1], join='inner', axis=1)
    print(res_pdf_5)


def test_concat_op():
    from pycylon.net import MPIConfig
    mpi_config = MPIConfig()
    ctx: CylonContext = CylonContext(config=mpi_config, distributed=True)
    columns = ['c1', 'c2', 'c3']
    dataset_1 = [[1, 2, 3, 4, 5], [20, 30, 40, 50, 51], [33, 43, 53, 63, 73]]
    dataset_2 = [[1, 20, 3, 4, 50], [20, 30, 40, 50, 51], [33, 43, 53, 63, 73]]
    dataset_3 = [[1, 20, 3, 40, 50, 60], [21, 31, 41, 51, 50, 70], [32, 42, 52, 62, 72, 82]]

    tb1 = Table.from_list(ctx, columns, dataset_1)
    tb1 = tb1.add_prefix('d1_')

    tb2 = Table.from_list(ctx, columns, dataset_2)
    tb2 = tb2.add_prefix('d2_')

    tb3 = Table.from_list(ctx, columns, dataset_3)
    tb3 = tb3.add_prefix('d3_')

    tb4 = Table.from_list(ctx, columns, dataset_3)
    tb4 = tb4.add_prefix('d1_')

    pdf1 = tb1.to_pandas()
    pdf2 = tb2.to_pandas()
    pdf3 = tb3.to_pandas()
    pdf4 = tb4.to_pandas()

    print(tb1)
    print("-" * 80)
    print(tb2)

    tb1.set_index(tb1.column_names[0], drop=True)
    tb2.set_index(tb2.column_names[0], drop=True)
    tb3.set_index(tb3.column_names[0], drop=True)
    tb4.set_index(tb4.column_names[0], drop=True)

    print("*" * 80)
    print("Indexed table")
    print(tb1)
    print("*" * 80)

    pdf1.set_index(pdf1.columns[0], drop=True, inplace=True)
    pdf2.set_index(pdf2.columns[0], drop=True, inplace=True)
    pdf3.set_index(pdf3.columns[0], drop=True, inplace=True)
    pdf4.set_index(pdf4.columns[0], drop=True, inplace=True)

    print("=" * 80)
    print("axis=1")
    print("=" * 80)
    res_pdf_1 = pd.concat([pdf1, pdf2], join='inner', axis=1)
    print(res_pdf_1)
    print("-" * 80)
    tables = [tb1, tb2]
    tb1_index_values = tb1.index.index_values
    tb2_index_values = tb2.index.index_values
    res_tb_1 = Table.concat(tables, join='inner', axis=1)
    print(res_tb_1)
    print("-" * 80)
    res_pdf_2 = pd.concat([pdf1, pdf2], join='inner', axis=1)
    print(res_pdf_2)
    assert res_pdf_2.values.tolist() == res_tb_1.to_pandas().values.tolist()
    assert res_tb_1.index.index_values == res_pdf_2.index.values.tolist()
    print("-" * 80)
    print(tb1.to_arrow())
    print(tb2.to_arrow())
    print(tb1.index.index_values, tb1_index_values)
    print(tb2.index.index_values, tb2_index_values)
    assert tb1.index.index_values.sort() == tb1_index_values.sort()
    assert tb2.index.index_values.sort() == tb2_index_values.sort()
    print("=" * 80)
    print("axis=0")
    print("=" * 80)
    res_pdf_3 = pd.concat([pdf1, pdf4], join='inner', axis=0)
    print(tb1.column_names, tb4.column_names)
    res_tb_2 = Table.concat([tb1, tb4], join='inner', axis=0)
    print(res_tb_2)
    print(res_tb_2.index.index_values)
    print(res_pdf_3)
    print(res_pdf_3.index.values.tolist())
    assert res_pdf_3.values.tolist() == res_tb_2.to_pandas().values.tolist()
    assert res_tb_2.index.index_values == res_pdf_3.index.values.tolist()


def test_astype():
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    columns = ['c1', 'c2', 'c3']
    dataset_1 = [[1, 2, 3, 4, 5], [20, 30, 40, 50, 51], [33, 43, 53, 63, 73]]
    tb = Table.from_list(ctx, columns, dataset_1)
    pdf: pd.DataFrame = tb.to_pandas()
    tb.set_index('c1', drop=True)
    pdf.set_index('c1', inplace=True)

    print(tb)
    print("-" * 80)
    print(pdf)
    print("-" * 80)

    pdf_astype = pdf.astype(float)

    tb_astype = tb.astype(float)

    print(tb_astype)
    print("-" * 80)
    print(pdf_astype)

    assert pdf_astype.values.tolist() == tb_astype.to_pandas().values.tolist()

    assert pdf_astype.index.values.tolist() == tb.index.values.tolist() == tb_astype.index.values.tolist()

    map_of_types = {'c2': 'int32', 'c3': 'float64'}

    pdf_astype_with_dict = pdf.astype(map_of_types)

    tb_astype_with_dict = tb.astype(map_of_types)

    print(tb_astype_with_dict)
    print("-" * 80)
    print(pdf_astype_with_dict)

    assert pdf_astype_with_dict.values.tolist() == tb_astype_with_dict.to_pandas().values.tolist()

    assert tb_astype_with_dict.index.values.tolist() == tb.index.values.tolist()


def test_str_astype():
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    columns = ['c1', 'c2', 'c3']
    dataset_1 = [[1, 2, 3, 4, 5], ['20', '30', '40', '50', '51'], [33, 43, 53, 63, 73]]
    tb = Table.from_list(ctx, columns, dataset_1)
    pdf: pd.DataFrame = tb.to_pandas()
    tb.set_index('c1', drop=True)
    pdf.set_index('c1', inplace=True)

    print(tb)
    print("-" * 80)
    print(pdf)
    print("-" * 80)

    pdf_astype = pdf.astype(float)

    tb_astype = tb.astype(float)

    print(tb_astype)
    print("-" * 80)
    print(pdf_astype)

    print(pdf_astype.values.tolist())
    print(tb_astype.to_pandas().values.tolist())
    assert pdf_astype.values.tolist() == tb_astype.to_pandas().values.tolist()

    assert pdf_astype.index.values.tolist() == tb.index.values.tolist() == tb_astype.index.values.tolist()

    map_of_types = {'c2': 'int32', 'c3': 'float64'}

    pdf_astype_with_dict = pdf.astype(map_of_types)

    tb_astype_with_dict = tb.astype(map_of_types)

    print(tb_astype_with_dict)
    print("-" * 80)
    print(pdf_astype_with_dict)

    assert pdf_astype_with_dict.values.tolist() == tb_astype_with_dict.to_pandas().values.tolist()

    assert tb_astype_with_dict.index.values.tolist() == tb.index.values.tolist()


def test_table_initialization_with_index():
    ctx = CylonContext(config=None, distributed=False)
    csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30)
    table_path = '/tmp/duplicate_data_0.csv'
    tb: Table = read_csv(ctx, table_path, csv_read_options)
    expected_index = [i for i in range(tb.row_count)]
    expected_index_1 = [0, 1, 2]

    print(tb)
    print(tb.index.values)

    assert expected_index == tb.index.values.tolist()

    pd_data = [[1, 2, 3], [4, 5, 6], [6, 7, 8]]
    cols = ['a', 'b', 'c']
    dict_data = {'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [6, 7, 8]}
    pdf = pd.DataFrame(pd_data, columns=cols)
    print(pdf)

    tb_from_pd = Table.from_pandas(ctx, pdf)
    print(tb_from_pd)

    assert tb_from_pd.index.values.tolist() == pdf.index.values.tolist()

    tb_from_list = Table.from_list(ctx, cols, pd_data)

    print(tb_from_list)
    print(tb_from_list.index.values)

    assert expected_index_1 == tb_from_list.index.values.tolist()

    tb_from_dict = Table.from_pydict(ctx, dict_data)
    print(tb_from_dict)
    print(tb_from_dict.index.values)

    assert expected_index_1 == tb_from_dict.index.values.tolist()


def test_getitem_with_index():
    ctx = CylonContext(config=None, distributed=False)
    csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30)
    table_path = '/tmp/duplicate_data_0.csv'
    tb: Table = read_csv(ctx, table_path, csv_read_options)
    pdf: pd.DataFrame = tb.to_pandas()

    print(tb)
    print("-" * 80)
    print(pdf)

    tb.set_index('a', drop=True)
    pdf.set_index('a', drop=True, inplace=True)

    assert tb.index.values.tolist() == pdf.index.values.tolist()

    tb_1 = tb['b']
    pdf_1 = pdf['b']

    print(tb_1.index.values)
    print(pdf_1.index.values)

    assert tb_1.index.values.tolist() == pdf_1.index.values.tolist()

    tb_2 = tb[0:10]
    pdf_2 = pdf[0:10]

    print(tb_2.index.values)
    print(pdf_2.index.values)

    assert tb_2.index.values.tolist() == pdf_2.index.values.tolist()

    tb_3 = tb[['c', 'd']]
    pdf_3 = pdf[['c', 'd']]

    print(tb_3.index.values)
    print(pdf_3.index.values)

    assert tb_3.index.values.tolist() == pdf_3.index.values.tolist()


def test_setitem_with_index():
    ctx = CylonContext(config=None, distributed=False)
    csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30)
    table_path = '/tmp/duplicate_data_0.csv'
    tb: Table = read_csv(ctx, table_path, csv_read_options)
    pdf: pd.DataFrame = tb.to_pandas()

    print(tb)
    print("-" * 80)
    print(pdf)

    tb.set_index('a', drop=True)
    pdf.set_index('a', drop=True, inplace=True)

    new_data = [i * 10 for i in range(tb.row_count)]
    new_tb = Table.from_list(ctx, ['new_col'], [new_data])
    tb['e'] = new_tb
    pdf['e'] = pd.DataFrame(new_data)

    print(tb.index.values)
    print(pdf.index.values)

    assert tb.index.values.tolist() == pdf.index.values.tolist()


def test_isin_with_index():
    ctx = CylonContext(config=None, distributed=False)
    csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30)
    table_path = '/tmp/duplicate_data_0.csv'
    tb1: Table = read_csv(ctx, table_path, csv_read_options)
    pdf: pd.DataFrame = tb1.to_pandas()
    filter_isin = [11, 20, 11, 23]
    print(tb1)

    print(pdf)

    tb1.set_index('a', drop=True)
    pdf.set_index('a', inplace=True)

    filter_pdf: pd.DataFrame = pdf[['b', 'c']].iloc[0:5]

    tb_res = tb1[tb1['b'].isin(filter_isin)]
    pdf_res = pdf[pdf['b'].isin(filter_isin)]

    print(tb_res)
    print(pdf_res)

    assert tb_res.to_pandas().values.tolist() == pdf_res.values.tolist()

    print(tb_res.index.values)
    print(pdf_res.index.values)

    assert tb_res.index.values.tolist() == pdf_res.index.values.tolist()


def test_dropna_with_index():
    columns = ['index', 'col1', 'col2', 'col3']
    dtype = 'int32'
    index = ['a', 'b', 'c', 'd', 'e', 'f']
    datum_1 = [index, [1.0, 2.0, 3.0, 4.0, 5.0, None], [None, 7.0, 8.0, 9.0, 10.0, 11.0], [12.0,
                                                                                           13.0,
                                                                                           14.0, 15.0,
                                                                                           16.0, 17.0]]
    datum_2 = [index, [1.0, 2.0, 3.0, 4.0, 5.0, None], [None, 7.0, 8.0, 9.0, 10.0, None],
               [12.0, 13.0, None, 15.0,
                16.0, 17.0]]

    dataset = [datum_1, datum_2]
    ctx: CylonContext = CylonContext(config=None, distributed=False)

    ## axis=0 => column-wise
    inplace_ops = [True, False]
    hows = ['any', 'all']
    axiz = [0, 1]
    for inplace in inplace_ops:
        for how in hows:
            for axis in axiz:
                for data in dataset:
                    cn_tb = cn.Table.from_list(ctx, columns, data)
                    df = cn_tb.to_pandas()
                    cn_tb.set_index('index', drop=True)
                    df.set_index('index', drop=True, inplace=True)
                    if inplace:
                        cn_tb.dropna(axis=axis, how=how, inplace=inplace)
                        df.dropna(axis=1 - axis, how=how, inplace=inplace)
                    else:
                        cn_tb = cn_tb.dropna(axis=axis, how=how, inplace=inplace)
                        df = df.dropna(axis=1 - axis, how=how, inplace=inplace)

                    pdf_values = df.fillna(0).values.flatten().tolist()
                    cn_tb_values = cn_tb.to_pandas().fillna(0).values.flatten().tolist()
                    assert pdf_values == cn_tb_values
                    assert cn_tb.index.values.tolist() == df.index.values.tolist()


test_math_ops_for_table_values()
