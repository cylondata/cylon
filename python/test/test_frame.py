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


import os
import numpy as np
import pandas as pd
import pycylon as cn
import pyarrow as pa
from pycylon import Series
from pycylon.frame import DataFrame
from pycylon import Table
from pycylon.io import CSVReadOptions
from pycylon.io import read_csv
from pycylon import CylonContext
import operator


def test_initialization_1():
    d1 = [[1, 2, 3], [4, 5, 6]]
    d2 = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    d3 = {'0': [1, 2, 3], '1': [4, 5, 6]}
    d4 = pd.DataFrame(d3)
    d5 = pa.Table.from_pydict(d3)

    cdf1 = DataFrame(d1)
    cdf2 = DataFrame(d2)
    cdf3 = DataFrame(d3)
    cdf4 = DataFrame(d4)
    cdf5 = DataFrame(d5)

    assert cdf1.shape == cdf2.shape == cdf3.shape == cdf4.shape == cdf5.shape


def test_get_set_item():
    d1 = [[1, 2, 3], [4, 5, 6]]
    cdf1 = DataFrame(d1)
    print(cdf1)

    print(cdf1.columns)

    c1 = cdf1['0']
    print(c1.shape)
    d1 = DataFrame([[10, 20, 30]])

    print(d1.shape)
    print(cdf1)
    cdf1['0'] = d1

    print(cdf1)


def test_filter():
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    table1_path = '/tmp/user_usage_tm_1.csv'
    table2_path = '/tmp/user_usage_tm_2.csv'

    assert os.path.exists(table1_path) and os.path.exists(table2_path)

    csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30)

    tb: Table = read_csv(ctx, table1_path, csv_read_options)
    df: DataFrame = DataFrame(tb)

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
        tb_cond_1 = comp_op[0](df[column_name], limit[0])
        tb_cond_2 = comp_op[1](df[column_name], limit[1])
        tb_cond_3 = comp_op[2](df[column_name], limit[2])

        res_1_op = op(tb_cond_1, tb_cond_2)
        res_2_op = op(res_1_op, tb_cond_3)

        res_1 = df[res_1_op]
        res_2 = df[res_2_op]

        column_pdf_1 = res_1[column_name].to_pandas()
        column_pdf_2 = res_2[column_name].to_pandas()

        column_1 = column_pdf_1[column_name]
        for col in column_1:
            assert op(comp_op[0](col, limit[0]), comp_op[1](col, limit[1]))

        column_2 = column_pdf_2[column_name]
        for col in column_2:
            assert op(op(comp_op[0](col, limit[0]), comp_op[1](col, limit[1])),
                      comp_op[2](col, limit[2]))


def test_invert():
    # Bool Invert Test

    data_list = [[False, True, False, True, True], [False, True, False, True, True]]
    pdf = pd.DataFrame(data_list)
    cdf = DataFrame(pdf)

    invert_cdf = ~cdf
    invert_pdf = ~pdf

    assert invert_cdf.to_pandas().values.tolist() == invert_pdf.values.tolist()


def test_neg():
    npr = np.array([[1, 2, 3, 4, 5, -6, -7], [-1, -2, -3, -4, -5, 6, 7]])
    pdf = pd.DataFrame(npr)
    cdf = DataFrame(pdf)
    neg_cdf = -cdf
    neg_pdf = -pdf
    assert neg_cdf.to_pandas().values.tolist() == neg_pdf.values.tolist()


def test_setitem():
    npr = np.array([[1, 2, 3, 4, 5], [-1, -2, -3, -4, -5]])
    pdf = pd.DataFrame(npr)

    cdf = DataFrame(pdf)
    # replacing an existing column
    cdf['0'] = cdf['4']
    assert cdf['0'].to_pandas().values.tolist() == cdf['4'].to_pandas().values.tolist()
    # adding a new column at the end
    cdf['5'] = cdf['4']
    assert cdf['5'].to_pandas().values.tolist() == cdf['4'].to_pandas().values.tolist()


def test_math_ops_for_scalar():
    npr = np.array([[20, 2, 3, 4, 5], [10, -20, -30, -40, -50], [10.2, 13.2, 16.4, 12.2, 10.8]])
    pdf = pd.DataFrame(npr)
    cdf = DataFrame(pdf)

    from operator import add, sub, mul, truediv
    ops = [add, sub, mul, truediv]

    for op in ops:
        cdf_1 = cdf
        pdf_1 = pdf
        # test column division
        cdf_1['0'] = op(cdf_1['0'], 2)
        pdf_1[0] = op(pdf_1[0], 2)

        assert pdf_1.values.tolist() == cdf_1.to_pandas().values.tolist()

        # test table division
        cdf_2 = cdf
        pdf_2 = pdf

        cdf_2 = op(cdf_2, 2)
        pdf_2 = op(pdf, 2)

        assert pdf_2.values.tolist() == cdf_2.to_pandas().values.tolist()


def test_i_bitwise_ops():
    # TODO: Improve test and functionality: https://github.com/cylondata/cylon/issues/229
    npr = np.array([[20, 2, 3, 4, 5], [10, -20, -30, -40, -50], [36.2, 13.2, 16.4, 12.2, 10.8]])
    pdf = pd.DataFrame(npr)
    cdf = DataFrame(pdf)

    a = cdf['0'] > 10
    b = cdf['1'] > 2
    a_pdf = pdf[0] > 10
    b_pdf = pdf[1] > 2

    d = a & b
    a &= b
    d_pdf = a_pdf & b_pdf
    a_pdf &= b_pdf

    assert d.to_pandas().values.tolist() == a.to_pandas().values.tolist()
    assert a.to_pandas().values.flatten().tolist() == a_pdf.values.tolist()

    ## OR

    a = cdf['0'] > 10
    b = cdf['1'] > 2
    a_pdf = pdf[0] > 10
    b_pdf = pdf[1] > 2

    d = a | b
    a |= b
    d_pdf = a_pdf | b_pdf
    a_pdf |= b_pdf

    assert d.to_pandas().values.tolist() == a.to_pandas().values.tolist()
    assert a.to_pandas().values.flatten().tolist() == a_pdf.values.tolist()


def test_math_i_ops_for_scalar():
    npr = np.array([[20, 2, 3, 4, 5], [10, -20, -30, -40, -50], [12.2, 13.2, 16.4, 12.2, 10.8]])
    pdf = pd.DataFrame(npr)
    cdf = DataFrame(pdf)

    cdf_1 = cdf
    pdf_1 = pdf
    # test column addition

    cdf_1['0'] += 2
    pdf_1[0] += 2

    assert pdf_1.values.tolist() == cdf_1.to_pandas().values.tolist()

    cdf_1['0'] -= 2
    pdf_1[0] -= 2

    assert pdf_1.values.tolist() == cdf_1.to_pandas().values.tolist()

    cdf_1['0'] *= 2
    pdf_1[0] *= 2

    assert pdf_1.values.tolist() == cdf_1.to_pandas().values.tolist()

    cdf_1['0'] /= 2
    pdf_1[0] /= 2

    assert pdf_1.values.tolist() == cdf_1.to_pandas().values.tolist()

    # test table division
    cdf_2 = cdf_1
    pdf_2 = pdf

    cdf_2 += 2
    pdf += 2

    assert pdf_2.values.tolist() == cdf_2.to_pandas().values.tolist()

    cdf_2 -= 2
    pdf -= 2

    assert pdf_2.values.tolist() == cdf_2.to_pandas().values.tolist()

    cdf_2 *= 2
    pdf *= 2

    assert pdf_2.values.tolist() == cdf_2.to_pandas().values.tolist()

    cdf_2 /= 2
    pdf /= 2

    assert pdf_2.values.tolist() == cdf_2.to_pandas().values.tolist()


def test_drop():
    ctx: CylonContext = CylonContext(config=None, distributed=False)

    table1_path = '/tmp/user_usage_tm_1.csv'

    assert os.path.exists(table1_path)

    csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30)

    tb: Table = read_csv(ctx, table1_path, csv_read_options)
    cdf = DataFrame(tb)

    drop_column = 'outgoing_sms_per_month'

    cdf_new = cdf.drop([drop_column])

    assert not cdf_new.columns.__contains__(drop_column)


def test_fillna():
    data_list_numeric = [[1, 2, None, 4, 5], [6, 7, 8, 9, None]]
    fill_value = 0

    cdf_numeric = DataFrame(data_list_numeric)

    cn_tb_numeric_fillna = cdf_numeric.fillna(fill_value)

    data_list = list(cn_tb_numeric_fillna.to_dict().values())
    for col in data_list:
        assert not col.__contains__(None)
        assert col.__contains__(fill_value)


def test_notna():
    data = [[1, 2, 3, 4, 5, None], [None, 7, 8, 9, 10, 11]]
    cdf = DataFrame(data)
    df = cdf.to_pandas()

    assert df.notna().values.tolist() == cdf.notna().to_pandas().values.tolist()


def test_notnull():
    data = [[1, 2, 3, 4, 5, None], [None, 7, 8, 9, 10, 11]]
    cdf = DataFrame(data)
    df = cdf.to_pandas()

    assert df.notnull().values.tolist() == cdf.notnull().to_pandas().values.tolist()


def test_isin():
    pdf = pd.DataFrame({'num_legs': [2, 4], 'num_wings': [2, 0]})
    cdf = DataFrame(pdf)

    arr = [0, 2]
    assert (pdf.isin(arr).values.tolist() == cdf.isin(arr).to_pandas().values.tolist())


def test_isna():
    data = [[1, 2, 3, 4, 5, None], [None, 7, 8, 9, 10, 11]]
    cdf = DataFrame(data)
    df = cdf.to_pandas()

    assert df.isna().values.tolist() == cdf.isna().to_pandas().values.tolist()


def test_isnull():
    data = [[1, 2, 3, 4, 5, None], [None, 7, 8, 9, 10, 11]]
    cdf = DataFrame(data)
    df = cdf.to_pandas()

    assert df.isnull().values.tolist() == cdf.isnull().to_pandas().values.tolist()


def test_rename():
    col_names = ['col1', 'col2', 'col3', 'col4']
    data_list_numeric = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15],
                         [16, 17, 18, 19, 20]]
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    index_values = [0, 1, 2, 3, 4]
    cn_tb = cn.Table.from_list(ctx, col_names, data_list_numeric)
    cn_tb.set_index(index_values)
    cdf = DataFrame(cn_tb)
    prev_col_names = cn_tb.column_names
    # with dictionary
    columns = {'col1': 'col-1', 'col3': 'col-3'}
    cdf.rename(columns)

    new_col_names = cdf.columns

    for key in columns:
        value = columns[key]
        assert prev_col_names.index(key) == new_col_names.index(value)

    # with list
    cn_tb_list = cn.Table.from_list(ctx, col_names, data_list_numeric)
    cn_tb_list.set_index(index_values)
    cdf_list = DataFrame(cn_tb_list)
    prev_col_names = cdf_list.columns
    new_column_names = ['col-1', 'col-2', 'col-3', 'col-4']
    cdf_list.rename(new_column_names)

    assert cdf_list.columns == new_column_names
