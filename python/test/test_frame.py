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
