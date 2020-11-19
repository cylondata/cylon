import pyarrow as pa
import pandas as pd
import pycylon as cn
from pycylon import CylonContext


def test_isin():
    df = pd.DataFrame({'num_legs': [2, 4], 'num_wings': [2, 0]}, index=['falcon', 'dog'])
    arw_tb = pa.Table.from_pandas(df)
    arw_ar: pa.array = pa.array([[2, 4], [2, 0]])
    print(df)


def test_isna():
    columns = ['col1', 'col2']
    data = [[1, 2, 3, 4, 5, None], [None, 7, 8, 9, 10, 11]]
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    cn_tb = cn.Table.from_list(ctx, columns, data)
    df = cn_tb.to_pandas()

    assert df.isna().values.tolist() == cn_tb.isna().to_pandas().values.tolist()


def test_isnull():
    columns = ['col1', 'col2']
    data = [[1, 2, 3, 4, 5, None], [None, 7, 8, 9, 10, 11]]
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    cn_tb = cn.Table.from_list(ctx, columns, data)
    df = cn_tb.to_pandas()

    assert df.isnull().values.tolist() == cn_tb.isnull().to_pandas().values.tolist()


def test_notna():
    columns = ['col1', 'col2']
    data = [[1, 2, 3, 4, 5, None], [None, 7, 8, 9, 10, 11]]
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    cn_tb = cn.Table.from_list(ctx, columns, data)
    df = cn_tb.to_pandas()

    assert df.notna().values.tolist() == cn_tb.notna().to_pandas().values.tolist()


def test_notnull():
    columns = ['col1', 'col2']
    data = [[1, 2, 3, 4, 5, None], [None, 7, 8, 9, 10, 11]]
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    cn_tb = cn.Table.from_list(ctx, columns, data)
    df = cn_tb.to_pandas()

    assert df.notnull().values.tolist() == cn_tb.notnull().to_pandas().values.tolist()


def test_dropna():
    import numpy as np
    columns = ['col1', 'col2', 'col3']
    dtype = 'int32'
    datum_1 = [[1.0, 2.0, 3.0, 4.0, 5.0, None], [None, 7.0, 8.0, 9.0, 10.0, 11.0], [12.0, 13.0,
                                                                                    14.0, 15.0,
                                                                                    16.0, 17.0]]
    datum_2 = [[1.0, 2.0, 3.0, 4.0, 5.0, None], [None, 7.0, 8.0, 9.0, 10.0, None],
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
                    if inplace:
                        cn_tb.dropna(axis=axis, how=how, inplace=inplace)
                        df.dropna(axis=1 - axis, how=how, inplace=inplace)
                    else:
                        cn_tb = cn_tb.dropna(axis=axis, how=how, inplace=inplace)
                        df = df.dropna(axis=1 - axis, how=how, inplace=inplace)

                    pdf_values = df.fillna(0).values.flatten().tolist()
                    cn_tb_values = cn_tb.to_pandas().fillna(0).values.flatten().tolist()
                    assert pdf_values == cn_tb_values
