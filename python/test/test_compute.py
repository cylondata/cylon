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



