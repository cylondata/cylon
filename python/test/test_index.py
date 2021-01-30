import pandas as pd
from pycylon.index import Index, RangeIndex, NumericIndex, CategoricalIndex, ColumnIndex, \
    range_calculator
from pycylon import Table
from pycylon import CylonContext
import pyarrow as pa
import numpy as np


def test_with_pandas():
    pdf = pd.DataFrame([[1, 2, 3, 4, 5, 'a'], [6, 7, 8, 9, 10, 'b'], [11, 12, 13, 14, 15, 'c'],
                        [16, 17, 18, 19, 20, 'a'], [16, 17, 18, 19, 20, 'd'],
                        [111, 112, 113, 114, 5,
                         'a']])

    # print(pdf)
    pdf1 = pdf.set_index([1, 2])
    # print(pdf1)
    print(pdf1.index)


def test_numeric_index():
    rg = range(0, 10, 1)
    rg1 = range(0, 10, 2)
    r = NumericIndex(data=rg)

    assert r.index_values == rg
    assert r.index_values != rg1


def test_range_index():
    rg = range(0, 10, 1)
    rg1 = range(0, 10, 2)
    r = RangeIndex(start=rg.start, stop=rg.stop, step=rg.step)

    assert r.index_values == rg
    assert r.index_values != rg1

    r1 = RangeIndex(rg)
    r2 = RangeIndex(rg)

    assert r1.index_values == rg
    assert r2.index_values != rg1


def calculate_range_size_manual(rg: range):
    sum = 0
    for i in rg:
        sum += 1
    return sum


def test_range_count():
    rg_1 = range(0, 10)
    rg_2 = range(0, 10, 2)
    rg_3 = range(0, 10, 3)
    rg_4 = range(0, 11, 2)
    rg_5 = range(0, 14, 3)
    rgs = [rg_1, rg_2, rg_3, rg_4, rg_5]
    for rg in rgs:
        assert range_calculator(rg) == calculate_range_size_manual(rg)


def test_set_index():
    pdf = pd.DataFrame([[1, 2, 3, 4, 5, 'a'], [6, 7, 8, 9, 10, 'b'], [11, 12, 13, 14, 15, 'c'],
                        [16, 17, 18, 19, 20, 'a'], [16, 17, 18, 19, 20, 'd'],
                        [111, 112, 113, 114, 5,
                         'a']])
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    cn_tb: Table = Table.from_pandas(ctx, pdf)
    # set index by row indices
    row_indices = ['a', 'b', 'c', 'd', 'e', 'f']
    cn_tb.set_index(row_indices)

    assert cn_tb.row_count == len(row_indices)
    assert isinstance(cn_tb.index, CategoricalIndex)
    assert cn_tb.index.index_values == row_indices

    print(cn_tb.column_names)
    # set index column name
    col_name = "0"
    cn_tb.set_index(col_name)

    col_index = cn_tb.index
    assert isinstance(col_index, ColumnIndex)
    col_index_values = col_index.index_values

    ar_0 = pa.chunked_array([pdf[int(col_name)].values])
    ar_1 = pa.chunked_array([pdf[1].values])
    assert ar_0 == col_index_values
    assert ar_1 != col_index_values

    # set index by column names

    col_names = ["0", "1"]
    ars = []
    for col_name in col_names:
        ars.append(pa.chunked_array([pdf[int(col_name)]]))
    cn_tb.set_index(col_names)

    col_index = cn_tb.index
    assert isinstance(col_index, ColumnIndex)
    col_index_values = col_index.index_values

    for col_index_value, ar in zip(col_index_values, ars):
        assert col_index_value == ar


def test_loc():
    df = pd.DataFrame([[1, 2], [4, 5], [7, 8]], index=['cobra', 'viper', 'sidewinder'],
                      columns=['max_speed', 'shield'])

    print(df)
    ld = df.loc['viper']

    print(type(ld))


def test_cylon_cpp_single_column_indexing():
    from pycylon.indexing.index import IndexingSchema
    from pycylon.indexing.index_utils import IndexUtil
    from pycylon.indexing.index import LocIndexer

    pdf_float = pd.DataFrame({'a': pd.Series([1, 4, 7, 10, 20, 23, 10], dtype=np.float64()),
                              'b': pd.Series([2, 5, 8, 11, 22, 25, 12], dtype='int')})
    pdf = pd.DataFrame([[1, 2], [4, 5], [7, 8], [10, 11], [20, 22], [23, 25], [10, 12]])
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    cn_tb: Table = Table.from_pandas(ctx, pdf)
    indexing_schema = IndexingSchema.LINEAR

    print("Input Table")
    print(cn_tb)
    print(cn_tb.to_arrow())

    output = IndexUtil.build_index(indexing_schema, cn_tb, 0, True)
    print("Output Indexed Table")
    print(output)

    loc_ix = LocIndexer(indexing_schema)
    start_index = 1
    end_index = 7
    column_index = 0

    loc_out = loc_ix.loc_with_single_column(slice(start_index, end_index), column_index, output)
    #
    print(loc_out)

    print(loc_out.to_arrow())

    index = loc_out.get_index()

    print(index)

    print(index.get_index_array())

    indices = [4, 7, 23, 20]

    loc_out2 = loc_ix.loc_with_single_column(indices, column_index, output)

    print(loc_out2)

    loc_index = 10
    loc_out3 = loc_ix.loc_with_single_column(loc_index, column_index, output)

    print(loc_out3)


def test_cylon_cpp_multi_column_indexing():
    from pycylon.indexing.index import IndexingSchema
    from pycylon.indexing.index_utils import IndexUtil
    from pycylon.indexing.index import LocIndexer

    pdf_float = pd.DataFrame({'a': pd.Series([1, 4, 7, 10, 20, 23, 10], dtype=np.float64()),
                              'b': pd.Series([2, 5, 8, 11, 22, 25, 12], dtype='int'),
                              'c': pd.Series([11, 12, 14, 15, 16, 17, 18], dtype='int')
                              })
    pdf = pd.DataFrame([[1, 2, 11], [4, 5, 12], [7, 8, 14], [10, 11, 15], [20, 22, 16], [23, 25,
                                                                                         17],
                        [10, 12, 18]])
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    cn_tb: Table = Table.from_pandas(ctx, pdf)
    indexing_schema = IndexingSchema.LINEAR

    print("Input Table")
    print(cn_tb)
    print(cn_tb.to_arrow())

    output = IndexUtil.build_index(indexing_schema, cn_tb, 0, True)
    print("Output Indexed Table")
    print(output)

    loc_ix = LocIndexer(indexing_schema)
    start_index = 1
    end_index = 7
    column_index = [0, 1]

    loc_out = loc_ix.loc_with_multi_column(slice(start_index, end_index), column_index, output)
    #
    print(loc_out)

    print(loc_out.to_arrow())

    index = loc_out.get_index()

    print(index)

    print(index.get_index_array())

    indices = [4, 7, 23, 20]

    loc_out2 = loc_ix.loc_with_multi_column(indices, column_index, output)

    print(loc_out2)

    loc_index = 10
    loc_out3 = loc_ix.loc_with_multi_column(loc_index, column_index, output)

    print(loc_out3)


def test_cylon_cpp_str_single_column_indexing():
    from pycylon.indexing.index import IndexingSchema
    from pycylon.indexing.index_utils import IndexUtil
    from pycylon.indexing.index import LocIndexer

    pdf_str = pd.DataFrame([["1", 2], ["4", 5], ["7", 8], ["10", 11], ["20", 22], ["23", 25], ["10",
                                                                                               12]])
    pdf_float = pd.DataFrame({'a': pd.Series([1, 4, 7, 10, 20, 23, 10], dtype=np.float64()),
                              'b': pd.Series([2, 5, 8, 11, 22, 25, 12], dtype='int')})
    pdf = pd.DataFrame([[1, 2], [4, 5], [7, 8], [10, 11], [20, 22], [23, 25], [10, 12]])
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    cn_tb: Table = Table.from_pandas(ctx, pdf_str)
    indexing_schema = IndexingSchema.LINEAR

    print("Input Table")
    print(cn_tb)
    print(cn_tb.to_arrow())

    output = IndexUtil.build_index(indexing_schema, cn_tb, 0, True)
    print("Output Indexed Table")
    print(output)

    loc_ix = LocIndexer(indexing_schema)
    start_index = b"1"
    end_index = b"7"
    column_index = 0

    loc_out = loc_ix.loc_with_single_column(slice(start_index, end_index), column_index, output)

    print(loc_out)

    print(loc_out.to_arrow())

    index = loc_out.get_index()

    print(index)

    print(index.get_index_array())

    indices = ["100", "4", "7", "23", "20"]

    indices = [b'4']

    # indices = [4, 7]

    loc_out2 = loc_ix.loc_with_single_column(indices, column_index, output)

    print(loc_out2)

    loc_index = b'10'
    loc_out3 = loc_ix.loc_with_single_column(loc_index, column_index, output)

    print(loc_out3)


def test_cylon_cpp_str_multi_column_indexing():
    from pycylon.indexing.index import IndexingSchema
    from pycylon.indexing.index_utils import IndexUtil
    from pycylon.indexing.index import LocIndexer

    pdf_str = pd.DataFrame([["1", 2, 3], ["4", 5, 4], ["7", 8, 10], ["10", 11, 12], ["20", 22, 20],
                            ["23", 25, 20], ["10", 12, 35]])
    pdf_float = pd.DataFrame({'a': pd.Series([1, 4, 7, 10, 20, 23, 10], dtype=np.float64()),
                              'b': pd.Series([2, 5, 8, 11, 22, 25, 12], dtype='int'),
                              'c': pd.Series([3, 4, 10, 12, 20, 20, 35], dtype='int')})
    pdf = pd.DataFrame([[1, 2], [4, 5], [7, 8], [10, 11], [20, 22], [23, 25], [10, 12]])
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    cn_tb: Table = Table.from_pandas(ctx, pdf_str)
    indexing_schema = IndexingSchema.LINEAR

    print("Input Table")
    print(cn_tb)
    print(cn_tb.to_arrow())

    output = IndexUtil.build_index(indexing_schema, cn_tb, 0, True)
    print("Output Indexed Table")
    print(output)

    loc_ix = LocIndexer(indexing_schema)
    start_index = b"1"
    end_index = b"7"
    column_index = [0, 1]

    loc_out = loc_ix.loc_with_multi_column(slice(start_index, end_index), column_index, output)

    print(loc_out)

    print(loc_out.to_arrow())

    index = loc_out.get_index()

    print(index)

    print(index.get_index_array())

    indices = ["100", "4", "7", "23", "20"]

    indices = [b'4']

    # indices = [4, 7]

    loc_out2 = loc_ix.loc_with_multi_column(indices, column_index, output)

    print(loc_out2)

    loc_index = b'10'
    loc_out3 = loc_ix.loc_with_multi_column(loc_index, column_index, output)

    print(loc_out3)


# test_cylon_cpp_single_column_indexing()
# test_cylon_cpp_str_single_column_indexing()

#test_cylon_cpp_multi_column_indexing()
#test_cylon_cpp_str_multi_column_indexing()
