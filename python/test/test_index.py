import pandas as pd
from pycylon.index import Index, RangeIndex, NumericIndex, CategoricalIndex, ColumnIndex, \
    range_calculator
from pycylon import Table
from pycylon import CylonContext
import pyarrow as pa
import numpy as np
from pycylon.io import CSVReadOptions
from pycylon.io import read_csv


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


def test_cylon_set_index_from_column():
    from pycylon.indexing.index import IndexingSchema
    from pycylon.indexing.index_utils import IndexUtil
    from pycylon.indexing.index import LocIndexer

    pdf_float = pd.DataFrame({'a': pd.Series([1, 4, 7, 10, 20, 23, 10], dtype=np.int64()),
                              'b': pd.Series([2, 5, 8, 11, 22, 25, 12], dtype='int')})
    pdf = pd.DataFrame([[1, 2], [4, 5], [7, 8], [10, 11], [20, 22], [23, 25], [10, 12]])
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    cn_tb: Table = Table.from_pandas(ctx, pdf_float)
    indexing_schema = IndexingSchema.LINEAR
    drop_index = True

    print("Before Indexing")
    print(cn_tb)

    cn_tb.set_index('a', indexing_schema, drop_index)

    print("After Indexing")
    assert cn_tb.column_names == ['b']

    assert cn_tb.get_index().get_schema() == IndexingSchema.LINEAR


def test_reset_index():
    from pycylon.indexing.index import IndexingSchema
    from pycylon.indexing.index_utils import IndexUtil
    from pycylon.indexing.index import LocIndexer

    pdf_float = pd.DataFrame({'a': pd.Series([1, 4, 7, 10, 20, 23, 10], dtype=np.int64()),
                              'b': pd.Series([2, 5, 8, 11, 22, 25, 12], dtype='int')})
    pdf = pd.DataFrame([[1, 2], [4, 5], [7, 8], [10, 11], [20, 22], [23, 25], [10, 12]])
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    cn_tb: Table = Table.from_pandas(ctx, pdf_float)
    indexing_schema = IndexingSchema.LINEAR
    drop_index = True

    cn_tb.set_index('a', indexing_schema, drop_index)

    assert cn_tb.get_index().get_schema() == IndexingSchema.LINEAR

    rest_drop_index = False
    cn_tb.reset_index(rest_drop_index)

    assert cn_tb.column_names == ['index', 'b']

    assert cn_tb.get_index().get_schema() == IndexingSchema.RANGE


def test_cylon_cpp_single_column_indexing():
    from pycylon.indexing.index import IndexingSchema
    from pycylon.indexing.index_utils import IndexUtil
    from pycylon.indexing.index import LocIndexer

    pdf_float = pd.DataFrame({'a': pd.Series([1, 4, 7, 10, 20, 23, 10], dtype=np.int64()),
                              'b': pd.Series([2, 5, 8, 11, 22, 25, 12], dtype='int')})
    pdf = pd.DataFrame([[1, 2], [4, 5], [7, 8], [10, 11], [20, 22], [23, 25], [10, 12]])
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    cn_tb: Table = Table.from_pandas(ctx, pdf_float)
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
    start_index = "1"
    end_index = "7"
    column_index = 0

    loc_out = loc_ix.loc_with_single_column(slice(start_index, end_index), column_index, output)

    print(loc_out)

    print(loc_out.to_arrow())

    index = loc_out.get_index()

    print(index)

    print(index.get_index_array())

    indices = ["100", "4", "7", "23", "20"]

    indices = ['4']

    # indices = [4, 7]

    loc_out2 = loc_ix.loc_with_single_column(indices, column_index, output)

    print(loc_out2)

    loc_index = '10'
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
    start_index = "1"
    end_index = "7"
    column_index = [0, 1]

    loc_out = loc_ix.loc_with_multi_column(slice(start_index, end_index), column_index, output)

    print(loc_out)

    print(loc_out.to_arrow())

    index = loc_out.get_index()

    print(index)

    print(index.get_index_array())

    indices = ["100", "4", "7", "23", "20"]

    indices = ['4']

    # indices = [4, 7]

    loc_out2 = loc_ix.loc_with_multi_column(indices, column_index, output)

    print(loc_out2)

    loc_index = '10'
    loc_out3 = loc_ix.loc_with_multi_column(loc_index, column_index, output)

    print(loc_out3)


def test_cylon_cpp_range_column_indexing():
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
    column_index = slice(0, 1)

    loc_out = loc_ix.loc_with_range_column(slice(start_index, end_index), column_index, output)
    #
    print(loc_out)

    print(loc_out.to_arrow())

    index = loc_out.get_index()

    print(index)

    print(index.get_index_array())

    indices = [4, 7, 23, 20]

    loc_out2 = loc_ix.loc_with_range_column(indices, column_index, output)

    print(loc_out2)

    loc_index = 10
    loc_out3 = loc_ix.loc_with_range_column(loc_index, column_index, output)

    print(loc_out3)


def test_cylon_cpp_str_range_column_indexing():
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
    start_index = "1"
    end_index = "7"
    column_index = slice(0, 1)

    loc_out = loc_ix.loc_with_range_column(slice(start_index, end_index), column_index, output)

    print(loc_out)

    print(loc_out.to_arrow())

    index = loc_out.get_index()

    print(index)

    print(index.get_index_array())

    indices = ["100", "4", "7", "23", "20"]

    indices = ['4']

    # indices = [4, 7]

    loc_out2 = loc_ix.loc_with_range_column(indices, column_index, output)

    print(loc_out2)

    loc_index = '10'
    loc_out3 = loc_ix.loc_with_range_column(loc_index, column_index, output)

    print(loc_out3)


def test_loc_op_mode_1():
    from pycylon.indexing.index import IndexingSchema
    from pycylon.indexing.index_utils import IndexUtil
    from pycylon.indexing.index import LocIndexer

    pdf_float = pd.DataFrame({'a': pd.Series([1, 4, 7, 10, 20, 23, 11], dtype=np.int64()),
                              'b': pd.Series([2, 5, 8, 11, 22, 25, 12], dtype='int'),
                              'c': pd.Series([12, 15, 18, 111, 122, 125, 112], dtype='int'),
                              'd': pd.Series([212, 215, 218, 211, 222, 225, 312], dtype='int'),
                              'e': pd.Series([1121, 12151, 12181, 12111, 12221, 12251, 13121],
                                             dtype='int')})
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    cn_tb: Table = Table.from_pandas(ctx, pdf_float)
    indexing_schema = IndexingSchema.LINEAR
    drop_index = True

    print("Before Indexing")
    print(cn_tb)

    cn_tb.set_index('a', indexing_schema, drop_index)

    pdf_float = pdf_float.set_index('a')

    print("After Indexing")
    assert cn_tb.column_names == ['b', 'c', 'd', 'e']

    assert cn_tb.get_index().get_schema() == IndexingSchema.LINEAR

    loc_cn_1 = cn_tb.loc[7:20, 'c':'e']
    loc_pd_1 = pdf_float.loc[7:20, 'c':'e']

    assert loc_pd_1.values.tolist() == loc_cn_1.to_pandas().values.tolist()
    assert loc_cn_1.get_index().get_index_array() == pa.array(loc_pd_1.index)

    loc_cn_2 = cn_tb.loc[7:20, 'd':]
    loc_pd_2 = pdf_float.loc[7:20, 'd':]

    assert loc_pd_2.values.tolist() == loc_cn_2.to_pandas().values.tolist()
    assert loc_cn_2.get_index().get_index_array() == pa.array(loc_pd_2.index)

    loc_cn_3 = cn_tb.loc[7:, 'd':]
    loc_pd_3 = pdf_float.loc[7:, 'd':]

    assert loc_pd_3.values.tolist() == loc_cn_3.to_pandas().values.tolist()
    assert loc_cn_3.get_index().get_index_array() == pa.array(loc_pd_3.index)

    loc_cn_4 = cn_tb.loc[:7, 'd':]
    loc_pd_4 = pdf_float.loc[:7, 'd':]

    assert loc_pd_4.values.tolist() == loc_cn_4.to_pandas().values.tolist()
    assert loc_cn_4.get_index().get_index_array() == pa.array(loc_pd_4.index)

    loc_cn_5 = cn_tb.loc[:, 'd':]
    loc_pd_5 = pdf_float.loc[:, 'd':]

    assert loc_pd_5.values.tolist() == loc_cn_5.to_pandas().values.tolist()
    assert loc_cn_5.get_index().get_index_array() == pa.array(loc_pd_5.index)


def test_loc_op_mode_2():
    from pycylon.indexing.index import IndexingSchema
    from pycylon.indexing.index_utils import IndexUtil
    from pycylon.indexing.index import LocIndexer

    pdf_float = pd.DataFrame({'a': pd.Series(["1", "4", "7", "10", "20", "23", "11"]),
                              'b': pd.Series([2, 5, 8, 11, 22, 25, 12], dtype='int'),
                              'c': pd.Series([12, 15, 18, 111, 122, 125, 112], dtype='int'),
                              'd': pd.Series([212, 215, 218, 211, 222, 225, 312], dtype='int'),
                              'e': pd.Series([1121, 12151, 12181, 12111, 12221, 12251, 13121],
                                             dtype='int')})
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    cn_tb: Table = Table.from_pandas(ctx, pdf_float)
    indexing_schema = IndexingSchema.LINEAR
    drop_index = True

    print("Before Indexing")
    print(cn_tb)

    cn_tb.set_index('a', indexing_schema, drop_index)

    pdf_float = pdf_float.set_index('a')

    print("After Indexing")
    assert cn_tb.column_names == ['b', 'c', 'd', 'e']

    assert cn_tb.get_index().get_schema() == IndexingSchema.LINEAR

    loc_cn_1 = cn_tb.loc["7":"20", 'c':'e']
    loc_pd_1 = pdf_float.loc["7":"20", 'c':'e']

    assert loc_pd_1.values.tolist() == loc_cn_1.to_pandas().values.tolist()
    assert loc_cn_1.get_index().get_index_array() == pa.array(loc_pd_1.index)

    loc_cn_2 = cn_tb.loc["7":"20", 'd':]
    loc_pd_2 = pdf_float.loc["7":"20", 'd':]

    assert loc_pd_2.values.tolist() == loc_cn_2.to_pandas().values.tolist()
    assert loc_cn_2.get_index().get_index_array() == pa.array(loc_pd_2.index)

    loc_cn_3 = cn_tb.loc["7":, 'd':]
    loc_pd_3 = pdf_float.loc["7":, 'd':]

    assert loc_pd_3.values.tolist() == loc_cn_3.to_pandas().values.tolist()
    assert loc_cn_3.get_index().get_index_array() == pa.array(loc_pd_3.index)

    loc_cn_4 = cn_tb.loc[:"7", 'd':]
    loc_pd_4 = pdf_float.loc[:"7", 'd':]

    assert loc_pd_4.values.tolist() == loc_cn_4.to_pandas().values.tolist()
    assert loc_cn_4.get_index().get_index_array() == pa.array(loc_pd_4.index)

    loc_cn_5 = cn_tb.loc[:, 'd':]
    loc_pd_5 = pdf_float.loc[:, 'd':]

    assert loc_pd_5.values.tolist() == loc_cn_5.to_pandas().values.tolist()
    assert loc_cn_5.get_index().get_index_array() == pa.array(loc_pd_5.index)


def test_loc_op_mode_3():
    from pycylon.indexing.index import IndexingSchema
    from pycylon.indexing.index_utils import IndexUtil
    from pycylon.indexing.index import LocIndexer

    pdf_float = pd.DataFrame({'a': pd.Series(["1", "4", "7", "10", "20", "23", "11"]),
                              'b': pd.Series([2, 5, 8, 11, 22, 25, 12], dtype='int'),
                              'c': pd.Series([12, 15, 18, 111, 122, 125, 112], dtype='int'),
                              'd': pd.Series([212, 215, 218, 211, 222, 225, 312], dtype='int'),
                              'e': pd.Series([1121, 12151, 12181, 12111, 12221, 12251, 13121],
                                             dtype='int')})
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    cn_tb: Table = Table.from_pandas(ctx, pdf_float)
    indexing_schema = IndexingSchema.LINEAR
    drop_index = True

    print("Before Indexing")
    print(cn_tb)

    cn_tb.set_index('a', indexing_schema, drop_index)

    pdf_float = pdf_float.set_index('a')

    print("After Indexing")
    assert cn_tb.column_names == ['b', 'c', 'd', 'e']

    assert cn_tb.get_index().get_schema() == IndexingSchema.LINEAR

    loc_cn_1 = cn_tb.loc["7":"20"]
    loc_pd_1 = pdf_float.loc["7":"20"]

    assert loc_pd_1.values.tolist() == loc_cn_1.to_pandas().values.tolist()
    assert loc_cn_1.get_index().get_index_array() == pa.array(loc_pd_1.index)

    loc_cn_2 = cn_tb.loc["7":]
    loc_pd_2 = pdf_float.loc["7":]

    assert loc_pd_2.values.tolist() == loc_cn_2.to_pandas().values.tolist()
    assert loc_cn_2.get_index().get_index_array() == pa.array(loc_pd_2.index)

    loc_cn_3 = cn_tb.loc[:"7"]
    loc_pd_3 = pdf_float.loc[:"7"]

    assert loc_pd_3.values.tolist() == loc_cn_3.to_pandas().values.tolist()
    assert loc_cn_3.get_index().get_index_array() == pa.array(loc_pd_3.index)

    loc_cn_4 = cn_tb.loc[:]
    loc_pd_4 = pdf_float.loc[:]

    assert loc_pd_4.values.tolist() == loc_cn_4.to_pandas().values.tolist()
    assert loc_cn_4.get_index().get_index_array() == pa.array(loc_pd_4.index)


def test_iloc_op_mode_1():
    from pycylon.indexing.index import IndexingSchema
    from pycylon.indexing.index_utils import IndexUtil
    from pycylon.indexing.index import LocIndexer

    pdf_float = pd.DataFrame({'a': pd.Series(["1", "4", "7", "10", "20", "23", "11"]),
                              'b': pd.Series([2, 5, 8, 11, 22, 25, 12], dtype='int'),
                              'c': pd.Series([12, 15, 18, 111, 122, 125, 112], dtype='int'),
                              'd': pd.Series([212, 215, 218, 211, 222, 225, 312], dtype='int'),
                              'e': pd.Series([1121, 12151, 12181, 12111, 12221, 12251, 13121],
                                             dtype='int')})
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    cn_tb: Table = Table.from_pandas(ctx, pdf_float)
    indexing_schema = IndexingSchema.LINEAR
    drop_index = True

    print("Before Indexing")
    print(cn_tb)

    cn_tb.set_index('a', indexing_schema, drop_index)

    pdf_float = pdf_float.set_index('a')

    print("After Indexing")
    assert cn_tb.column_names == ['b', 'c', 'd', 'e']

    assert cn_tb.get_index().get_schema() == IndexingSchema.LINEAR

    iloc_cn_1 = cn_tb.iloc[3:5, 1:3]
    iloc_pd_1 = pdf_float.iloc[3:5, 1:4]

    print(iloc_cn_1)
    print(iloc_pd_1)

    assert iloc_pd_1.values.tolist() == iloc_cn_1.to_pandas().values.tolist()

    iloc_cn_2 = cn_tb.iloc[3:5, 1:]
    iloc_pd_2 = pdf_float.iloc[3:5, 1:]

    assert iloc_pd_2.values.tolist() == iloc_cn_2.to_pandas().values.tolist()

    iloc_cn_3 = cn_tb.iloc[3:, 1:]
    iloc_pd_3 = pdf_float.iloc[3:, 1:]

    assert iloc_pd_3.values.tolist() == iloc_cn_3.to_pandas().values.tolist()

    iloc_cn_4 = cn_tb.iloc[:3, 1:]
    iloc_pd_4 = pdf_float.iloc[:3, 1:]

    assert iloc_pd_4.values.tolist() == iloc_cn_4.to_pandas().values.tolist()

    iloc_cn_5 = cn_tb.iloc[:, :]
    iloc_pd_5 = pdf_float.iloc[:, :]

    assert iloc_pd_5.values.tolist() == iloc_cn_5.to_pandas().values.tolist()


def test_isin():
    ctx = CylonContext(config=None, distributed=False)
    csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30)
    table_path = '/tmp/duplicate_data_0.csv'
    tb: Table = read_csv(ctx, table_path, csv_read_options)
    pdf: pd.DataFrame = tb.to_pandas()

    tb.set_index(tb.column_names[0], drop=True)
    pdf.set_index(pdf.columns[0], drop=True, inplace=True)

    assert tb.index.values.tolist() == pdf.index.values.tolist()

    compare_values = [4, 1, 10, 100, 150]

    tb_res_isin = tb.index.isin(compare_values)
    pdf_res_isin = pdf.index.isin(compare_values)

    assert tb_res_isin.tolist() == pdf_res_isin.tolist()


def test_isin_with_getitem():
    ctx = CylonContext(config=None, distributed=False)
    csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30)
    table_path = '/tmp/duplicate_data_0.csv'
    tb: Table = read_csv(ctx, table_path, csv_read_options)
    pdf: pd.DataFrame = tb.to_pandas()

    tb.set_index(tb.column_names[0], drop=True)
    pdf.set_index(pdf.columns[0], drop=True, inplace=True)

    assert tb.index.values.tolist() == pdf.index.values.tolist()

    compare_values = [4, 1, 10, 100, 150]

    tb_res_isin = tb.index.isin(compare_values)
    pdf_res_isin = pdf.index.isin(compare_values)

    assert tb_res_isin.tolist() == pdf_res_isin.tolist()

    print(tb_res_isin)

    print(pdf_res_isin)

    pdf1 = pdf[pdf_res_isin]

    print(pdf1)

    tb_filter = Table.from_list(ctx, ['filter'], [tb_res_isin.tolist()])
    tb1 = tb[tb_filter]
    resultant_index = tb.index.values[tb_res_isin].tolist()
    tb1.set_index(resultant_index)
    print(tb1)

    assert pdf1.values.tolist() == tb1.to_pandas().values.tolist()
    assert tb1.index.values.tolist() ==  pdf1.index.values.tolist()






test_isin_with_getitem()