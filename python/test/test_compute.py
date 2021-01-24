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


def test_isin():
    dict_elems = {'num_legs': [2, 4], 'num_wings': [2, 0]}

    indices = ['falcon', 'dog']
    indices_cmp = ['spider', 'falcon']
    df = pd.DataFrame(dict_elems, index=indices)
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    cn_tb = cn.Table.from_pydict(ctx, dict_elems)
    cn_tb.set_index(indices)

    ########

    list_comp_values = [2, 0]
    dict_comp_values = {'num_legs': [2, 0]}
    dict_comp_elements = {'num_legs': [8, 2], 'num_wings': [0, 2]}
    cn_tb_other = cn.Table.from_pydict(ctx, dict_comp_elements)
    cn_tb_other.set_index(indices_cmp)
    other = pd.DataFrame(dict_comp_elements, index=indices_cmp)

    comp_values = [list_comp_values, dict_comp_values]

    for comp_val in comp_values:
        assert df.isin(comp_val).values.tolist() == cn_tb.isin(comp_val).to_pandas(
        ).values.tolist()

    assert df.isin(other).values.tolist() == cn_tb.isin(cn_tb_other).to_pandas().values.tolist()


def test_table_is_in_dev():
    from typing import List
    from pyarrow.compute import and_
    from pyarrow import compute as a_compute
    col_validity = [False, True]
    # comparison data needs to be broadcasted in such a manner that it equals to the number of
    # rows in the table
    cols = 2
    rows = 4
    col_names_ = ['col-1', 'col-2']
    comp_col_names_ = ['col-11', 'col-2']
    row_indices_ = ['1', '2', '3', '4']
    row_indices_cmp_ = ['1', '21', '3', '41']

    data = [[2, 4, 3, 1], [0, 2, 1, 2]]
    cmp_data = [[12, 4, 13, 1], [10, 2, 12, 2]]
    ctx: CylonContext = CylonContext(config=None, distributed=False)
    tb = cn.Table.from_list(ctx, col_names_, data)
    tb.set_index(row_indices_)
    tb_cmp = cn.Table.from_list(ctx, comp_col_names_, cmp_data)
    tb_cmp.set_index(row_indices_cmp_)

    def compare_array_like_values(l_org_ar, l_cmp_ar, skip_null=True):
        s = a_compute.SetLookupOptions(value_set=l_cmp_ar, skip_null=skip_null)
        return a_compute.is_in(l_org_ar, options=s)

    def broadcast(ar, broadcast_coefficient=1):
        bcast_ar = []
        for elem in ar:
            bcast_elems = []
            for i in range(broadcast_coefficient):
                bcast_elems.append(elem.as_py())
            bcast_ar.append(pa.array(bcast_elems))
        return bcast_ar

    def compare_two_arrays(l_ar, r_ar):
        return a_compute.and_(l_ar, r_ar)

    def compare_row_and_column(row, columns):
        comp_res = []
        for column in columns:
            print(type(column), type(row))
            comp_res.append(compare_two_arrays(l_ar=row, r_ar=column))
        return comp_res

    def populate_column_with_single_value(value, row_count):
        column_values = []
        for i in range(row_count):
            column_values.append(value)
        return column_values

    def tb_compare_values(tb, tb_cmp, skip_null=True):

        col_names = tb.column_names
        comp_col_names = tb_cmp.column_names

        row_indices = tb.index.index_values
        row_indices_cmp = tb_cmp.index.index_values

        col_comp_res = compare_array_like_values(l_org_ar=pa.array(col_names), l_cmp_ar=pa.array(
            comp_col_names))
        row_comp_res = compare_array_like_values(l_org_ar=pa.array(row_indices), l_cmp_ar=pa.array(
            row_indices_cmp))
        bcast_col_comp_res = broadcast(ar=col_comp_res, broadcast_coefficient=rows)
        row_col_comp = compare_row_and_column(row=row_comp_res, columns=bcast_col_comp_res)

        tb_ar = tb.to_arrow().combine_chunks()
        tb_cmp_ar = tb_cmp.to_arrow().combine_chunks()

        col_data_map = {}
        for col_name, validity, row_col_validity in zip(col_names, col_comp_res, row_col_comp):
            if validity.as_py():
                chunk_ar_org = tb_ar.column(col_name)
                chunk_ar_cmp = tb_cmp_ar.column(col_name)
                s = a_compute.SetLookupOptions(value_set=chunk_ar_cmp, skip_null=skip_null)
                data_cmp_res = a_compute.is_in(chunk_ar_org, options=s)
                print(data_cmp_res, row_col_validity)
                col_data_map[col_name] = compare_two_arrays(data_cmp_res, row_col_validity)
            else:
                col_data_map[col_name] = pa.array(
                    populate_column_with_single_value(False, tb.row_count))

        is_in_values = list(col_data_map.values())
        return cn.Table.from_list(tb.context, col_names, is_in_values)

    new_tb = tb_compare_values(tb, tb_cmp)
    print(new_tb)
