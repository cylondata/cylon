import os
from pycylon import Table
from pycylon import CylonContext
from pycylon.data.aggregates import AggregationOp
from pycylon.io import CSVReadOptions
from pycylon.io import read_csv
from pycylon.io import CSVWriteOptions

import pandas as pd
import numpy as np
from pyarrow import compute as a_compute
import pyarrow as pa

import time

ctx = CylonContext(config=None, distributed=False)


def test_additions_and_maps():
    from pycylon import Table
    from pycylon import CylonContext
    import pandas as pd
    import numpy as np

    pdf = pd.DataFrame(
        {'idx': ['x', 'y', 'z'], 'col-1': ["a", "b", "c"], 'col-2': [10, 20, 30], 'col-3': [
            'Y',
            'N',
            'Y']})

    tb = Table.from_pandas(ctx, pdf)

    print(tb)

    tb_s = tb['col-1'].applymap(lambda x: x + "_i")
    tb_log = tb['col-2'].applymap(lambda x: np.log10(x))
    tb_y = tb['col-3'].applymap(lambda x: (x == 'Y'))

    tb['col-1'] = tb_s
    tb['col-2'] = tb_log

    tb = tb[tb_y]
    pdf = pdf[pdf['col-3'].map(lambda x: (x == 'Y'))]

    print(pdf.to_dict())

    print(tb.to_pydict())


def test_default_indexing():
    pdf = pd.DataFrame(
        {'idx': ['x', 'y', 'z'], 'col-1': ["a", "b", "c"], 'col-2': [10, 20, 30], 'col-3': [
            'Y',
            'N',
            'Y']})

    tb = Table.from_pandas(ctx, pdf)

    tb_idx_values = tb.index.index_values
    pdf_idx_values = pdf.index.values.tolist()

    assert tb_idx_values == pdf_idx_values


def test_str_ops():
    pdf = pd.DataFrame(
        {'idx': ['x', 'y', 'z'], 'col-1': ["a", "b", "c"], 'col-2': [10, 20, 30], 'col-3': [
            'Y',
            'N',
            'Y']})

    tb = Table.from_pandas(ctx, pdf)

    pdf_str_val = pdf['col-1'] + "_" + pdf['col-3']
    tb_str_val = tb['col-1'] + "_" + tb['col-3']

    assert pdf_str_val.values.tolist() == tb_str_val.to_pandas().values.flatten().tolist()


def test_tb_to_pydict_with_index():
    pdf = pd.DataFrame(
        {'idx': ['x', 'y', 'z'], 'col-1': ["a", "b", "c"], 'col-2': [10, 20, 30], 'col-3': [
            'Y',
            'N',
            'Y']})

    tb = Table.from_pandas(ctx, pdf)

    assert tb.to_pydict(with_index=True) == pdf.to_dict()


def test_pdf_to_pdf_assign():
    index1 = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    index2 = [0, 1, 2, 3, 4]
    index3 = [10, 11, 12, 13, 14, 15, 16, 17, 7]
    pdf1 = pd.DataFrame({'a': [1, 2, 3, 4, 5, 110, 111, 112, 113], 'b': [10, 11, 12, 13, 14, 5, 4,
                                                                         3, 2]}, index=index1)

    pdf2 = pd.DataFrame({'a': [10, 20, 30, 40, 50], 'b': [100, 101, 102, 103, 104]}, index=index2)

    pdf3 = pd.DataFrame({'a': [1, 2, 3, 4, 5, 110, 111, 112, 113], 'b': [1110, 1111, 1112, 1113,
                                                                         1114, 115, 114, 113, 112]},
                        index=index3)

    tb1 = Table.from_pandas(ctx, pdf1)
    tb1.set_index(index1)
    tb2 = Table.from_pandas(ctx, pdf2)
    tb2.set_index(index2)
    tb3 = Table.from_pandas(ctx, pdf3)
    tb3.set_index(index3)

    print(pdf1)
    print("-----------")
    print(pdf2)
    print("-----------")
    print(pdf3)
    print("-----------")
    gp = pdf1['b']
    # print(pdf1['b'] < 6)
    print(gp[pdf1['b'] < 6])
    print(gp)
    print("-----------")
    gp[pdf1['b'] < 6] = pdf3['b']
    print(gp)

    tb_gp = tb1['b']
    print(tb_gp)
    print(tb_gp.index.index_values)
    tb_sample = tb_gp[tb1['b'] < 6]
    print(tb_sample)
    print(tb_sample.index.index_values)

    # tb_gp[tb1['b'] < 6] = tb3['b']
    # print(tb_gp)

    # print(tb_gp.index.index_values)
    # print(tb3.index.index_values)

    # l_index = tb_gp.index.get_index_array()
    # r_index = tb3.index.get_index_array()
    #
    # s = a_compute.SetLookupOptions(value_set=r_index, skip_null=True)
    # available_index = a_compute.index_in(l_index, options=s)
    #
    # l_table = tb_gp.to_arrow().combine_chunks()
    # r_table = tb3.to_arrow().combine_chunks()
    #
    # l_array = l_table.column(0).chunk(0).tolist()
    # r_array = r_table.column(0).chunk(0).tolist()
    # print(l_index)
    # print(r_index)
    #
    # print(available_index)
    #
    # for l_pos, valid_index_pos in enumerate(available_index):
    #     if valid_index_pos.as_py():
    #         match_r_index = r_index[valid_index_pos.as_py()]
    #         l_array[l_pos] = r_array[match_r_index.as_py()]
    #         print(l_pos, valid_index_pos, match_r_index)
    #     else:
    #         l_array[l_pos] = None
    # l_array = pa.array(l_array)
    # print(l_array)


def test_uno_data_load():
    file_path = "/home/vibhatha/sandbox/UNO/Benchmarks/Data/Pilot1/"
    file_name = "combined_single_response_agg"
    save_file = "/tmp/combined_single_response_agg_enum"
    path = os.path.join(file_path, file_name)
    csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30).with_delimiter(
        "\t")
    t1 = time.time()
    tb = read_csv(ctx, path, csv_read_options)
    t2 = time.time()

    print(t2 - t1)

    print(tb.shape)

    print(tb.to_arrow())

    print(tb.column_names)

    tb_drugs = tb['DRUG']

    tb_drug = tb.unique(columns=['DRUG'], keep='first')['DRUG']

    tb_drug_ar_tb = tb_drug.to_arrow().combine_chunks()
    tb_drug_list = tb_drug_ar_tb.column(0).chunk(0).tolist()

    tb_drugs_ar_tb = tb_drugs.to_arrow().combine_chunks()
    tb_drugs_list = tb_drugs_ar_tb.column(0).chunk(0).tolist()

    tb_drug_list_dict = {}

    for index, drug in enumerate(tb_drug_list):
        tb_drug_list_dict[drug] = index

    tb_drugs_enum_list = []

    for drug in tb_drugs_list:
        tb_drugs_enum_list.append(tb_drug_list_dict[drug])

    tb_enum_drug = Table.from_list(ctx, ['DRUG'], [tb_drugs_enum_list])

    print(tb_enum_drug.shape, tb_drugs.shape)
    tb = tb.drop(['DRUG'])

    tb['DRUG'] = tb_enum_drug

    print(tb.to_arrow())

    pdf = tb.to_pandas()

    pdf.to_csv(save_file, sep="\t")


def test_headerless_data_load():
    file_path = '/home/vibhatha/data/cylon/none_data.csv'
    df = pd.read_csv(file_path, na_values='Nan', header=0)

    print("Initially loaded dataframe")
    print(df)

    print(df.shape)
    print("Dropna")
    tb_cn = Table.from_pandas(ctx, df)
    tb_cn.dropna(axis=1, inplace=True)
    df.dropna(axis=0, inplace=True)
    print(tb_cn.shape)
    print("-" * 10)
    print(df.shape)

    print(tb_cn)
    print("*" * 10)
    print(df)


def test_column_str_ops():
    file_path = '/home/vibhatha/data/cylon/none_data.csv'
    df = pd.read_csv(file_path, na_values='Nan', header=0)
    tb_cn = Table.from_pandas(ctx, df)
    print(df)

    df['d'] = df['d'].str.replace('-', '')
    print(df)
    tb_cn['d'] = tb_cn['d'].applymap(lambda x: x.replace('-',''))

    print(tb_cn)



