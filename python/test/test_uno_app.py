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
    tb_cn['d'] = tb_cn['d'].applymap(lambda x: x.replace('-', ''))

    print(tb_cn)


def test_filter():
    npr = np.array([1, 2, 3, 4, 5])

    def map_func(x):
        return x + 1

    npr_map = map_func(npr)
    print(npr_map)


def test_unique():
    ctx = CylonContext(config=None, distributed=False)
    csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30)
    table_path = '/tmp/duplicate_data_0.csv'
    tb1: Table = read_csv(ctx, table_path, csv_read_options)
    pdf: pd.DataFrame = tb1.to_pandas()

    expected_indices_of_sort_col = [1, 2, 3, 4, 5, 7, 10, 12, 13, 14, 15]

    print("Original Data")
    print(pdf)

    tb2 = tb1['b'].unique()
    pdf2 = pdf['b'].unique()
    tb2.show()

    print("Unique Pdf")
    print(pdf2)
    print(type(pdf2))

    print("Unique Cylon")
    print(tb2)

    tb3_list = list(tb2.to_pydict().items())[0][1]
    pdf3_list = pdf2.tolist()

    assert tb3_list == pdf3_list

    set_pdf4 = set(pdf2)
    set_tb4 = set(tb3_list)

    assert set_tb4 == set_pdf4

    ctx.finalize()


def test_series_tolist():
    ctx = CylonContext(config=None, distributed=False)
    csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30)
    table_path = '/tmp/duplicate_data_0.csv'
    tb1: Table = read_csv(ctx, table_path, csv_read_options)
    pdf: pd.DataFrame = tb1.to_pandas()

    series = pdf[pdf.columns[0]]

    print(type(series))

    lst = series.tolist()
    npy = series.to_numpy()

    print(lst)
    idx = series.index.values
    print(type(idx), idx)


def test_set_list_conv():
    lst = [i for i in range(1_000_000)]
    ar = np.array(lst)
    t1 = time.time()
    st = set(ar)
    t2 = time.time()
    lst = list(st)
    t3 = time.time()

    print(t2 - t1, t3 - t2)


def test_df_with_index():
    ctx = CylonContext(config=None, distributed=False)
    csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30)
    table_path = '/tmp/duplicate_data_0.csv'
    tb1: Table = read_csv(ctx, table_path, csv_read_options)
    pdf: pd.DataFrame = tb1.to_pandas()

    print(pdf.columns[0])
    pdf1 = pdf[[pdf.columns[0]]]

    print(pdf)

    print(pdf1)

    pdf3 = pdf.set_index(pdf.columns[0], drop=True)

    print(pdf3)

    artb = pa.Table.from_pandas(df=pdf3, schema=None,
                                preserve_index=True,
                                nthreads=None, columns=None, safe=False)

    print(artb)


def test_df_iterrows():
    ctx = CylonContext(config=None, distributed=False)
    csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30)
    table_path = '/tmp/duplicate_data_0.csv'
    tb1: Table = read_csv(ctx, table_path, csv_read_options)
    pdf: pd.DataFrame = tb1.to_pandas()

    tb1.set_index(tb1.column_names[0], drop=True)
    pdf.set_index(pdf.columns[0], drop=True, inplace=True)
    num_records = tb1.row_count
    print(pdf)

    for idx, row in pdf.iterrows():
        print(idx)
        print(row)

    dict = tb1.to_pydict(with_index=False)
    indices = tb1.index.index_values
    rows = []

    for index_id, index in enumerate(indices):
        row = []
        for col in dict:
            row.append(dict[col][index_id])
        rows.append(row)

    for index, row in zip(indices, rows):
        print(index, row)

    for index1, row1, composite in zip(indices, rows, pdf.iterrows()):
        index2 = composite[0]
        row2 = composite[1].tolist()
        assert index1 == index2
        assert row1 == row2
        # print(type(index1), index1, type(index2), index2, type(row1), row1, type(row2), row2)


def test_df_perf_iterrows():
    ctx = CylonContext(config=None, distributed=False)

    dataset = []
    num_rows = 100_000
    num_columns = 2

    data = np.random.randn(num_rows)

    pdf = pd.DataFrame({'data{}'.format(i): data
                        for i in range(num_columns)})

    tb1 = Table.from_pandas(ctx, pdf)

    tb1.set_index(tb1.column_names[0], drop=True)
    pdf.set_index(pdf.columns[0], drop=True, inplace=True)

    print(pdf)
    t1 = time.time()
    for idx, row in pdf.iterrows():
        idx = idx
        row = row
    t2 = time.time()
    dict = tb1.to_pydict(with_index=True)
    indices = tb1.index.index_values
    rows = []
    for index in indices:
        row = []
        for col in dict:
            row.append(dict[col][index])
        rows.append(row)

    for index, row in zip(indices, rows):
        index = index
        row = row
    t3 = time.time()
    print(t2 - t1, t3 - t2)


def test_read_partition_from_csv():
    file_name = '/tmp/multi_data.csv'
    RECORDS = 12
    num_processes = 4
    RECORDS_PER_PROC = RECORDS // num_processes
    OFFSET = -RECORDS_PER_PROC
    LIMIT = RECORDS_PER_PROC

    for i in range(num_processes):
        print(f"Process [{i}]")
        OFFSET = OFFSET + RECORDS_PER_PROC
        reader = pd.read_csv(file_name, sep=',',
                             header=None,
                             skiprows=lambda idx: idx < OFFSET,  # Skip lines
                             nrows=LIMIT)
        print(reader)


def test_isin_column():
    ctx = CylonContext(config=None, distributed=False)
    csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30)
    table_path = '/tmp/duplicate_data_0.csv'
    tb1: Table = read_csv(ctx, table_path, csv_read_options)
    pdf: pd.DataFrame = tb1.to_pandas()

    tb1.set_index(tb1.column_names[0], drop=True)
    pdf.set_index(pdf.columns[0], drop=True, inplace=True)

    print(tb1)
    print(pdf)

    isin_values = [10, 20, 30, 5, 2, 8]

    tbx = tb1['b'].isin(isin_values)
    pdfx = pdf['b'].isin(isin_values)

    print(tbx)

    print(pdfx)

    tb_list = tbx.to_pandas().values.flatten().tolist()
    pd_list = pdfx.values.tolist()

    assert tb_list == pd_list

    print(tb_list)
    print(pd_list)


def test_loc_with_list():
    ctx = CylonContext(config=None, distributed=False)

    dataset = []
    num_rows = 10_000
    num_columns = 2
    filter_size = 1_000

    data = np.random.randn(num_rows)
    index_vals = [i for i in range(0, num_rows)]
    filter_vals = [i for i in range(0, filter_size)]

    pdf = pd.DataFrame({'data{}'.format(i): data
                        for i in range(num_columns)})
    index_df_col = pd.DataFrame(index_vals)
    pdf['index'] = index_df_col

    tb1 = Table.from_pandas(ctx, pdf)
    tb1['index'] = Table.from_pandas(ctx, index_df_col)
    index_column = 'index'
    tb1.set_index(index_column, drop=True)
    pdf.set_index(index_column, drop=True, inplace=True)

    print(tb1.shape, pdf.shape)
    i0 = pdf.index.values[0]
    print(type(i0), i0)


def test_numpy_conversion():
    ctx = CylonContext(config=None, distributed=False)

    dataset = []
    num_rows = 10_000_000
    num_columns = 2

    data = np.random.randn(num_rows)

    pdf = pd.DataFrame({'data{}'.format(i): data
                        for i in range(num_columns)})

    tb1 = Table.from_pandas(ctx, pdf)

    t1 = time.time()
    np1 = tb1.to_numpy(zero_copy_only=True)
    t2 = time.time()
    np2 = pdf.values
    t3 = time.time()
    print(t2-t1, t3-t2)


def test_col_access():
    ctx = CylonContext(config=None, distributed=False)
    csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30)
    table_path = '/tmp/duplicate_data_0.csv'
    tb1: Table = read_csv(ctx, table_path, csv_read_options)
    pdf: pd.DataFrame = tb1.to_pandas()
    print(tb1)
    tbx = tb1[tb1.column_names[0]]
    print(tbx)
    npy = tbx.to_numpy().flatten().tolist()
    print(npy)


test_col_access()