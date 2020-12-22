import pandas as pd
import pycylon as cn
from pycylon.net import MPIConfig
from pycylon.io import CSVReadOptions
from pycylon.io import read_csv

def test_shuffle():
    mpi_config = MPIConfig()

    ctx = cn.CylonContext(config=mpi_config, distributed=True)

    tb: cn.Table = None

    rank = ctx.get_rank()

    if rank == 0:
        tb = cn.Table.from_pydict(ctx, {'c1': [1, 1, 3, 3, 4, 5], 'c2': [2, 2, 2, 4, 6, 6],
                                        'c3': [3, 3,
                                               3, 5,
                                               7,
                                               7]})

    if rank == 1:
        tb = cn.Table.from_pydict(ctx, {'c1': [5, 1, 1, 4, 1, 10], 'c2': [6, 2, 1, 5, 0, 1],
                                        'c3': [7, 3,
                                               0, 5,
                                               1,
                                               5]})

    tb = tb.shuffle(tb.column_names)

    print(rank, "\n>>>", tb)

    ctx.finalize()


def test_unique():

    ctx = cn.CylonContext(config=None, distributed=False)
    csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30)
    table_path = '/tmp/duplicate_data_0.csv'
    tb1: cn.Table = read_csv(ctx, table_path, csv_read_options)
    pdf: pd.DataFrame = tb1.to_pandas()

    expected_indices_of_sort_col = [1, 2, 3, 4, 5, 7, 10, 12, 13, 14, 15]

    tb2 = tb1.unique(columns=['a', 'b'], keep='first')
    pdf2 = pdf.drop_duplicates(subset=['a', 'b'])

    sort_col = tb2.sort(3).to_pydict()['d']

    assert sort_col == expected_indices_of_sort_col

    assert pdf2['d'].values.tolist() == sort_col


test_unique()
