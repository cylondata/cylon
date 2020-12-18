import numpy as np
from pycylon import Table
from pycylon import CylonContext
from pycylon.net import MPIConfig


def shuffle():
    mpi_config = MPIConfig()

    ctx = CylonContext(config=mpi_config, distributed=True)
    rows = 5
    tb: Table = Table.from_pydict(ctx, {'c1': [i for i in range(rows)], 'c2': [i * 2 for i in range(
        rows)], 'c3': [i * 3 for i in range(rows)]})

    tb: Table = Table.from_numpy(ctx, ['c1', 'c2', 'c3'], [np.random.random(size=rows),
                                                           np.random.random(size=rows),
                                                           np.random.random(size=rows)])

    print(tb.shape)

    tb_shuffle = tb.shuffle(['c1'])

    tb_shuffle_dna = tb_shuffle.dropna(axis=1, how='all')

    print("Rank : ", ctx.get_rank(), tb_shuffle.shape, tb.shape, tb_shuffle_dna.shape)

    from pycylon.io import CSVWriteOptions

    csv_write_options = CSVWriteOptions().with_delimiter(',')
    #
    # tb_shuffle.to_csv(f'/tmp/shuffle_{rows}_{ctx.get_rank()}.csv', csv_write_options)

    ctx.finalize()
