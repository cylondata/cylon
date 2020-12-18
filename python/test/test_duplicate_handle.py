import pycylon as cn
from pycylon.net import MPIConfig

mpi_config = MPIConfig()

ctx = cn.CylonContext(config=mpi_config, distributed=True)

tb: cn.Table = None

rank = ctx.get_rank()

if rank == 0:
    tb = cn.Table.from_pydict({'c1': [1, 1, 3, 3, 4, 5], 'c2': [2, 2, 2, 4, 6, 6], 'c3': [3, 3,
                                                                                          3, 5,
                                                                                          7, 7]})

if rank == 1:
    tb = cn.Table.from_pydict({'c1': [5, 1, 1, 4, 1, 10], 'c2': [6, 2, 1, 5, 0, 1], 'c3': [7, 3,
                                                                                           0, 5,
                                                                                           1, 5]})


tb.shuffle(tb.column_names)
