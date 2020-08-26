from pycylon.data.table import csv_reader
from pycylon.data.table import Table
from pycylon.ctx.context import CylonContext

ctx: CylonContext = CylonContext("mpi")

tb1: Table = csv_reader.read(ctx, '/tmp/csv1.csv', ',')

tb2: Table = csv_reader.read(ctx, '/tmp/csv2.csv', ',')

configs = {'join_type':'left', 'algorithm':'hash', 'left_col':0, 'right_col':0}

tb3: Table = tb1.distributed_join(ctx, table=tb2, join_type=configs['join_type'], algorithm=configs['algorithm'],
                                  left_col=configs['left_col'], right_col=configs['right_col'])

tb3.show()

ctx.finalize()