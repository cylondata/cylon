"""
Running Example

In terminal 0: Start Scheduler

$ dask-scheduler

In terminals [1,2] : Start Worker Instances

dask-worker tcp://192.168.0.10:8786

In terminal 4:

>>> python test/test_cylon_dask_client.py

"""
from pycylon.data.table import csv_reader
from pycylon.data.table import Table
from pycylon.ctx.context import CylonContext

from dask.distributed import Client
from dask.distributed import Future
from dask import delayed
client = Client('127.0.0.1:8786')

ctx: CylonContext = CylonContext("mpi")

def read_csv(file_path) -> Table:
    ctx: CylonContext = CylonContext("mpi")
    tb1: Table = csv_reader.read(ctx, file_path, ',')
    return Table.to_arrow(tb1)

def join(tb1_arw, tb2_arw, configs):
    ctx: CylonContext = CylonContext("mpi")
    tb1: Table = Table.from_arrow(tb1_arw)
    tb3: Table = tb1.distributed_join(ctx, table=Table.from_arrow(tb2_arw), join_type=configs['join_type'], algorithm=configs['algorithm'],
                                      left_col=configs['left_col'], right_col=configs['right_col'])
    return Table.to_arrow(tb3)


read_csv_delayed = delayed(read_csv)
join_delayed = delayed(join)

configs = {'join_type':'left', 'algorithm':'hash', 'left_col':0, 'right_col':0}
file_path_1 = '/tmp/csv1.csv'
file_path_2 = '/tmp/csv2.csv'

tb1_read = read_csv_delayed(file_path_1)
tb2_read = read_csv_delayed(file_path_2)
join_comp = join_delayed(tb1_read, tb2_read, configs)

tb_res1 = tb1_read.compute()
tb_res2 = tb2_read.compute()
join_res = join_comp.compute()

tb_res:Table = Table.from_arrow(join_res)
print("Distributed Join with Dask")
tb_res.show()

ctx.finalize()
