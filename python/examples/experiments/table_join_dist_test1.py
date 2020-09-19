import argparse
import logging
import os
import time

import pyarrow
from pyarrow import csv as ar_csv
from pycylon.ctx.context import CylonContext
from pycylon.data.table import Table
from pycylon.data.table import csv_reader

parser = argparse.ArgumentParser(description='generate random data')
parser.add_argument('-s', dest='src_dir', type=str, help='source dir', required=True)
parser.add_argument('-b', dest='base_dir', type=str, help='base dir', required=True)

args = parser.parse_args()
args = vars(args)

src_dir = args['src_dir']
base_dir = args['base_dir']


def RunJoin(rank: int, ctx: CylonContext, table1: Table, table2: Table, join_type: str,
            join_algorithm: str, left_col: int, right_col: int) -> bool:
    #w_sz: int = ctx.get_world_size()
    w_sz = 1111
    staus: bool = False
    t1 = time.time()
    table3: Table = table1.distributed_join(ctx, table=table2, join_type=join_type,
                                            algorithm=join_algorithm, left_col=left_col,
                                            right_col=right_col)
    t2 = time.time()
    ctx.barrier()
    t3 = time.time()

    line = f"####time {w_sz} {rank} j_t {t2 - t1:.3f} w_t {t3 - t2:.3f} tot {t3 - t1:.3f} lines " \
           f"{table3.rows} t {join_type} a {join_algorithm}"
    print(line, flush=True)

    return staus


ctx: CylonContext = CylonContext("mpi")

rank: int = ctx.get_rank()
world_size: int = ctx.get_world_size()

srank = f"{rank:03}"
# srank = f"{rank}"
sworld_size = str(world_size)

os.system(f"mkdir -p {base_dir}; rm -f {base_dir}/*.csv")

csv1: str = os.path.join(base_dir, f"csv1_{srank}.csv")
csv2: str = os.path.join(base_dir, f"csv2_{srank}.csv")

src1 = f"{src_dir}/csv1_{srank}.csv"
src2 = f"{src_dir}/csv2_{srank}.csv"

print("src files ", src1, src2, world_size, flush=True)

os.system(f"cp {src1} {csv1}")
os.system(f"cp {src2} {csv2}")

logging.info(f"{srank} Reading tables")

opts = ar_csv.ReadOptions(column_names=['0', '1', '2', '3'])
arrow_tb1 = ar_csv.read_csv(csv1, read_options=opts)
arrow_tb2 = ar_csv.read_csv(csv2, read_options=opts)


table1: Table = Table.from_arrow(arrow_tb1)

table2: Table = Table.from_arrow(arrow_tb2)


try:
    logging.info(
        f"Table 1 & 2 Rows [{table1.rows},{table2.rows}], Columns [{table1.columns},{table1.columns}]")
except Exception:
    raise Exception("Something went wrong in loading tables from disk")

logging.info("Inner Join Start")

RunJoin(rank=rank, ctx=ctx, table1=table1, table2=table2, join_type="inner", join_algorithm="hash",
        left_col=0, right_col=0)
'''
RunJoin(rank=rank, ctx=ctx, table1=table1, table2=table2, join_type="inner", join_algorithm="sort",
        left_col=0, right_col=0)
'''
ctx.finalize()

print(f"Removing File {csv1}")
print(f"Removing File {csv2}")
os.system("rm " + csv1)
os.system("rm " + csv2)
