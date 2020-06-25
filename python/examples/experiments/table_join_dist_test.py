import os
import sys
import socket
import logging
import time

import pytwisterx as twx
import pyarrow as pa
from pytwisterx.data.table import csv_reader
from pytwisterx.data.table import Table
from pyarrow import Table as PyArrowTable
import time
from pytwisterx.ctx.context import TwisterxContext
   

def RunJoin(rank: int, ctx: TwisterxContext, table1: Table, table2: Table, join_type:str, join_algorithm:str, left_col:int, right_col:int) -> bool:
    staus: bool = False
    t1 = time.time()
    table3: Table = table1.distributed_join(ctx, table=table2, join_type=join_type, algorithm=join_algorithm, left_col=left_col, right_col=right_col)
    t2 = time.time()
    ctx.barrier()
    t3 = time.time()
    try:
        logging.info(f"j_t {str(t2-t1)} sec, w_t {t3-t2} ,lines {table3.rows} t {join_type} a {join_algorithm}")
        with open("results.csv","a") as fp:
            line = f"{join_algorithm} {str(ctx.get_world_size())} >> j_t {str(t2-t1)} sec, w_t {t3-t2} ,lines {table3.rows} t {join_type} a {join_algorithm}"
            write_line = line + "\n"
            fp.write(write_line)
        staus = True
    except Exception:
        logging.error("Something went wrong in the distributed join")
    finally:
        return staus



ctx: TwisterxContext = TwisterxContext("mpi")

argv = sys.argv
argc = len(argv)

hostname: str = socket.gethostname()

rank: int = ctx.get_rank()
world_size: int = ctx.get_world_size()

srank = str(rank)
sworld_size = str(world_size)

base_dir: str = "/tmp" if argc > 1 else f"/scratch/{hostname}"

os.system("mkdir -p " + base_dir)

csv1: str = os.path.join(base_dir, f"csv1_{srank}.csv")
csv2: str = os.path.join(base_dir, f"csv2_{srank}.csv")

os.system(f"cp ~/temp/csv1_{srank}.csv {csv1}")
os.system(f"cp ~/temp/csv2_{srank}.csv {csv2}")

logging.info(f"{srank} Reading tables")

table1: Table = csv_reader.read(ctx, csv1, ',')
table2: Table = csv_reader.read(ctx, csv2, ',')

try:
    logging.info(f"Table 1 & 2 Rows [{table1.rows},{table2.rows}], Columns [{table1.columns},{table1.columns}]")
except Exception:
    raise("Something went wrong in loading tables from disk")

logging.info("Inner Join Start")

RunJoin(rank=rank, ctx=ctx, table1=table1, table2=table2, join_type="inner", join_algorithm="sort", left_col=0, right_col=0)
RunJoin(rank=rank, ctx=ctx, table1=table1, table2=table2, join_type="inner", join_algorithm="hash", left_col=0, right_col=0)

ctx.finalize()



