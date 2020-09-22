from pycylon.csv import csv_reader
from pycylon import Table
from pycylon import CylonContext
import argparse

ctx: CylonContext = CylonContext("mpi")

parser = argparse.ArgumentParser(description='PyCylon Table Conversion')
parser.add_argument('--table1_path', type=str, help='Path to table 1 csv')
parser.add_argument('--table2_path', type=str, help='Path to table 2 csv')

args = parser.parse_args()


tb1: Table = csv_reader.read(ctx, args.table1_path, ',')

tb2: Table = csv_reader.read(ctx, args.table2_path, ',')

configs = {'join_type': 'left', 'algorithm': 'hash', 'left_col': 0, 'right_col': 0}

tb3: Table = tb1.distributed_join(ctx, table=tb2, join_type=configs['join_type'], algorithm=configs['algorithm'],
                                  left_col=configs['left_col'], right_col=configs['right_col'])

tb3.show()

ctx.finalize()
