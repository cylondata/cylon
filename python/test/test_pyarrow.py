import argparse
import pyarrow as pa
from pyarrow import csv
from pycylon.data.table import Table
from pycylon.ctx.context import CylonContext

'''
running test case 
>>> python test/test_pyarrow.py --table_path /tmp/csv.csv 
'''


parser = argparse.ArgumentParser(description='PyCylon Table Conversion')
parser.add_argument('--table1_path', type=str, help='Path to table 1 csv')

args = parser.parse_args()

ctx: CylonContext = CylonContext('mpi')

tb = csv.read_csv(args.table1_path)

print(tb)

tb = Table.from_arrow(tb, ctx)

print(ctx.get_world_size())

tb.show()

ctx.finalize()