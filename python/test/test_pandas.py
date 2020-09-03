from pycylon.data.table import Table
from pycylon.ctx.context import CylonContext
import pandas as pd
import argparse

'''
running test case 
>>> python test/test_pandas.py --table_path /tmp/csv.csv 
'''

parser = argparse.ArgumentParser(description='PyCylon with Pandas')
parser.add_argument('--table_path', type=str, help='Path to csv')

args = parser.parse_args()

ctx: CylonContext = CylonContext('mpi')
pdf: pd.DataFrame = pd.read_csv(args.table_path)
cylon_table: Table = Table.from_pandas(pdf)

cylon_table.show()

print(f"Rows {cylon_table.rows}, Columns {cylon_table.columns}")

ctx.finalize()