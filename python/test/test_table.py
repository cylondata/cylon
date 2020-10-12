##
 # Licensed under the Apache License, Version 2.0 (the "License");
 # you may not use this file except in compliance with the License.
 # You may obtain a copy of the License at
 #
 # http://www.apache.org/licenses/LICENSE-2.0
 #
 # Unless required by applicable law or agreed to in writing, software
 # distributed under the License is distributed on an "AS IS" BASIS,
 # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 # See the License for the specific language governing permissions and
 # limitations under the License.
 ##


"""
Run test:

>>> python python/test/test_table.py --table_path /tmp/csv.csv
"""

import pyarrow as pa
from pyarrow.csv import read_csv
from pycylon.data.table import Table
from pycylon.ctx.context import CylonContext
from pycylon.net.mpi_config import MPIConfig
import argparse

mpi_config = MPIConfig()
ctx: CylonContext = CylonContext(config=mpi_config, distributed=False)

parser = argparse.ArgumentParser(description='PyCylon Table')
parser.add_argument('--table_path', type=str, help='Path to table csv')

args = parser.parse_args()

pyarrow_table = read_csv(args.table_path)

print(pyarrow_table)

tb = Table(pyarrow_table, ctx)

tb.show()

tb2 = Table.from_arrow(ctx, pyarrow_table)

tb2.show()



# tb1: Table = csv_reader.read(ctx, args.table_path, ',')
#
# print(f"Cylon Table Rows {tb1.rows}, Columns {tb1.columns}")

ctx.finalize()