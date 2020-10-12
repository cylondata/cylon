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
from pycylon.io.csv_read_config import CSVReadOptions
from pycylon.io.csv_write_config import CSVWriteOptions
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

ar_tb2 = tb.to_arrow()

print(ar_tb2)

tb2 = Table.from_arrow(ctx, pyarrow_table)

tb2.show()

csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30)

tb3 = Table.from_csv(ctx, args.table_path, csv_read_options)

tb3.show()

csv_write_options = CSVWriteOptions().with_delimiter(',')

tb3.to_csv('/tmp/temp.csv', csv_write_options)

tb4 = tb3.sort(1)

tb4.show()

tb5 = tb3.sort('use_type_id')

tb5.show()

print(tb4.column_names)

tb6 = Table.merge(ctx, [tb4, tb4])

tb6.show()

print(f"Tb5 : Rows {tb5.rows_count}, Columns : {tb5.columns_count}")

print(f"After Merge Tb6 : Rows {tb6.rows_count}, Columns : {tb6.columns_count}")

ctx.finalize()
