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

>>> python python/test/test_table.py --table_path /tmp/user_device_tm_1.csv
"""

from pyarrow.csv import read_csv as pyarrow_read_csv
from pycylon import Table
from pycylon import CylonContext
from pycylon.io import CSVReadOptions
from pycylon.io import CSVWriteOptions
from pycylon.io import read_csv
import argparse

ctx: CylonContext = CylonContext(config=None, distributed=False)

parser = argparse.ArgumentParser(description='PyCylon Table')
parser.add_argument('--table_path', type=str, help='Path to table csv')

args = parser.parse_args()

pyarrow_table = pyarrow_read_csv(args.table_path)

print(pyarrow_table)

tb = Table(pyarrow_table, ctx)

ar_tb2 = tb.to_arrow()

print("Arrow Table Info \n", ar_tb2)

tb2 = Table.from_arrow(ctx, pyarrow_table)

print(f"Tb2 : Rows={tb2.row_count}, Columns={tb2.column_count}")

csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30)

tb3 = read_csv(ctx, args.table_path, csv_read_options)

print(f"Tb3 : Rows={tb3.row_count}, Columns={tb3.column_count}")

csv_write_options = CSVWriteOptions().with_delimiter(',')

tb3.to_csv('/tmp/temp.csv', csv_write_options)

tb4 = tb3.sort(1)

print(tb4.column_names)

print(f"Sort Tb 1: Rows={tb4.row_count}, Columns={tb4.column_count}")

tb5 = tb3.sort('use_type_id')

print(f"Sort Tb 2: Rows={tb5.row_count}, Columns={tb5.column_count}")

print(tb5.column_names)

tb6 = Table.merge(ctx, [tb4, tb4])

print(f"Merge Tb: Rows={tb6.row_count}, Columns={tb6.column_count}")

tb7 = tb6

print(f"Copy Tb: Rows={tb7.row_count}, Columns={tb7.column_count}")

tb8 = tb3.project([0, 1])

print(f"Project Tb: Rows={tb8.row_count}, Columns={tb8.column_count}")

tb9 = tb3.project(['use_id', 'platform_version'])

print(f"Project Tb By Columns: Rows={tb9.row_count}, Columns={tb9.column_count}")

print(tb9.column_names)

tb9.show(row1=0, row2=5, col1=0, col2=2)

ctx.finalize()
