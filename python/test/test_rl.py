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

'''
Run this file:
>>> python python/test/test_rl.py \
    --table1_path /tmp/user_usage_tm_1.csv \
    --table2_path /tmp/user_usage_tm_2.csv
'''

import time
import argparse
from pycylon import Table
from pyarrow import Table as PyArrowTable
from pycylon.net import MPIConfig
from pycylon import CylonContext
from pycylon.io import CSVReadOptions

ctx: CylonContext = CylonContext(config=None, distributed=False)

parser = argparse.ArgumentParser(description='PyCylon Table Conversion')
parser.add_argument('--table1_path', type=str, help='Path to table 1 csv')
parser.add_argument('--table2_path', type=str, help='Path to table 2 csv')

args = parser.parse_args()

csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30)

tb1: Table = Table.from_csv(ctx, args.table1_path, csv_read_options)

tb2: Table = Table.from_csv(ctx, args.table2_path, csv_read_options)

print("First Hello World From Rank {}, Size {}".format(ctx.get_rank(), ctx.get_world_size()))

tb3: Table = tb1.join(table=tb2, join_type='inner', algorithm='hash', left_on=[0],
                      right_on=[0])

print(f"Join Completed : Rows={tb3.row_count}, Columns={tb3.column_count}")

tb4: Table = tb1.union(tb2)

print(f"Union Completed : Rows={tb4.row_count}, Columns={tb4.column_count}")

tb5: Table = tb1.subtract(tb2)

print(f"Subtract Completed : Rows={tb5.row_count}, Columns={tb5.column_count}")

tb6: Table = tb1.intersect(tb2)

print(f"Intersect Completed : Rows={tb6.row_count}, Columns={tb6.column_count}")

