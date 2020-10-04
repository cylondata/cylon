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

from pycylon.csv import csv_reader
from pycylon import Table
from pyarrow import Table as PyArrowTable
import time
from pycylon import CylonContext
import argparse

ctx: CylonContext = CylonContext(config="mpi")

parser = argparse.ArgumentParser(description='PyCylon Table Conversion')
parser.add_argument('--table1_path', type=str, help='Path to table 1 csv')
parser.add_argument('--table2_path', type=str, help='Path to table 2 csv')

args = parser.parse_args()

tb1: Table = csv_reader.read(ctx, args.table1_path, ',')
tb2: Table = csv_reader.read(ctx, args.table2_path, ',')

tb1.show()

print("First Hello World From Rank {}, Size {}".format(ctx.get_rank(), ctx.get_world_size()))

tb3: Table = tb1.distributed_join(ctx, table=tb2, join_type='left', algorithm='hash', left_on=[0],
                                  right_on=[0])

tb3.show()

# print("Union Test")
# tb4: Table = tb1.union(ctx, table=tb2)
# tb4.show()
#
# print("Distributed Union Test")
# tb5: Table = tb1.distributed_union(ctx, table=tb2)
# tb5.show()

print("Intersect Test")
tb4: Table = tb1.intersect(ctx, table=tb2)
tb4.show()

print("Distributed Intersect Test")
tb5: Table = tb1.distributed_intersect(ctx, table=tb2)
tb5.show()

print("Subtract Test")
tb4: Table = tb1.subtract(ctx, table=tb2)
tb4.show()

print("Distributed Subtract Test")
tb5: Table = tb1.distributed_subtract(ctx, table=tb2)
tb5.show()

ctx.finalize()
