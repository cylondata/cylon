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

from pycylon.data.table import csv_reader
from pycylon.data.table import Table
from pyarrow import Table as PyArrowTable
import time
from pycylon.ctx.context import CylonContext

ctx: CylonContext = CylonContext("mpi")


tb1: Table = csv_reader.read(ctx, '/tmp/csv.csv', ',')
tb2: Table = csv_reader.read(ctx, '/tmp/csv.csv', ',')

tb1.show()

print("First Hello World From Rank {}, Size {}".format(ctx.get_rank(), ctx.get_world_size()))

tb3: Table = tb1.distributed_join(ctx, table=tb2, join_type='left', algorithm='hash', left_col=0, right_col=0)

tb3.show()

print("Union Test")
tb4: Table = tb1.union(ctx, table=tb2)
tb4.show()

print("Distributed Union Test")
tb5: Table = tb1.distributed_union(ctx, table=tb2)
tb5.show()

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
