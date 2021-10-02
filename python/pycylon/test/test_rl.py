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
>> pytest -q python/pycylon/test/test_rl.py
'''

import os
from pycylon import Table
from pyarrow import Table as PyArrowTable
from pycylon.net import MPIConfig
from pycylon import CylonContext
from pycylon.io import CSVReadOptions
from pycylon.io import read_csv

def test_rl():
    ctx: CylonContext = CylonContext(config=None, distributed=False)

    table1_path = 'data/input/user_usage_tm_1.csv'
    table2_path = 'data/input/user_usage_tm_2.csv'

    assert os.path.exists(table1_path) and os.path.exists(table2_path)

    csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30)

    tb1: Table = read_csv(ctx, table1_path, csv_read_options)

    tb2: Table = read_csv(ctx, table2_path, csv_read_options)

    print("First Hello World From Rank {}, Size {}".format(ctx.get_rank(), ctx.get_world_size()))

    tb3: Table = tb1.join(table=tb2, join_type='inner', algorithm='hash', left_on=[0],
                          right_on=[0])

    assert tb3.row_count == 458 and tb3.column_count == 8

    tb4: Table = tb1.union(tb2)

    assert tb4.row_count == 240 and tb4.column_count == 4

    tb5: Table = tb1.subtract(tb2)

    assert tb5.row_count == 0 and tb5.column_count == 4

    tb6: Table = tb1.intersect(tb2)

    assert tb6.row_count == 240 and tb6.column_count == 4

    ctx.finalize()
