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
Run test
>> pytest -q python/test/test_ra_by_column_names.py
'''

import os
from pycylon import Table
from pycylon import CylonContext
from pycylon.io import CSVReadOptions
from pycylon.io import read_csv
from pycylon.net import MPIConfig


def test_single_process():
    ctx: CylonContext = CylonContext(config=None, distributed=False)

    csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30)

    table1_path = '/tmp/user_device_tm_1.csv'
    table2_path = '/tmp/user_usage_tm_1.csv'

    assert os.path.exists(table1_path) and os.path.exists(table2_path)

    tb1: Table = read_csv(ctx, table1_path, csv_read_options)

    tb2: Table = read_csv(ctx, table2_path, csv_read_options)

    configs = {'join_type': 'inner', 'algorithm': 'sort'}

    tb3: Table = tb1.join(table=tb2,
                          join_type=configs['join_type'],
                          algorithm=configs['algorithm'],
                          left_on=[0],
                          right_on=[3]
                          )

    print(tb3.row_count, tb3.column_count)

    tb4: Table = tb1.join(table=tb2,
                          join_type=configs['join_type'],
                          algorithm=configs['algorithm'],
                          left_on=['use_id'],
                          right_on=['use_id']
                          )

    tb5: Table = tb1.join(table=tb2,
                          join_type=configs['join_type'],
                          algorithm=configs['algorithm'],
                          on=['use_id']
                          )

    # tb6: Table = tb1.join(ctx, table=tb2,
    #                       join_type=configs['join_type'],
    #                       algorithm=configs['algorithm'],
    #                       on=[0]
    #                       )
    #
    # tb5.show()
    assert tb3.row_count == tb4.row_count == tb5.row_count and tb3.column_count == \
           tb4.column_count == tb5.column_count
    ctx.finalize()
