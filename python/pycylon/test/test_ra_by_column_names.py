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
>> pytest -q python/pycylon/test/test_ra_by_column_names.py
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

    table1_path = 'data/input/user_device_tm_1.csv'
    table2_path = 'data/input/user_usage_tm_1.csv'

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

    assert tb3.row_count == tb4.row_count == tb5.row_count and tb3.column_count == \
           tb4.column_count == tb5.column_count
    ctx.finalize()


def test_prefix_process():
    ctx: CylonContext = CylonContext(config=None, distributed=False)

    csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30)

    table1_path = 'data/input/user_device_tm_1.csv'
    table2_path = 'data/input/user_usage_tm_1.csv'

    assert os.path.exists(table1_path) and os.path.exists(table2_path)

    tb1: Table = read_csv(ctx, table1_path, csv_read_options)

    tb2: Table = read_csv(ctx, table2_path, csv_read_options)

    configs = {'join_type': 'inner', 'algorithm': 'sort'}

    tb3: Table = tb1.join(table=tb2,
                          join_type=configs['join_type'],
                          algorithm=configs['algorithm'],
                          left_on=[0],
                          right_on=[3],
                          left_prefix="l_",
                          right_prefix="r_",
                          )

    print(tb3.row_count, tb3.column_count)

    tb4: Table = tb1.join(table=tb2,
                          join_type=configs['join_type'],
                          algorithm=configs['algorithm'],
                          left_on=['use_id'],
                          right_on=['use_id'],
                          left_prefix="l_1_",
                          right_prefix="r_1_",
                          )

    tb5: Table = tb1.join(table=tb2,
                          join_type=configs['join_type'],
                          algorithm=configs['algorithm'],
                          on=['use_id'],
                          left_prefix="l_2_",
                          right_prefix="r_2_",
                          )

    assert tb3.row_count == tb4.row_count == tb5.row_count and tb3.column_count == \
           tb4.column_count == tb5.column_count
    expected_column_names_1 = ['l_use_id', 'user_id', 'platform_version', 'use_type_id',
                               'outgoing_mins_per_month', 'outgoing_sms_per_month',
                               'monthly_mb', 'r_use_id']

    expected_column_names_2 = ['l_1_use_id', 'user_id', 'platform_version',
                               'use_type_id', 'outgoing_mins_per_month',
                               'outgoing_sms_per_month',
                               'monthly_mb', 'r_1_use_id']

    expected_column_names_3 = ['l_2_use_id', 'user_id', 'platform_version',
                               'use_type_id', 'outgoing_mins_per_month',
                               'outgoing_sms_per_month',
                               'monthly_mb', 'r_2_use_id']

    assert expected_column_names_1 == tb3.column_names
    assert expected_column_names_2 == tb4.column_names
    assert expected_column_names_3 == tb5.column_names

    ctx.finalize()


test_prefix_process()
