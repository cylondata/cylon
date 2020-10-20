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
>> mpirun -n 4 python -m pytest --with-mpi -q python/test/test_ra_by_column_names.py
'''

import os
import pytest
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


@pytest.mark.mpi
def test_multi_process():
    mpi_config = MPIConfig()
    ctx: CylonContext = CylonContext(config=mpi_config, distributed=True)

    rank, size = ctx.get_rank(), ctx.get_world_size()

    assert size == 4

    csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30)

    table1_path = f'/tmp/user_device_tm_{rank + 1}.csv'
    table2_path = f'/tmp/user_usage_tm_{rank + 1}.csv'

    tb1: Table = read_csv(ctx, table1_path, csv_read_options)

    tb2: Table = read_csv(ctx, table2_path, csv_read_options)

    print(tb1.column_names)
    print(tb2.column_names)

    configs = {'join_type': 'inner', 'algorithm': 'sort', 'left_col': 0,
               'right_col': 0}

    tb3: Table = tb1.distributed_join(table=tb2,
                                      join_type=configs['join_type'],
                                      algorithm=configs['algorithm'],
                                      left_on=[0],
                                      right_on=[3]
                                      )

    tb4: Table = tb1.distributed_join(table=tb2,
                                      join_type=configs['join_type'],
                                      algorithm=configs['algorithm'],
                                      left_on=['use_id'],
                                      right_on=['use_id']
                                      )

    tb5: Table = tb1.distributed_join(table=tb2,
                                      join_type=configs['join_type'],
                                      algorithm=configs['algorithm'],
                                      on=['use_id']
                                      )

    assert tb3.column_count == tb4.column_count == tb4.column_count == 8

    if rank == 0:
        assert tb3.row_count == tb4.row_count == tb5.row_count == 736
    if rank == 1:
        assert tb3.row_count == tb4.row_count == tb5.row_count == 592
    if rank == 2:
        assert tb3.row_count == tb4.row_count == tb5.row_count == 528
    if rank == 3:
        assert tb3.row_count == tb4.row_count == tb5.row_count == 688

    # tb5: Table = tb1.distributed_join(ctx, table=tb2,
    #                       join_type=configs['join_type'],
    #                       algorithm=configs['algorithm'],
    #                       on=[0]
    #                       )
    #
    # tb5.show()

    #ctx.finalize()

test_single_process()

test_multi_process()
