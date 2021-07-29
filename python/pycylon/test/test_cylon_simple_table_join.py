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
Run test:
>> mpirun -n 4 python -m pytest --with-mpi -q python/test/test_cylon_simple_table_join.py
'''

import os
import pytest
from pycylon import Table
from pycylon import CylonContext
from pycylon.net import MPIConfig
from pycylon.io import CSVReadOptions
from pycylon.io import read_csv


@pytest.mark.mpi
def test_distributed_run():
    mpi_config = MPIConfig()
    ctx: CylonContext = CylonContext(config=mpi_config, distributed=True)

    table1_path = '/tmp/user_usage_tm_1.csv'
    table2_path = '/tmp/user_device_tm_1.csv'

    assert os.path.exists(table1_path)
    assert os.path.exists(table2_path)

    csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30)

    tb1: Table = read_csv(ctx, table1_path, csv_read_options)

    tb2: Table = read_csv(ctx, table2_path, csv_read_options)

    configs = {'join_type': 'inner', 'algorithm': 'sort'}

    tb3: Table = tb1.distributed_join(table=tb2,
                                      join_type=configs['join_type'],
                                      algorithm=configs['algorithm'],
                                      left_on=[3],
                                      right_on=[0]
                                      )
    row_count = tb3.row_count
    column_count = tb3.column_count

    assert ctx.get_world_size() == 4
    assert column_count == 8

    rank = ctx.get_rank()
    if rank == 0:
        assert row_count == 640
    elif rank == 1:
        assert row_count == 624
    elif rank == 2:
        assert row_count == 592
    elif rank == 3:
        assert row_count == 688
    else:
        raise Exception("Parallelism not supported in this test")

    # Note: Not needed when using PyTest with MPI
    #ctx.finalize()


