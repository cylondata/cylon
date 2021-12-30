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
>> mpirun -n 2 python -m pytest --with-mpi -q python/pycylon/test/test_cylon_table_conversion.py
"""

import os
import pytest
from pycylon.io import read_csv
from pycylon import Table
from pycylon.net import MPIConfig
from pycylon import CylonContext
from pycylon.io import CSVReadOptions
import pandas as pd
import numpy as np


@pytest.mark.mpi
def test_conversion_check():
    mpi_config = MPIConfig()
    ctx: CylonContext = CylonContext(config=mpi_config, distributed=True)

    rank, size = ctx.get_rank(), ctx.get_world_size()
    assert 0 <= rank < size and size > 0

    assert size == 2

    table1_path = f'data/input/user_usage_tm_{rank + 1}.csv'
    table2_path = f'data/input/user_device_tm_{rank + 1}.csv'

    assert os.path.exists(table1_path)
    assert os.path.exists(table2_path)

    csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30)

    tb1: Table = read_csv(ctx, table1_path, csv_read_options)

    tb2: Table = read_csv(ctx, table2_path, csv_read_options)

    tb3: Table = tb1.distributed_join(table=tb2, join_type='inner', algorithm='sort', left_on=[3],
                                      right_on=[0])

    # pdf: pd.DataFrame = tb3.to_pandas()
    npy: np.ndarray = tb3.to_numpy(order='C')

    # Cylon table rows must be equal to the rows of pandas dataframe extracted from the table
    # assert tb3.rows == pdf.shape[0]
    # Cylon table columns must be equal to the columns of pandas dataframe extracted from the table
    # assert tb3.columns == pdf.shape[1]
    # Cylon table rows must be equal to the rows of numpy ndarray extracted from the table
    assert tb3.row_count == npy.shape[0]
    # Cylon table columns must be equal to the columns of numpy ndarray extracted from the table
    assert tb3.column_count == npy.shape[1]

    print(f"Rank[{ctx.get_rank()}]: Table.Rows={tb3.row_count}, Table.Columns={tb3.column_count}, "
          f"Numpy Array Shape = {npy.shape}")

    print(f"Array Config Rank[{ctx.get_rank()}], {npy.flags} {npy.dtype}")

    # Note: Not needed when using PyTest with MPI
    # ctx.finalize()
