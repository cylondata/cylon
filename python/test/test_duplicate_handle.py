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

import pytest
import pandas as pd
import pycylon as cn
from pycylon.net import MPIConfig
from pycylon.io import CSVReadOptions
from pycylon.io import read_csv

'''
Run test:
>> mpirun -n 2 python -m pytest --with-mpi -q python/test/test_duplicate_handle.py
'''

@pytest.mark.mpi
def test_shuffle():
    # only testing the functionality
    # accuracy was tested via shuffle op and unique test cases in C++ test cases
    mpi_config = MPIConfig()

    ctx = cn.CylonContext(config=mpi_config, distributed=True)

    tb: cn.Table = None

    rank = ctx.get_rank()
    size = ctx.get_world_size()

    assert size == 2

    if rank == 0:
        tb = cn.Table.from_pydict(ctx, {'c1': [1, 1, 3, 3, 4, 5], 'c2': [2, 2, 2, 4, 6, 6],
                                        'c3': [3, 3,
                                               3, 5,
                                               7,
                                               7]})

    if rank == 1:
        tb = cn.Table.from_pydict(ctx, {'c1': [5, 1, 1, 4, 1, 10], 'c2': [6, 2, 1, 5, 0, 1],
                                        'c3': [7, 3,
                                               0, 5,
                                               1,
                                               5]})

    tb = tb.distributed_unique(['c1', 'c2', 'c3'])

    # NOTE: do not call ctx.finalize() since a pytest is doing a finalize call


def test_unique():

    ctx = cn.CylonContext(config=None, distributed=False)
    csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30)
    table_path = '/tmp/duplicate_data_0.csv'
    tb1: cn.Table = read_csv(ctx, table_path, csv_read_options)
    pdf: pd.DataFrame = tb1.to_pandas()

    expected_indices_of_sort_col = [1, 2, 3, 4, 5, 7, 10, 12, 13, 14, 15]

    tb2 = tb1.unique(columns=['a', 'b'], keep='first')
    pdf2 = pdf.drop_duplicates(subset=['a', 'b'])
    tb2.show()
    sort_col = tb2.sort(3).to_pydict()['d']

    assert sort_col == expected_indices_of_sort_col

    assert pdf2['d'].values.tolist() == sort_col

    ctx.finalize()