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
>> mpirun -n 4 python -m pytest --with-mpi -q python/pycylon/test/test_aggregate.py
"""

import pandas as pd
import numpy as np
import pytest

import pycylon as cn
from pycylon.data.aggregates import *
from pycylon import MPIConfig


@pytest.mark.mpi
def test_table_aggregates():
    ctx: CylonContext = CylonContext(config=MPIConfig(), distributed=True)
    rank, size = ctx.get_rank(), ctx.get_world_size()

    pdf = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    table = cn.Table.from_pandas(context=ctx, df=pdf)

    g_table = pd.concat([pdf] * size)  # global table

    test_methods = [
        ("sum", sum_table, g_table.sum),
        ("min", min_table, g_table.min),
        ("max", max_table, g_table.max),
        ("mean", mean_table, g_table.mean),
        # ("count", count_table, g_table.count),
        ("mean", mean_table, g_table.mean),
        ("var", var_table, g_table.var),
        ("std", std_table, g_table.std),
    ]

    for fn_name, cylon_fn, pandas_fn in test_methods:
        col = cylon_fn(ctx, table)
        exp = pandas_fn().to_list()

        print(f'{fn_name} got: ', col.data.type, col.data.tolist(), 'exp: ', exp)

        np.testing.assert_array_almost_equal(exp, col.data.tolist())
