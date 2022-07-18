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
>> mpirun -n 4 python -m pytest -q python/pycylon/test/test_ucx_mpi.py
"""

import pandas as pd
from numpy.random import default_rng

from pycylon import CylonEnv, DataFrame
from pycylon.net.ucx_config import UCXConfig

ROWS = 5


def test_ucx_mpi():  # confirms that the code is under main function
    conf = UCXConfig()
    env = CylonEnv(config=conf)

    rng = default_rng(seed=env.rank)
    data1 = rng.integers(0, ROWS, size=(ROWS, 2))
    data2 = rng.integers(0, ROWS, size=(ROWS, 2))

    df1 = DataFrame(pd.DataFrame(data1).add_prefix("col"))
    df2 = DataFrame(pd.DataFrame(data2).add_prefix("col"))

    print("Distributed Merge")
    df3 = df1.merge(right=df2, on=[0], env=env)
    print(f'res len {len(df3)}')

    env.finalize()