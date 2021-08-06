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
>> mpirun -n 4 python -m pytest -q python/pycylon/test/test_df_dist_sorting.py
"""
import pytest
from pycylon import DataFrame, CylonEnv
from pycylon.net import MPIConfig
import random


@pytest.mark.mpi
def test_df_dist_sorting():
    df1 = DataFrame([random.sample(range(10, 30), 5),
                     random.sample(range(10, 30), 5)])

    def check_sort(df, col, ascending):
        arr = df.to_pandas()[col]
        for i in range(len(arr) - 1):
            if ascending:
                assert arr[i] <= arr[i + 1]
            else:
                assert arr[i] >= arr[i + 1]

    # local sort
    df = df1.sort_values('0', ascending=True)
    check_sort(df, '0', True)

    df = df1.sort_values('0', ascending=False)
    check_sort(df, '0', False)

    # distributed sort
    env = CylonEnv(config=MPIConfig(), distributed=True)

    print("Distributed Sort", env.rank, env.world_size)

    df3 = df1.sort_values(by=[0], env=env, ascending=True)
    check_sort(df3, '0', True)

    df3 = df1.sort_values(by=[0], env=env, ascending=False)
    check_sort(df3, '0', False)
