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

from pycylon import CylonEnv
from utils import create_df, assert_eq
from pycylon.net import MPIConfig
import random

"""
Run test:
>> mpirun -n 4 python -m pytest --with-mpi -q python/pycylon/test/test_repartition.py
"""
import pytest


@pytest.mark.mpi
def test_repartition_equal_parts():
    env = CylonEnv(config=MPIConfig())
    world_sz = env.world_size
    df1, _ = create_df([random.sample(range(10, 300), 50),
                        random.sample(range(10, 300), 50),
                        random.sample(range(10, 300), 50)])
    # distributed repartition
    df2 = df1.repartition([50 for _ in range(world_sz)], None, env=env)

    # still the local partitions would be equal
    assert_eq(df1, df2)


@pytest.mark.mpi
def test_repartition():
    env = CylonEnv(config=MPIConfig())
    world_sz = env.world_size

    local_rows = 50
    global_rows = local_rows * world_sz

    df1, _ = create_df([random.sample(range(10, 300), local_rows),
                        random.sample(range(10, 300), local_rows),
                        random.sample(range(10, 300), local_rows)])
    # distributed repartition
    temp = list(range(1, world_sz + 1))
    s = sum(temp)

    part_list = [global_rows * p / s for p in temp]
    df2 = df1.repartition(part_list, None, env=env)

    assert (len(df2) == part_list[env.rank])
