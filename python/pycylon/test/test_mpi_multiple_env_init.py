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
Run test
>>  mpirun -n 2 python -m pytest --with-mpi -q python/pycylon/test/test_mpi_multiple_env_init.py
"""
import pandas as pd
import pytest
from mpi4py import MPI
from numpy.random import default_rng

from pycylon import CylonEnv, DataFrame
from pycylon.net import MPIConfig
from pycylon.net.gloo_config import GlooMPIConfig
from pycylon.net.ucx_config import UCXConfig


def create_and_destroy_env(conf):
    env = CylonEnv(config=conf)
    print(f"{conf.comm_type} rank: {env.rank} world size: {env.world_size}")
    r = 10

    rng = default_rng()
    data1 = rng.integers(0, r, size=(r, 2))
    data2 = rng.integers(0, r, size=(r, 2))

    df1 = DataFrame(pd.DataFrame(data1).add_prefix("col"))
    df2 = DataFrame(pd.DataFrame(data2).add_prefix("col"))

    print(len(df1.merge(right=df2, on=[0], env=env)))
    env.finalize()


@pytest.mark.mpi
def test_mpi_multiple_env_init():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    world_sz = comm.size
    print(f"mpi rank: {rank} world size: {world_sz}")

    create_and_destroy_env(MPIConfig())
    create_and_destroy_env(GlooMPIConfig())
    create_and_destroy_env(UCXConfig())
