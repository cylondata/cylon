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
>> mpirun -n 4 pytest -q python/pycylon/test/test_custom_mpi_comm.py --comm [mpi/gloo-mpi/ucx]
"""
import os

from mpi4py import MPI
from pycylon.frame import CylonEnv, read_csv, DataFrame
from pycylon.net import MPIConfig
from pyarrow.csv import ReadOptions, read_csv as pa_read_csv


def get_comm_config(comm_str, new_comm):
    if comm_str == 'mpi':
        return MPIConfig(comm=new_comm)

    if os.environ.get('CYLON_GLOO') and comm_str == 'gloo-mpi':
        from pycylon.net.gloo_config import GlooMPIConfig
        return GlooMPIConfig(comm=new_comm)

    if os.environ.get('CYLON_UCC') and comm_str == 'ucx':
        from pycylon.net.ucx_config import UCXConfig
        return UCXConfig(comm=new_comm)

    raise ValueError(f'unknown comm string {comm_str}')


def join_test(comm_str):
    comm = MPI.COMM_WORLD
    rank = comm.rank

    l_rank = rank % 2
    color = int(rank / 2)
    new_comm = comm.Split(color, l_rank)
    config = get_comm_config(comm_str, new_comm)
    env = CylonEnv(config=config, distributed=True)

    df1 = read_csv(f"data/input/csv1_{l_rank}.csv", env=env)
    df2 = read_csv(f"data/input/csv2_{l_rank}.csv", env=env)
    out = DataFrame(pa_read_csv(f"data/output/join_inner_2_{l_rank}.csv",
                                ReadOptions(skip_rows=1,
                                            column_names=['x0', 'x1', 'y0', 'y1'])))

    res = df1.merge(df2, on=[0], suffixes=('x', 'y'), env=env)

    assert out.equals(res, ordered=False)


def custom_comm_test(comm_str):
    comm_ = MPI.COMM_WORLD
    rank = comm_.rank

    l_rank = rank if rank < 3 else rank - 3
    color = 0 if rank < 3 else 1
    new_comm = comm_.Split(color, l_rank)

    config = get_comm_config(comm_str, new_comm)
    env = CylonEnv(config=config, distributed=True)

    print(f"local rank {env.rank} sz {env.world_size}")
    assert l_rank == env.rank
    if color == 0:
        assert env.world_size == 3
    else:
        assert env.world_size == 1
    env.finalize()


def test_custom_mpi_comm(comm):
    comm_ = MPI.COMM_WORLD
    rank = comm_.rank
    sz = comm_.size
    print(f"comm {comm} rank {rank} sz {sz}")

    if sz != 4:
        return

    custom_comm_test(comm)
    join_test(comm)

    MPI.Finalize()
