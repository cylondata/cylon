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
>> mpirun -n 4 python -m pytest -q python/pycylon/test/test_custom_mpi_comm.py
"""

from mpi4py import MPI
from pycylon.frame import CylonEnv, read_csv, DataFrame
from pycylon.net import MPIConfig
from pyarrow.csv import ReadOptions, read_csv as pa_read_csv


def join_test():
    comm = MPI.COMM_WORLD
    rank = comm.rank

    l_rank = rank % 2
    color = int(rank / 2)
    new_comm = comm.Split(color, l_rank)
    config = MPIConfig(new_comm)
    env = CylonEnv(config=config, distributed=True)

    df1 = read_csv(f"data/input/csv1_{l_rank}.csv", env=env)
    df2 = read_csv(f"data/input/csv2_{l_rank}.csv", env=env)
    out = DataFrame(pa_read_csv(f"data/output/join_inner_2_{l_rank}.csv",
                                ReadOptions(skip_rows=1,
                                            column_names=['x0', 'x1', 'y0', 'y1'])))

    res = df1.merge(df2, on=[0], suffixes=('x', 'y'), env=env)

    assert out.equals(res, ordered=False)


def custom_comm_test():
    comm = MPI.COMM_WORLD
    rank = comm.rank

    l_rank = rank if rank < 3 else rank - 3
    color = 0 if rank < 3 else 1
    new_comm = comm.Split(color, l_rank)

    config = MPIConfig(new_comm)
    env = CylonEnv(config=config, distributed=True)

    print(f"local rank {env.rank} sz {env.world_size}")
    assert l_rank == env.rank
    if color == 0:
        assert env.world_size == 3
    else:
        assert env.world_size == 1
    env.finalize()


def test_custom_mpi_comm():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    sz = comm.size
    print(f"rank {rank} sz {sz}")

    if sz != 4:
        return

    custom_comm_test()
    join_test()

    MPI.Finalize()
