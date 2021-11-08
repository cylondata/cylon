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
running test case
>>  mpirun --mca opal_cuda_support 1 -n 4 -quiet python -m pytest --with-mpi -q python/pygcylon/test/test_comms.py
'''

import pytest
import cudf
import pycylon as cy
import pygcylon as gcy


@pytest.mark.mpi
def test_repartition():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonContext Initialized: My rank: ", env.rank)

    input_file = "data/mpiops/sales_nulls_nunascii_" + str(env.rank) + ".csv"
    initial_sizes = [[0,0,0,0],
                     [1,1,1,1],
                     [10,0,0,0],
                     [10,10,10,10],]
    repart_sizes = [[0,0,0,0],
                    [4,0,0,0],
                    [0,0,0,10],
                    [0,20,20,0],]
    initial_offsets = [0,1,10,0]

    df = gcy.DataFrame.from_cudf(cudf.read_csv(input_file))
    for init_sizes, part_sizes, init_offset in zip(initial_sizes, repart_sizes, initial_offsets):
        df1 = df[init_offset:(init_offset + init_sizes[env.rank])]
        reparted_df = gcy.comms.repartition(df1, env=env, rows_per_partition=part_sizes)
        assert len(reparted_df) == part_sizes[env.rank], \
            "Repartitioned DataFrame row count does not match requested row count"


@pytest.mark.mpi
def test_gather():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonContext Initialized: My rank: ", env.rank)

    input_file = "data/mpiops/sales_nulls_nunascii_" + str(env.rank) + ".csv"
    initial_sizes = [[0,0,0,0],
                     [1,1,1,1],
                     [10,0,0,0],
                     [10,10,10,10],]
    initial_offsets = [0,1,10,0]
    gather_roots = [3,1,0,2]

    df = gcy.DataFrame.from_cudf(cudf.read_csv(input_file))
    for init_sizes, init_offset, gather_root in zip(initial_sizes, initial_offsets, gather_roots):
        df1 = df[init_offset:(init_offset + init_sizes[env.rank])]
        gathered_df = gcy.comms.gather(df1, env=env, gather_root=gather_root)
        assert len(gathered_df) == sum(init_sizes) if env.rank == gather_root else True, \
            "Gathered DataFrame row count does not match the sum of row counts of gathered DataFrames"


@pytest.mark.mpi
def test_allgather():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonContext Initialized: My rank: ", env.rank)

    input_file = "data/mpiops/sales_nulls_nunascii_" + str(env.rank) + ".csv"
    initial_sizes = [[0,0,0,0],
                     [1,1,1,1],
                     [10,0,0,0],
                     [10,10,10,10],]
    initial_offsets = [0,1,10,0]

    df = gcy.DataFrame.from_cudf(cudf.read_csv(input_file))
    for init_sizes, init_offset in zip(initial_sizes, initial_offsets):
        df1 = df[init_offset:(init_offset + init_sizes[env.rank])]
        gathered_df = gcy.comms.allgather(df1, env=env)
        assert len(gathered_df) == sum(init_sizes), \
            "AllGathered DataFrame row count does not match the sum of row counts of all gathered DataFrames"


@pytest.mark.mpi
def test_broadcast():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonContext Initialized: My rank: ", env.rank)

    input_file = "data/mpiops/sales_nulls_nunascii_" + str(env.rank) + ".csv"
    initial_sizes = [[0,0,0,0],
                     [1,1,1,1],
                     [10,0,0,0],
                     [10,10,10,10],]
    initial_offsets = [0,1,10,0]
    bcast_roots = [0,1,0,3]

    df = gcy.DataFrame.from_cudf(cudf.read_csv(input_file))
    for init_sizes, init_offset, bcast_root in zip(initial_sizes, initial_offsets, bcast_roots):
        df1 = df[init_offset:(init_offset + init_sizes[env.rank])]
        bcasted_df = gcy.comms.broadcast(df1, env=env, root=bcast_root)
        assert len(bcasted_df) == init_sizes[bcast_root], \
            "Broadcasted DataFrame row count does not match the received DataFrame row count"



#    env.finalize()
