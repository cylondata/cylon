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
>>  mpirun --mca opal_cuda_support 1 -n 4 -quiet python -m pytest --with-mpi -q python/pygcylon/test/test_head_tail.py
'''

import pytest
import pycylon as cy
import pygcylon as gcy
import util


@pytest.mark.mpi
def test_row_counts():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonContext Initialized: My rank: ", env.rank)

    row_counts = [20 + i for i in range(env.world_size)]
    cdf = util.create_sorted_cudf_df(ncols=2, nrows=row_counts[env.rank])
    df = gcy.DataFrame.from_cudf(cdf)
    calculated_row_counts = df.row_counts_for_all(env=env)

    assert row_counts == calculated_row_counts, \
        "Row Counts and Calculated Row Counts are not equal"

#    env.finalize()


@pytest.mark.mpi
def test_head():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonContext Initialized: My rank: ", env.rank)

    row_counts = [20 + i for i in range(env.world_size)]
    cdf = util.create_sorted_cudf_df(ncols=2, nrows=row_counts[env.rank])
    df = gcy.DataFrame.from_cudf(cdf)

    head_n_list = [15, 30, 100]
    list_of_head_counts = [[15, 0, 0, 0], [20, 10, 0, 0], [20, 21, 22, 23]]
    for head_n, head_counts in zip(head_n_list, list_of_head_counts):
        head_df = df.head(n=head_n, env=env)
        assert head_counts[env.rank] == len(head_df), \
            "Given row_count and head_df row_count are not equal"


@pytest.mark.mpi
def test_tail():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonContext Initialized: My rank: ", env.rank)

    row_counts = [20 + i for i in range(env.world_size)]
    cdf = util.create_sorted_cudf_df(ncols=2, nrows=row_counts[env.rank])
    df = gcy.DataFrame.from_cudf(cdf)

    tail_n_list = [15, 30, 100]
    list_of_tail_counts = [[0, 0, 0, 15], [0, 0, 7, 23], [20, 21, 22, 23]]
    for tail_n, tail_counts in zip(tail_n_list, list_of_tail_counts):
        tail_df = df.tail(n=tail_n, env=env)
        assert tail_counts[env.rank] == len(tail_df), \
            "Given row_count and tail_df row_count are not equal"

