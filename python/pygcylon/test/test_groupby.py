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
>>  mpirun --mca opal_cuda_support 1 -n 4 -quiet python -m pytest --with-mpi -q python/pygcylon/test/test_groupby.py
'''

import pytest
import cudf
import pycylon as cy
import pygcylon as gcy


@pytest.mark.mpi
def test_groupby():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonContext Initialized: My rank: ", env.rank)

    inputFile = "data/input/cities_a_" + str(env.rank) + ".csv"
    gbyFile1 = "data/output/groupby_sum_cities_a_" + str(env.rank) + ".csv"
    gbyFile2 = "data/output/groupby_max_cities_a_" + str(env.rank) + ".csv"

    df1 = gcy.DataFrame.from_cudf(cudf.read_csv(inputFile))
    df1 = df1[["state_id", "population"]]
    gby = df1.groupby("state_id", env=env)
    sum_df = gby.sum().sort_index()
    max_df = gby.max().sort_index()

    saved_sum_df = cudf.read_csv(gbyFile1, index_col="state_id").sort_index()
    saved_max_df = cudf.read_csv(gbyFile2, index_col="state_id").sort_index()

    assert sum_df.equals(saved_sum_df), "Groupbyed Sum DataFrame and DataFrame from file are not equal"
    assert max_df.equals(saved_max_df), "Groupbyed Maz DataFrame and DataFrame from file are not equal"
#    env.finalize()

