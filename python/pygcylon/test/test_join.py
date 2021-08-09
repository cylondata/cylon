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
>>  mpirun --mca opal_cuda_support 1 -n 4 -quiet python -m pytest --with-mpi -q python/pygcylon/test/test_join.py
'''

import pytest
import cudf
import pycylon as cy
import pygcylon as gcy


@pytest.mark.mpi
def test_join():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonContext Initialized: My rank: ", env.rank)

    input_file1 = "data/input/cities_a_" + str(env.rank) + ".csv"
    input_file2 = "data/input/cities_b_" + str(env.rank) + ".csv"
    join_file = "data/output/join_cities_" + str(env.rank) + ".csv"

    df1 = gcy.DataFrame.from_cudf(cudf.read_csv(input_file1, index_col="state_id"))
    df2 = gcy.DataFrame.from_cudf(cudf.read_csv(input_file2, index_col="state_id"))

    joined_df = df1.join(other=df2, how="inner", env=env)
    joined_sorted = joined_df.to_cudf() \
        .sort_values(by=["cityl", "populationl", "cityr", "populationr"])

    saved_sorted = cudf.read_csv(join_file, index_col="state_id") \
        .sort_values(by=["cityl", "populationl", "cityr", "populationr"])

    assert len(joined_sorted) == len(saved_sorted)
    assert joined_sorted.equals(saved_sorted), \
        "Joined DataFrame and DataFrame from file are not equal"
#    env.finalize()

