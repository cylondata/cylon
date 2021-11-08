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
>>  mpirun --mca opal_cuda_support 1 -n 4 -quiet python -m pytest --with-mpi -q python/pygcylon/test/test_shuffle.py
'''

import pytest
import cudf
import pycylon as cy
import pygcylon as gcy


@pytest.mark.mpi
def test_shuffle():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonContext Initialized: My rank: ", env.rank)

    input_file = "data/input/cities_a_" + str(env.rank) + ".csv"
    str_shuffle_file = "data/output/shuffle_str_cities_a_" + str(env.rank) + ".csv"
    int_shuffle_file = "data/output/shuffle_int_cities_a_" + str(env.rank) + ".csv"

    df1 = gcy.DataFrame.from_cudf(cudf.read_csv(input_file))

    str_shuffled = df1.shuffle(on="state_id", ignore_index=True, env=env)
    str_shuffled_sorted = str_shuffled.to_cudf()\
        .sort_values(by=["state_id", "city", "population"], ignore_index=True)

    int_shuffled = df1.shuffle(on="population", ignore_index=True, env=env)
    int_shuffled_sorted = int_shuffled.to_cudf()\
        .sort_values(by=["state_id", "city", "population"], ignore_index=True)

    str_shuffled_saved = cudf.read_csv(str_shuffle_file)\
        .sort_values(by=["state_id", "city", "population"], ignore_index=True)
    int_shuffled_saved = cudf.read_csv(int_shuffle_file)\
        .sort_values(by=["state_id", "city", "population"], ignore_index=True)

    assert str_shuffled_sorted.equals(str_shuffled_saved), \
        "String based Shuffled DataFrame and DataFrame from file are not equal"
    assert int_shuffled_sorted.equals(int_shuffled_saved), \
        "Integer based Shuffled DataFrame and DataFrame from file are not equal"


@pytest.mark.mpi
def test_index_shuffle():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonContext Initialized: My rank: ", env.rank)

    input_file = "data/input/cities_a_" + str(env.rank) + ".csv"
    str_shuffle_file = "data/output/shuffle_str_cities_a_" + str(env.rank) + ".csv"
    int_shuffle_file = "data/output/shuffle_int_cities_a_" + str(env.rank) + ".csv"

    df1 = gcy.DataFrame.from_cudf(cudf.read_csv(input_file, index_col="state_id"))

    str_shuffled = df1.shuffle(index_shuffle=True, env=env)
    str_shuffled_sorted = str_shuffled.to_cudf().sort_index()

    int_shuffled_sorted = df1.reset_index().set_index(keys="population")\
        .shuffle(index_shuffle=True, env=env).sort_index()

    str_shuffled_saved = cudf.read_csv(str_shuffle_file, index_col="state_id").sort_index()
    int_shuffled_saved = cudf.read_csv(int_shuffle_file, index_col="population").sort_index()

    assert str_shuffled_sorted.index.equals(str_shuffled_saved.index), \
        "String based Shuffled DataFrame and DataFrame from file are not equal"
    assert int_shuffled_sorted.to_cudf().index.equals(int_shuffled_saved.index), \
        "Integer based Shuffled DataFrame and DataFrame from file are not equal"

#    env.finalize()
