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
def test_int64_based_shuffle():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonContext Initialized: My rank: ", env.rank)

    inputFile = "data/input/csv1_" + str(env.rank) + ".csv"
    shuffledFile = "data/output/shuffled_" + str(env.rank) + ".csv"

    df = gcy.DataFrame.from_cudf(cudf.read_csv(inputFile))

    shuffledDF = df.shuffle(on="0", ignore_index=True, env=env)
    shuffledSortedDF = shuffledDF.to_cudf().sort_values(by=["0", "1"], ignore_index=True)

    cdf = cudf.read_csv(shuffledFile)
    cdf = cdf.sort_values(by=["0", "1"], ignore_index=True)
    assert shuffledSortedDF.equals(cdf), "Shuffled DataFrame and DataFrame from file are not equal"
#    env.finalize()


@pytest.mark.mpi
def test_str_based_shuffle():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonContext Initialized: My rank: ", env.rank)

    inputFile = "data/input/cities_" + str(env.rank) + ".csv"
    shuffledFile = "data/output/shuffled_cities_" + str(env.rank) + ".csv"

    df = gcy.DataFrame.from_cudf(cudf.read_csv(inputFile))

    shuffledDF = df.shuffle(on="city", ignore_index=True, env=env)
    shuffledSortedDF = shuffledDF.to_cudf().sort_values(by=["city", "state_id"], ignore_index=True)

    cdf = cudf.read_csv(shuffledFile)
    cdf = cdf.sort_values(by=["city", "state_id"], ignore_index=True)
    assert shuffledSortedDF.equals(cdf), "Shuffled DataFrame and DataFrame from file are not equal"
#    env.finalize()

