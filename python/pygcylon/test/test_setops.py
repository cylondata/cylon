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
>>  mpirun --mca opal_cuda_support 1 -n 4 -quiet python -m pytest --with-mpi -q python/pygcylon/test/test_setops.py
'''

import pytest
import cudf
import pycylon as cy
import pygcylon as gcy


@pytest.mark.mpi
def test_dist_diff():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    inputFile1 = "data/input/cities_" + str(env.rank) + ".csv"
    inputFile2 = "data/input/cities_setops_" + str(env.rank) + ".csv"
    diffFile1 = "data/output/diff_df1-df2_" + str(env.rank) + ".csv"
    diffFile2 = "data/output/diff_df2-df1_" + str(env.rank) + ".csv"

    df1 = gcy.DataFrame.from_cudf(cudf.read_csv(inputFile1))
    df2 = gcy.DataFrame.from_cudf(cudf.read_csv(inputFile2))

    diff1 = df1.set_difference(other=df2, env=env)
    diff2 = df2.set_difference(other=df1, env=env)

    #  sort difference dataframes
    diff1_sorted = diff1.to_cudf().sort_values(by=["city", "state_id"], ignore_index=True)
    diff2_sorted = diff2.to_cudf().sort_values(by=["city", "state_id"], ignore_index=True)

    saved_diff1 = cudf.read_csv(diffFile1).sort_values(by=["city", "state_id"], ignore_index=True)
    saved_diff2 = cudf.read_csv(diffFile2).sort_values(by=["city", "state_id"], ignore_index=True)

    assert diff1_sorted.equals(saved_diff1), "First Difference DataFrame and the DataFrame from file are not equal"
    assert diff2_sorted.equals(saved_diff2), "Second Difference DataFrame and the DataFrame from file are not equal"
#    env.finalize()


@pytest.mark.mpi
def test_dist_union():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    inputFile1 = "data/input/cities_" + str(env.rank) + ".csv"
    inputFile2 = "data/input/cities_setops_" + str(env.rank) + ".csv"
    unionFile = "data/output/union_cities_" + str(env.rank) + ".csv"

    df1 = gcy.DataFrame.from_cudf(cudf.read_csv(inputFile1))
    df2 = gcy.DataFrame.from_cudf(cudf.read_csv(inputFile2))

    unionDf = df1.set_union(other=df2, env=env)
    union_sorted = unionDf.to_cudf().sort_values(by=["city", "state_id"], ignore_index=True)

    saved_union = cudf.read_csv(unionFile).sort_values(by=["city", "state_id"], ignore_index=True)

    assert union_sorted.equals(saved_union), "Union DataFrame and the DataFrame from file are not equal"
#    env.finalize()


@pytest.mark.mpi
def test_dist_intersect():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    inputFile1 = "data/input/cities_" + str(env.rank) + ".csv"
    inputFile2 = "data/input/cities_setops_" + str(env.rank) + ".csv"
    intersectFile = "data/output/intersect_cities_" + str(env.rank) + ".csv"

    df1 = gcy.DataFrame.from_cudf(cudf.read_csv(inputFile1))
    df2 = gcy.DataFrame.from_cudf(cudf.read_csv(inputFile2))

    intersectDf = df1.set_intersect(other=df2, env=env)
    intersect_sorted = intersectDf.to_cudf().sort_values(by=["city", "state_id"], ignore_index=True)

    saved_intersect = cudf.read_csv(intersectFile).sort_values(by=["city", "state_id"], ignore_index=True)

    assert intersect_sorted.equals(saved_intersect), "Intersect DataFrame and the DataFrame from file are not equal"
#    env.finalize()


@pytest.mark.mpi
def test_dist_concat():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    inputFile1 = "data/input/cities_" + str(env.rank) + ".csv"
    inputFile2 = "data/input/cities_setops_" + str(env.rank) + ".csv"
    concatFile = "data/output/concat_cities_" + str(env.rank) + ".csv"

    df1 = gcy.DataFrame.from_cudf(cudf.read_csv(inputFile1))
    df2 = gcy.DataFrame.from_cudf(cudf.read_csv(inputFile2))

    concatedDf = gcy.concat([df1, df2], env=env)
    concated_sorted = concatedDf.to_cudf().sort_values(by=["city", "state_id"], ignore_index=True)

    saved_concated = cudf.read_csv(concatFile).sort_values(by=["city", "state_id"], ignore_index=True)

    assert concated_sorted.equals(saved_concated), "Concatanated DataFrame and the DataFrame from file are not equal"
#    env.finalize()

