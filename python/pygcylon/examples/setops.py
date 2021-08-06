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

import cudf
import pycylon as cy
import pygcylon as gcy

#
# supposed to be run by 4 workers, 2 should also work
# these functions starting with gen can generate data for running setop tests
# other functions can be executed to verify the set operations
#
# functions can be executed as:
# $ mpirun --mca opal_cuda_support 1 -n 4 python python/pygcylon/examples/setops.py
#


def gen_dist_diff_files():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    inputFile1 = "data/input/cities_" + str(env.rank) + ".csv"
    inputFile2 = "data/input/cities_setops_" + str(env.rank) + ".csv"
    diffFile1 = "data/output/diff_df1-df2_" + str(env.rank) + ".csv"
    diffFile2 = "data/output/diff_df2-df1_" + str(env.rank) + ".csv"

    df1 = gcy.DataFrame.from_cudf(cudf.read_csv(inputFile1))
    df2 = gcy.DataFrame.from_cudf(cudf.read_csv(inputFile2))

    print("df1: \n", df1)
    print("df2: \n", df2)
    diff1 = df1.set_difference(other=df2, env=env)
    diff2 = df2.set_difference(other=df1, env=env)

    diff1.to_cudf().to_csv(diffFile1, index=False)
    diff2.to_cudf().to_csv(diffFile2, index=False)
    print(env.rank, " written diff1 to the file: ", diffFile1)
    print(env.rank, " written diff2 to the file: ", diffFile2)
    env.finalize()


def dist_diff():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    inputFile1 = "data/input/cities_" + str(env.rank) + ".csv"
    inputFile2 = "data/input/cities_setops_" + str(env.rank) + ".csv"
    diffFile1 = "data/output/diff_df1-df2_" + str(env.rank) + ".csv"
    diffFile2 = "data/output/diff_df2-df1_" + str(env.rank) + ".csv"

    df1 = gcy.DataFrame.from_cudf(cudf.read_csv(inputFile1))
    df2 = gcy.DataFrame.from_cudf(cudf.read_csv(inputFile2))

    print("df1: \n", df1)
    print("df2: \n", df2)
    diff1 = df1.set_difference(other=df2, env=env)
    diff2 = df2.set_difference(other=df1, env=env)

    #  sort difference dataframes
    diff1_sorted = diff1.to_cudf().sort_values(by=["city", "state_id"], ignore_index=True)
    diff2_sorted = diff2.to_cudf().sort_values(by=["city", "state_id"], ignore_index=True)

    savedDiff1 = cudf.read_csv(diffFile1).sort_values(by=["city", "state_id"], ignore_index=True)
    savedDiff2 = cudf.read_csv(diffFile2).sort_values(by=["city", "state_id"], ignore_index=True)

    print(env.rank, " equal") if savedDiff1.equals(diff1_sorted) else print(env.rank, " not equal")
    print(env.rank, " equal") if savedDiff2.equals(diff2_sorted) else print(env.rank, " not equal")

    env.finalize()


def gen_dist_union_files():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    inputFile1 = "data/input/cities_" + str(env.rank) + ".csv"
    inputFile2 = "data/input/cities_setops_" + str(env.rank) + ".csv"
    unionFile = "data/output/union_cities_" + str(env.rank) + ".csv"

    df1 = gcy.DataFrame.from_cudf(cudf.read_csv(inputFile1))
    df2 = gcy.DataFrame.from_cudf(cudf.read_csv(inputFile2))

    print("df1: \n", df1)
    print("df2: \n", df2)
    unionDf = df1.set_union(other=df2, env=env)

    unionDf.to_cudf().to_csv(unionFile, index=False)
    print(env.rank, " written unionFile to the file: ", unionFile)

    env.finalize()


def dist_union():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    inputFile1 = "data/input/cities_" + str(env.rank) + ".csv"
    inputFile2 = "data/input/cities_setops_" + str(env.rank) + ".csv"
    unionFile = "data/output/union_cities_" + str(env.rank) + ".csv"

    df1 = gcy.DataFrame.from_cudf(cudf.read_csv(inputFile1))
    df2 = gcy.DataFrame.from_cudf(cudf.read_csv(inputFile2))

    print("df1: \n", df1)
    print("df2: \n", df2)
    unionDf = df1.set_union(other=df2, env=env)

    #  sort union dataframes
    union_sorted = unionDf.to_cudf().sort_values(by=["city", "state_id"], ignore_index=True)

    savedUnion = cudf.read_csv(unionFile).sort_values(by=["city", "state_id"], ignore_index=True)

    print(env.rank, " equal") if savedUnion.equals(union_sorted) else print(env.rank, " not equal")
    env.finalize()


def gen_dist_intersect_files():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    inputFile1 = "data/input/cities_" + str(env.rank) + ".csv"
    inputFile2 = "data/input/cities_setops_" + str(env.rank) + ".csv"
    intersectFile = "data/output/intersect_cities_" + str(env.rank) + ".csv"

    df1 = gcy.DataFrame.from_cudf(cudf.read_csv(inputFile1))
    df2 = gcy.DataFrame.from_cudf(cudf.read_csv(inputFile2))

    print("df1: \n", df1)
    print("df2: \n", df2)
    intersectDf = df1.set_intersect(other=df2, env=env)

    intersectDf.to_cudf().to_csv(intersectFile, index=False)
    print(env.rank, " written intersectFile to the file: ", intersectFile)

    env.finalize()


def dist_intersect():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    inputFile1 = "data/input/cities_" + str(env.rank) + ".csv"
    inputFile2 = "data/input/cities_setops_" + str(env.rank) + ".csv"
    intersectFile = "data/output/intersect_cities_" + str(env.rank) + ".csv"

    df1 = gcy.DataFrame.from_cudf(cudf.read_csv(inputFile1))
    df2 = gcy.DataFrame.from_cudf(cudf.read_csv(inputFile2))

    intersectDf = df1.set_intersect(other=df2, env=env)

    #  sort dataframe
    intersect_sorted = intersectDf.to_cudf().sort_values(by=["city", "state_id"], ignore_index=True)

    saved_intersect = cudf.read_csv(intersectFile).sort_values(by=["city", "state_id"], ignore_index=True)

    print(env.rank, " equal") if intersect_sorted.equals(saved_intersect) else print(env.rank, " not equal")
    env.finalize()


def gen_dist_concat_files():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    inputFile1 = "data/input/cities_" + str(env.rank) + ".csv"
    inputFile2 = "data/input/cities_setops_" + str(env.rank) + ".csv"
    concatFile = "data/output/concat_cities_" + str(env.rank) + ".csv"

    df1 = gcy.DataFrame.from_cudf(cudf.read_csv(inputFile1))
    df2 = gcy.DataFrame.from_cudf(cudf.read_csv(inputFile2))

    print("df1: \n", df1)
    print("df2: \n", df2)
    concatedDf = gcy.concat([df1, df2], env=env)

    concatedDf.to_cudf().to_csv(concatFile, index=False)
    print(env.rank, " written concatFile to the file: ", concatFile)

    env.finalize()


def dist_concat():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    inputFile1 = "data/input/cities_" + str(env.rank) + ".csv"
    inputFile2 = "data/input/cities_setops_" + str(env.rank) + ".csv"
    concatFile = "data/output/concat_cities_" + str(env.rank) + ".csv"

    df1 = gcy.DataFrame.from_cudf(cudf.read_csv(inputFile1))
    df2 = gcy.DataFrame.from_cudf(cudf.read_csv(inputFile2))

    concatedDf = gcy.concat([df1, df2], env=env)

    #  sort dataframe
    concated_sorted = concatedDf.to_cudf().sort_values(by=["city", "state_id"], ignore_index=True)

    saved_concated = cudf.read_csv(concatFile).sort_values(by=["city", "state_id"], ignore_index=True)

    print(env.rank, " equal") if concated_sorted.equals(saved_concated) else print(env.rank, " not equal")
    env.finalize()


#####################################################

# gen_dist_diff_files()
# dist_diff()

# gen_dist_union_files()
# dist_union()

# gen_dist_intersect_files()
# dist_intersect()

# gen_dist_concat_files()
dist_concat()