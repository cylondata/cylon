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
first enable any of the function by commenting out the function name at the end of the file
running test data generation script as:
>>  mpirun --mca opal_cuda_support 1 -n 4 python python/pygcylon/test/gen_setops_test_data.py
'''

import cudf
import pycylon as cy
import pygcylon as gcy


def gen_diff_files():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    inputFile1 = "data/input/cities_a_" + str(env.rank) + ".csv"
    inputFile2 = "data/input/cities_b_" + str(env.rank) + ".csv"
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


def gen_union_files():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    inputFile1 = "data/input/cities_a_" + str(env.rank) + ".csv"
    inputFile2 = "data/input/cities_b_" + str(env.rank) + ".csv"
    unionFile = "data/output/union_cities_" + str(env.rank) + ".csv"

    df1 = gcy.DataFrame.from_cudf(cudf.read_csv(inputFile1))
    df2 = gcy.DataFrame.from_cudf(cudf.read_csv(inputFile2))

    print("df1: \n", df1)
    print("df2: \n", df2)
    unionDf = df1.set_union(other=df2, env=env)

    unionDf.to_cudf().to_csv(unionFile, index=False)
    print(env.rank, " written unionFile to the file: ", unionFile)

    env.finalize()


def gen_intersect_files():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    inputFile1 = "data/input/cities_a_" + str(env.rank) + ".csv"
    inputFile2 = "data/input/cities_b_" + str(env.rank) + ".csv"
    intersectFile = "data/output/intersect_cities_" + str(env.rank) + ".csv"

    df1 = gcy.DataFrame.from_cudf(cudf.read_csv(inputFile1))
    df2 = gcy.DataFrame.from_cudf(cudf.read_csv(inputFile2))

    print("df1: \n", df1)
    print("df2: \n", df2)
    intersectDf = df1.set_intersect(other=df2, env=env)

    intersectDf.to_cudf().to_csv(intersectFile, index=False)
    print(env.rank, " written intersectFile to the file: ", intersectFile)

    env.finalize()


def gen_concat_files():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    inputFile1 = "data/input/cities_a_" + str(env.rank) + ".csv"
    inputFile2 = "data/input/cities_b_" + str(env.rank) + ".csv"
    concatFile = "data/output/concat_cities_" + str(env.rank) + ".csv"

    df1 = gcy.DataFrame.from_cudf(cudf.read_csv(inputFile1))
    df2 = gcy.DataFrame.from_cudf(cudf.read_csv(inputFile2))

    print("df1: \n", df1)
    print("df2: \n", df2)
    concatedDf = gcy.concat([df1, df2], env=env)

    concatedDf.to_cudf().to_csv(concatFile, index=False)
    print(env.rank, " written concatFile to the file: ", concatFile)

    env.finalize()


#####################################################

# gen_diff_files()
# gen_union_files()
# gen_intersect_files()
# gen_concat_files()
