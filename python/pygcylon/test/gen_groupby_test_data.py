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
>>  mpirun --mca opal_cuda_support 1 -n 4 python python/pygcylon/test/gen_groupby_test_data.py
'''

import cudf
import pycylon as cy
import pygcylon as gcy


def gen_groupby_test_data():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    inputFile = "data/input/cities_a_" + str(env.rank) + ".csv"
    gbyFile1 = "data/output/groupby_sum_cities_a_" + str(env.rank) + ".csv"
    gbyFile2 = "data/output/groupby_max_cities_a_" + str(env.rank) + ".csv"

    df1 = gcy.DataFrame.from_cudf(cudf.read_csv(inputFile))
    df1 = df1[["state_id", "population"]]

    print("df1: \n", df1)
    gby = df1.groupby("state_id", env=env)
    gby.sum().to_csv(gbyFile1)
    gby.max().to_csv(gbyFile2)

    print(env.rank, " written gbyFile1 to the file: ", gbyFile1)
    print(env.rank, " written gbyFile2 to the file: ", gbyFile2)
    env.finalize()


#####################################################
gen_groupby_test_data()
