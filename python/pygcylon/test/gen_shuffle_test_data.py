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
>>  mpirun --mca opal_cuda_support 1 -n 4 python python/pygcylon/test/gen_shuffle_test_data.py
'''

import cudf
import pycylon as cy
import pygcylon as gcy


def gen_shuffle_test_data():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    input_file = "data/input/cities_a_" + str(env.rank) + ".csv"
    str_shuffle_file = "data/output/shuffle_str_cities_a_" + str(env.rank) + ".csv"
    int_shuffle_file = "data/output/shuffle_int_cities_a_" + str(env.rank) + ".csv"

    df1 = gcy.DataFrame.from_cudf(cudf.read_csv(input_file))

    str_shuffled = df1.shuffle(on="state_id", ignore_index=True, env=env)
    str_shuffled.to_cudf().to_csv(str_shuffle_file, index=False)

    int_shuffled = df1.shuffle(on="population", ignore_index=True, env=env)
    int_shuffled.to_cudf().to_csv(int_shuffle_file, index=False)

    print(env.rank, " written gbyFile1 to the file: ", str_shuffle_file)
    print(env.rank, " written gbyFile2 to the file: ", int_shuffle_file)
    env.finalize()


#####################################################
gen_shuffle_test_data()