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
>>  mpirun --mca opal_cuda_support 1 -n 4 python python/pygcylon/test/gen_join_test_data.py
'''

import cudf
import pycylon as cy
import pygcylon as gcy


def gen_join_test_data():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    input_file1 = "data/input/cities_a_" + str(env.rank) + ".csv"
    input_file2 = "data/input/cities_b_" + str(env.rank) + ".csv"
    join_file = "data/output/join_cities_" + str(env.rank) + ".csv"

    df1 = gcy.DataFrame.from_cudf(cudf.read_csv(input_file1, index_col="state_id"))
    df2 = gcy.DataFrame.from_cudf(cudf.read_csv(input_file2, index_col="state_id"))

    joined_df = df1.join(other=df2, how="inner", env=env)
    joined_df.to_cudf().to_csv(join_file)

    print(env.rank, " written join_file to the file: ", join_file)
    env.finalize()


#####################################################
gen_join_test_data()
