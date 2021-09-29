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
>>  mpirun --mca opal_cuda_support 1 -n 4 python python/pygcylon/test/gen_sort_test_data.py
'''

import cudf
import pycylon as cy
import pygcylon as gcy


def gen_index_sorted_test_data():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    input_file = "data/gather/sales_records_nulls_nunascii_" + str(env.rank) + ".csv"
    out_file = "data/gather/sales_index_sorted_" + str(env.rank) + ".csv"

    df = gcy.DataFrame.from_cudf(cudf.read_csv(input_file,
                                               parse_dates=["Order Date"],
                                               infer_datetime_format=True))
    df = df.set_index(keys="Order Date")

    index_sorted = df.sort_index(env=env)
    index_sorted.to_cudf().to_csv(out_file, index=True, date_format='%Y%m%d')

    print(env.rank, " written index sorted DataFrame to the file: ", out_file)
    env.finalize()


#####################################################
gen_index_sorted_test_data()