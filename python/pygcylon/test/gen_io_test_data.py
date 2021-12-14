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
>>  mpirun --mca opal_cuda_support 1 -n 4 python python/pygcylon/test/gen_io_test_data.py
'''

import pycylon as cy
import pygcylon as gcy


def gen_parquet_test_data():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    input_files = "data/mpiops/sales_nulls_nunascii_*.csv"
    dir_path = "data/parquet"

    df = gcy.read_csv(paths=input_files, env=env, parse_dates=["Order Date"], infer_datetime_format=True)
    outfile = df.to_parquet(dir_path=dir_path, env=env, index=False, date_format='%Y%m%d')

    print(env.rank, " written to the parquet file: ", outfile)
    env.finalize()


#####################################################
gen_parquet_test_data()