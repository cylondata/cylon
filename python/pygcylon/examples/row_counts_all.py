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
running the example
>>  mpirun --mca opal_cuda_support 1 -n 4 python python/pygcylon/examples/row_counts_all.py
'''


import cupy
import cudf
import pycylon as cy
import pygcylon as gcy
import util


def num_rows_all_example():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    cdf = util.random_data_df(nrows=20 + env.rank, col_lows=[0, 100, 200, 300], col_highs=[50, 150, 250, 350])
    cdf = cdf.set_index(keys=["col-0", "col-3"])
    print("\n\nnumber of rows: ", len(cdf))

    gdf = gcy.DataFrame(cdf)
    row_counts = gdf.row_counts_for_all(env=env)
    print("\n\nnumber of rows in all dataframe partitions: ", row_counts)
    if isinstance(row_counts, list):
        print("row_counts is a list. len: ", len(row_counts))
    else:
        print("row_counts is NOT a list")

    env.finalize()

###########################################################################

# cupy.cuda.Device(1).use()

num_rows_all_example()


