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
>>  mpirun --mca opal_cuda_support 1 -n 4 python python/pygcylon/examples/sorting.py
'''


import cupy
import cudf
import pycylon as cy
import pygcylon as gcy
import util


def index_based_sorting_example():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    cdf = util.random_data_df(nrows=20, col_lows=[0, 100, 200, 300], col_highs=[50, 150, 250, 350])
    cdf = cdf.set_index(keys=["col-0", "col-3"])
    print("\n\ninitial dataframe from the worker: ", env.rank, "\n", cdf)

    gdf = gcy.DataFrame(cdf)
    sorted = gdf.sort_index(ascending=True, env=env)
    print("\n\nsorted dataframe from the worker: ", env.rank, "\n", sorted)

    env.finalize()


def values_based_sorting_example():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    cdf = util.random_data_df(nrows=20, col_lows=[0, 100, 200, 300], col_highs=[50, 150, 250, 350])
    print("\n\ninitial dataframe from the worker: ", env.rank, "\n", cdf)

    gdf = gcy.DataFrame(cdf)
    sorted = gdf.sort_values(by=["col-0", "col-3"], ascending=True, ignore_index=True, env=env)
    print("\n\nsorted dataframe from the worker: ", env.rank, "\n", sorted)

    env.finalize()


###########################################################################

# cupy.cuda.Device(1).use()

# index_based_sorting_example()
values_based_sorting_example()


