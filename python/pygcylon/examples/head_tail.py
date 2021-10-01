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

"""
running the example
>>  mpirun --mca opal_cuda_support 1 -n 4 python python/pygcylon/examples/head_tail.py
"""

import pycylon as cy
import pygcylon as gcy
import util


def head_example():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    cdf = util.random_data_df(nrows=20 + env.rank, col_lows=[0, 100, 200, 300], col_highs=[50, 150, 250, 350])
    print(env.rank, ", number of rows: ", len(cdf))

    gdf = gcy.DataFrame(cdf)
    head_n = 30
    head_df = gdf.head(head_n, env=env)
    print(env.rank, ", created head: ", head_n, ", number of rows in the head_df: ", len(head_df))
    env.finalize()


def tail_example():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    cdf = util.random_data_df(nrows=20 + env.rank, col_lows=[0, 100, 200, 300], col_highs=[50, 150, 250, 350])
    print(env.rank, ", number of rows: ", len(cdf))

    gdf = gcy.DataFrame(cdf)
    tail_n = 30
    tail_df = gdf.tail(tail_n, env=env)
    print(env.rank, ", created tail:", tail_n, ", number of rows in the tail_df: ", len(tail_df))
    env.finalize()

###########################################################################

# cupy.cuda.Device(1).use()

head_example()
# print("-------------------------------------------------------------------------")
# tail_example()

