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
# running this example with 2 mpi workers (-n 2) on the local machine:
mpirun -n 2 --mca opal_cuda_support 1 \
    python python/pygcylon/examples/concat.py

# running this example with ucx enabled:
mpirun -n 2 --mca opal_cuda_support 1 \
    --mca pml ucx --mca osc ucx \
    python python/pygcylon/examples/concat.py

# running this example with ucx and infiniband enabled:
mpirun -n 2 --mca opal_cuda_support 1 \
    --mca pml ucx --mca osc ucx \
    --mca btl_openib_allow_ib true \
    python python/pygcylon/examples/concat.py
'''

import cupy as cp
import pycylon as cy
import pygcylon as gcy


def dist_concat():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    df1 = gcy.DataFrame({'first': cp.random.randint(0, 10, 5), 'second': cp.random.randint(100, 110, 5)})
    df2 = gcy.DataFrame({'second': cp.random.randint(100, 110, 5), 'first': cp.random.randint(0, 10, 5)})
    print(df1)
    print(df2)
    df3 = gcy.concat([df1, df2], join="inner", env=env)
    print("distributed concated df:\n", df3)
    env.finalize()


def drop_cuplicates():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)
    df1 = gcy.DataFrame({'first': cp.random.randint(100, 110, 20), 'second': cp.random.randint(100, 110, 20)})
    print("df1: \n", df1)
    df2 = df1.drop_duplicates(ignore_index=True, env=env)
    print("duplicates dropped: \n", df2) if df2 else print("duplicates dropped: \n", df1)
    env.finalize()


def local_concat():
    df1 = gcy.DataFrame({
        'name': ["John", "Smith"],
        'age': [44, 55],
    })
    df2 = gcy.DataFrame({
        'age': [44, 66],
        'name': ["John", "Joseph"],
    })
    print("df1: \n", df1)
    print("df2: \n", df2)
    df3 = gcy.concat(df1, df2)
    print("locally set difference: \n", df3)


def set_index():
    df1 = gcy.DataFrame({'first': cp.random.randint(100, 110, 20), 'second': cp.random.randint(100, 110, 20)})
    print("df1: \n", df1)
    df2 = df1.set_index("first")
    print("set index to first: \n")
    print(df2)
    df3 = df2.reset_index()
    print("index reset: \n", df3)


#####################################################
# distributed join
dist_concat()

# drop duplicates test
# drop_cuplicates()

# set index
# set_index()