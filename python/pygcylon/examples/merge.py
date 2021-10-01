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
    python python/pygcylon/examples/merge.py

# running this example with ucx enabled:
mpirun -n 2 --mca opal_cuda_support 1 \
    --mca pml ucx --mca osc ucx \
    python python/pygcylon/examples/merge.py

# running this example with ucx and infiniband enabled:
mpirun -n 2 --mca opal_cuda_support 1 \
    --mca pml ucx --mca osc ucx \
    --mca btl_openib_allow_ib true \
    python python/pygcylon/examples/merge.py
'''

import cupy as cp
import pycylon as cy
import pygcylon as gcy


def local_merge():
    df1 = gcy.DataFrame({'first': cp.random.randint(100, 110, 5), 'second': cp.random.randint(100, 110, 5)})
    df2 = gcy.DataFrame({'first': cp.random.randint(100, 110, 5), 'second': cp.random.randint(100, 110, 5)})
    print("df1: \n", df1)
    print("df2: \n", df2)
    df3 = df1.merge(right=df2, how="left", on="first", left_index=False, right_index=False)
    print("locally merged df: \n", df3)


def dist_merge():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    df1 = gcy.DataFrame({'first': cp.random.randint(100, 110, 5), 'second': cp.random.randint(100, 110, 5)})
    df2 = gcy.DataFrame({'first': cp.random.randint(100, 110, 5), 'second': cp.random.randint(100, 110, 5)})

    print(df1)
    print(df2)
    df3 = df1.merge(right=df2, on="first", how="left", left_on=None, right_on=None, left_index=False, right_index=False, env=env)
    print("distributed joined df:\n", df3)
    env.finalize()

#####################################################
# local join test
# local_merge()

# distributed join
dist_merge()
