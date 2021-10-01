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
    python python/pygcylon/examples/groupby.py

# running this example with ucx enabled:
mpirun -n 2 --mca opal_cuda_support 1 \
    --mca pml ucx --mca osc ucx \
    python python/pygcylon/examples/groupby.py

# running this example with ucx and infiniband enabled:
mpirun -n 2 --mca opal_cuda_support 1 \
    --mca pml ucx --mca osc ucx \
    --mca btl_openib_allow_ib true \
    python python/pygcylon/examples/groupby.py
'''

import pycylon as cy
import pygcylon as gcy


def local_groupby():
    df = gcy.DataFrame({'a': [1, 1, 1, 2, 2], 'b': [1, 1, 2, 2, 3], 'c': [1, 2, 3, 4, 5]})
    print("df: \n", df)

    gby = df.groupby("a")
    print("df grouped-by on column 'a', performed 'sum': \n", gby.sum())
    print("performed 'max' on the same groupby object: \n", gby.max())
    print("performed 'sum' on the same groupby object, aggregated on the column 'b' only: \n", gby["b"].sum())
    print("performed 'sum' on the same groupby object, aggregated on the column 'c' only: \n", gby["c"].sum())
    print("performed 'mean' on the same groupby object: \n", gby.mean())
    print("iterate through the groups: ")
    for name, gr in gby:
        print("group: ", name)
        print(gr)

    print("iterate through the group indices: ")
    for key in gby.groups:
        print("group key: ", key)
        print(gby.groups[key])

    gby = df.groupby(["a", "b"])
    print("df grouped-by on columns 'a' and 'b', performed 'sum': \n", gby.sum())
    print("performed 'max' on the same groupby object: \n", gby.max())


def dist_groupby():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    if env.rank == 0:
        df = gcy.DataFrame({
            'a': [1, 1, 2],
            'b': [1, 2, 3],
            'c': [1, 3, 5]
        })
        print("df on rank 0: \n", df)
    elif env.rank == 1:
        df = gcy.DataFrame({
            'a': [1, 2, 3],
            'b': [1, 2, 4],
            'c': [2, 4, 6]
        })
        print("df on rank 1: \n", df)

    gby = df.groupby("a", env=env)
    print("df grouped-by on column 'a', performed 'sum': \n", gby.sum())
    print("performed 'max' on the same groupby object: \n", gby.max())
    print("performed 'sum' on the same groupby object, aggregated on the column 'b' only: \n", gby["b"].sum())
    print("performed 'mean' on the same groupby object: \n", gby.mean())
    print("sizes of each group: \n", gby.size())

    gby = df.groupby(["a", "b"], env=env)
    print("df grouped-by on columns a and b, performed 'sum': \n", gby.sum())
    print("performed 'max' on the same groupby object: \n", gby.max())

    # groupby on index column with "level" parameter
    df1 = df.set_index("a")
    gby = df1.groupby(level="a", env=env)
    print("df grouped-by on index 'a', performed 'sum': \n", gby.sum())
    print("performed 'max' on the same groupby object: \n", gby.max())

    # if the original dataframe has many columns and
    # we only want to perform the groupby on some columns only,
    # the best way is to create a new dataframe with a subset of columns and
    # perform the groupby on this new dataframe
    df2 = df[["a", "b"]]
    print("two columns projected dataframe:\n", df2)
    gby = df2.groupby("a", env=env)
    print("grouped-by on column 'a' of projected df, performed 'sum': \n", gby.sum())
    print("performed 'max' on the same groupby object: \n", gby.max())

    env.finalize()


#####################################################

# local_groupby()
dist_groupby()