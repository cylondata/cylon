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

import pycylon as cy
import pygcylon as gcy

def local_union():
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
    df3 = df1.set_union(df2)
    print("set union: \n", df3)
    df3 = df1.set_union(df2, keep_duplicates=True)
    print("set union with duplicates: \n", df3)


def dist_union():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    df1 = gcy.DataFrame({
        'weight': [60 + env.rank, 80 + env.rank],
        'age': [44, 55],
    })
    df2 = gcy.DataFrame({
        'age': [44, 66],
        'weight': [60 + env.rank, 80 + env.rank],
    })
    print(df1)
    print(df2)
    df3 = df1.set_union(other=df2, env=env)
    print("distributed set union:\n", df3)

    df3 = df1.set_union(other=df2, keep_duplicates=True, ignore_index=True, env=env)
    print("distributed set union with duplicates:\n", df3)
    env.finalize()


#####################################################
# local_union()

dist_union()
