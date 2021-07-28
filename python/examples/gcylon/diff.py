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


def dist_diff():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    df1 = gcy.DataFrame({
        'weight': [60 + env.rank, 80 + env.rank],
        'age': [44, 55],
    })
    df2 = gcy.DataFrame({
        'weight': [60 + env.rank, 80 + env.rank],
        'age': [44, 66],
    })
    print("df1: \n", df1)
    print("df2: \n", df2)
    df3 = df1.set_difference(other=df2, env=env)
    print("df1 distributed set difference df2:\n", df3)
    df3 = df2.set_difference(other=df1, env=env)
    print("df2 distributed set difference df1:\n", df3)
#    df3 = df1.set_difference(df2, subset=["age"], env=env)
#    print("df1 distributed set difference df2 on subset=['age']: \n", df3)
    df3 = df2.set_difference(df1, subset=["age"], env=env)
    print("df2 distributed set difference df1 on subset=['age']: \n", df3)
    env.finalize()


def local_diff():
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
    df3 = df1.set_difference(df2)
    print("df1 set difference df2: \n", df3)
    df3 = df2.set_difference(df1)
    print("df2 set difference df1: \n", df3)
    df3 = df1.set_difference(df2, subset=["name"])
    print("df1 set difference df2 on subset=['name']: \n", df3)
    df3 = df2.set_difference(df1, subset=["name"])
    print("df2 set difference df1 on subset=['name']: \n", df3)


#####################################################
# local diff test
# local_diff()

# distributed diff
dist_diff()
