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

import cupy as cp
import pycylon as cy
import pygcylon as gcy

env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
print("CylonContext Initialized: My rank: ", env.rank)

start = 100 * env.rank
df = gcy.DataFrame({'first': cp.random.randint(start, start + 10, 10),
                   'second': cp.random.randint(start, start + 10, 10)})
print("initial df from rank: ", env.rank, "\n", df)

shuffledDF = df.shuffle(on="first", ignore_index=True, env=env)

print("shuffled df from rank: ", env.rank, "\n", shuffledDF)

env.finalize()
print("after finalize from the rank:", env.rank)
