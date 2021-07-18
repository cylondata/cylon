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

import random

from pycylon import DataFrame, CylonEnv
from pycylon.net import MPIConfig

df1 = DataFrame([random.sample(range(10, 100), 50),
                 random.sample(range(10, 100), 50)])

# local sort
df3 = df1.sort_values(by=[0])
print("Local Sort")
print(df3)

# distributed sort
env = CylonEnv(config=MPIConfig())

df1 = DataFrame([random.sample(range(10 * env.rank, 15 * (env.rank + 1)), 5),
                 random.sample(range(10 * env.rank, 15 * (env.rank + 1)), 5)])
print("Distributed Sort", env.rank)
df3 = df1.sort_values(by=[0], env=env)
print(df3)

# distributed sort
print("Distributed Sort with sort options", env.rank)
bins = env.world_size * 2
df3 = df1.sort_values(by=[0], num_bins=bins, num_samples=bins, env=env)
print(df3)

env.finalize()
