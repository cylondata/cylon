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

from pycylon import DataFrame, CylonEnv
from pycylon.net import MPIConfig
import random

df1 = DataFrame([random.sample(range(10, 100), 50),
                 random.sample(range(10, 100), 50)])
df2 = DataFrame([random.sample(range(10, 100), 50),
                 random.sample(range(10, 100), 50)])

# local merge
df3 = df1.merge(right=df2, on=[0])
print("Local Merge")
print(df3)

# distributed join
env = CylonEnv(config=MPIConfig())

df1 = DataFrame([random.sample(range(10*env.rank, 15*(env.rank+1)), 5),
                 random.sample(range(10*env.rank, 15*(env.rank+1)), 5)])
df2 = DataFrame([random.sample(range(10*env.rank, 15*(env.rank+1)), 5),
                 random.sample(range(10*env.rank, 15*(env.rank+1)), 5)])
print("Distributed Merge")
df3 = df1.merge(right=df2, on=[0], env=env)
print(df3)

env.finalize()
