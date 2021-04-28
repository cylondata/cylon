import random

import pyarrow as pa

from pycylon import DataFrame, CylonEnv
from pycylon.net import MPIConfig

t1 = pa.table([random.sample(range(10, 100), 5),
               random.sample(range(10, 100), 5)], names=['0', '0'])

df1 = DataFrame([random.sample(range(10, 100), 5),
                 random.sample(range(10, 100), 5)])
df2 = DataFrame([random.sample(range(10, 100), 5),
                 random.sample(range(10, 100), 5)])
df3 = DataFrame([random.sample(range(10, 100), 10),
                 random.sample(range(10, 100), 10)])

# local unique
df4 = DataFrame.concat(axis=0, objs=[df2, df3])
print("Local concat axis0")
print(df4)

df3.rename(['00', '11'])
df4 = df1.concat(axis=1, objs=[df2, df3])
print("Local concat axis1")
print(df4)

# distributed unique
env = CylonEnv(config=MPIConfig())

df1 = DataFrame([random.sample(range(10 * env.rank, 15 * (env.rank + 1)), 5),
                 random.sample(range(10 * env.rank, 15 * (env.rank + 1)), 5)])
df2 = DataFrame([random.sample(range(10 * env.rank, 15 * (env.rank + 1)), 5),
                 random.sample(range(10 * env.rank, 15 * (env.rank + 1)), 5)])
df3 = DataFrame([random.sample(range(10 * env.rank, 15 * (env.rank + 1)), 10),
                 random.sample(range(10 * env.rank, 15 * (env.rank + 1)), 10)])
print("Distributed concat axis0", env.rank)
df4 = df1.concat(axis=0, objs=[df2, df3], env=env)
print(df4)

df3.rename(['00', '11'])
df4 = df1.concat(axis=1, objs=[df2, df3], env=env)
print("Distributed concat axis1", env.rank)
print(df4)

env.finalize()
