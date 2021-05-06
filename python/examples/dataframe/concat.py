import random

import pycylon as cn
from pycylon import DataFrame, CylonEnv
from pycylon.net import MPIConfig

df1 = DataFrame([random.sample(range(10, 100), 5),
                 random.sample(range(10, 100), 5)])
df2 = DataFrame([random.sample(range(10, 100), 5),
                 random.sample(range(10, 100), 5)])
df3 = DataFrame([random.sample(range(10, 100), 10),
                 random.sample(range(10, 100), 10)])

# local unique
df4 = cn.concat(axis=0, objs=[df1, df2, df3])
print("Local concat axis0")
print(df4)

df2.rename(['00', '11'])
df3.rename(['000', '111'])
df4 = cn.concat(axis=1, objs=[df1, df2, df3])
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
df4 = cn.concat(axis=0, objs=[df1, df2, df3], env=env)
print(df4)

df2.rename(['00', '11'])
df3.rename(['000', '111'])
df4 = cn.concat(axis=1, objs=[df1, df2, df3], env=env)
print("Distributed concat axis1", env.rank)
print(df4)

env.finalize()
