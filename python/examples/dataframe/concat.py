from pycylon import DataFrame, CylonEnv
from pycylon.net import MPIConfig
import random

df1 = DataFrame([random.sample(range(10, 100), 50),
                 random.sample(range(10, 100), 50)])
df2 = DataFrame([random.sample(range(10, 100), 50),
                 random.sample(range(10, 100), 50)])
df3 = DataFrame([random.sample(range(10, 100), 50),
                 random.sample(range(10, 100), 50)])

# local unique
df4 = df1.concat(axis=0, objs=[df2, df3])
print("Local Unique")
print(df4)

# distributed unique
env = CylonEnv(config=MPIConfig())

if env.world_size > 1:
    df1 = DataFrame([random.sample(range(10*env.rank, 15*(env.rank+1)), 5),
                     random.sample(range(10*env.rank, 15*(env.rank+1)), 5)])
    df2 = DataFrame([random.sample(range(10*env.rank, 15*(env.rank+1)), 5),
                     random.sample(range(10*env.rank, 15*(env.rank+1)), 5)])
    df3 = DataFrame([random.sample(range(10*env.rank, 15*(env.rank+1)), 5),
                     random.sample(range(10*env.rank, 15*(env.rank+1)), 5)])
    print("Distributed Unique", env.rank)
    df4 = df1.concat(axis=0, objs=[df2, df3], env=env)
    print(df4)
