from pycylon import DataFrame, CylonEnv
from pycylon.net import MPIConfig
import random

df1 = DataFrame([random.sample(range(1, 5), 3),
                 random.sample(range(1, 5), 3)])

df2 = DataFrame([random.sample(range(1, 5), 3),
                 random.sample(range(1, 5), 3)])

df3 = DataFrame([random.sample(range(1, 5), 3),
                 random.sample(range(1, 5), 3)])

# local unique
df4 = df1.concat(axis=0, objs=[df2, df3])
print("Local Unique")
print(df4)

# distributed unique
env = CylonEnv(config=MPIConfig())

print("Distributed Unique", env.rank)
df4 = df1.concat(axis=0, objs=[df2, df3], env=env)
print(df4)
