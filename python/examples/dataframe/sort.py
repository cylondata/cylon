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
df3 = df1.sort_values(by=[0], sort_options={'num_bins': bins, 'num_samples': bins}, env=env)
print(df3)

env.finalize()
